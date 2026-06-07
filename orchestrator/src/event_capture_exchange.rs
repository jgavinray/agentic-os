use serde_json::Value;

use crate::db;
use crate::event_capture_background::spawn_feature_extraction;
use crate::state::{AppState, AppendEventRequest};

#[allow(clippy::too_many_arguments)]
pub(crate) async fn persist_exchange_with_correlation(
    state: &AppState,
    session_id: &str,
    repo: &str,
    user_content: &str,
    assistant_content: &str,
    correlation_id: Option<uuid::Uuid>,
    request_metadata: Option<Value>,
) -> Option<uuid::Uuid> {
    let make_req = |event_type: &str,
                    actor: &str,
                    content: &str,
                    parent_event_id: Option<uuid::Uuid>,
                    metadata: Option<Value>|
     -> AppendEventRequest {
        AppendEventRequest {
            session_id: session_id.to_string(),
            repo: repo.to_string(),
            actor: Some(actor.to_string()),
            event_type: event_type.to_string(),
            summary: content.chars().take(500).collect(),
            evidence: None,
            metadata,
            correlation_id,
            parent_event_id: correlation_id.and(parent_event_id),
            trajectory_id: None,
            attempt_index: None,
            event_role: None,
            task: None,
            error_type: None,
            error_description: None,
        }
    };

    if let Some(classifier) = &state.sentiment {
        if classifier.is_negative(user_content) {
            tracing::info!(
                target: "sentiment",
                session_id,
                repo,
                "negative feedback detected - storing failed_attempt event"
            );
            let req = make_req("failed_attempt", "user", user_content, None, None);
            if let Err(e) =
                db::append_event_from_request(&state.pool, &state.embedder, &state.qdrant_url, &req)
                    .await
            {
                tracing::warn!(target: "sentiment", "failed to store failed_attempt event: {e}");
            }
        }
    }

    let mut parent_event_id = None;
    let mut assistant_event_id = None;
    for (event_type, actor, content) in [
        ("user_message", "user", user_content),
        ("assistant_message", "assistant", assistant_content),
    ] {
        let metadata = if event_type == "user_message" {
            request_metadata.clone()
        } else {
            None
        };
        let req = make_req(event_type, actor, content, parent_event_id, metadata);
        for attempt in 0u32..3 {
            match db::append_event_from_request(
                &state.pool,
                &state.embedder,
                &state.qdrant_url,
                &req,
            )
            .await
            {
                Ok((event_id, false)) => {
                    tracing::warn!("{event_type} stored in postgres but not qdrant-indexed");
                    let parsed = uuid::Uuid::parse_str(&event_id).ok();
                    if event_type == "assistant_message" {
                        assistant_event_id = parsed;
                    }
                    parent_event_id = correlation_id.and(parsed);
                    break;
                }
                Ok((event_id, true)) => {
                    let parsed = uuid::Uuid::parse_str(&event_id).ok();
                    if event_type == "assistant_message" {
                        assistant_event_id = parsed;
                    }
                    parent_event_id = correlation_id.and(parsed);
                    break;
                }
                Err(e) if attempt < 2 => {
                    let delay = tokio::time::Duration::from_millis(200 * 2u64.pow(attempt));
                    tracing::debug!(
                        attempt,
                        "persist {event_type} failed, retrying in {delay:?}: {e}"
                    );
                    tokio::time::sleep(delay).await;
                }
                Err(e) => {
                    tracing::warn!("failed to persist {event_type} after 3 attempts: {e}");
                    break;
                }
            }
        }
    }

    spawn_feature_extraction(state, repo, session_id, None);
    assistant_event_id
}
