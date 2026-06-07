use serde_json::Value;

use crate::db;
use crate::event_capture_background::{spawn_qdrant_index_event, spawn_trajectory_result_emit};
use crate::orchestration_policy;
use crate::state::{AppState, AppendEventRequest};

pub(crate) use crate::event_capture_background::spawn_feature_extraction;
pub(crate) use crate::event_capture_context::maybe_write_context_pack_event;
pub(crate) use crate::event_capture_exchange::persist_exchange_with_correlation;
pub(crate) use crate::event_capture_tool_results::capture_tool_results_background;

pub(crate) async fn begin_trajectory_for_request(
    state: &AppState,
    session_id: &str,
) -> crate::trajectory::TrajectoryContext {
    if let Ok(Some(latest)) = db::latest_trajectory_event_for_session(&state.pool, session_id).await
    {
        if let Some(trajectory_id) = latest.trajectory_id {
            let already_ended = latest.event_role.as_deref()
                == Some(crate::trajectory::EventRole::TrajectoryResult.as_str())
                || db::get_trajectory_result(&state.pool, trajectory_id)
                    .await
                    .ok()
                    .flatten()
                    .is_some();
            if !already_ended {
                let idle_for = chrono::Utc::now()
                    .signed_duration_since(latest.created_at)
                    .num_seconds()
                    .max(0) as u64;
                let reason = if idle_for > state.trajectory_idle_timeout_sec {
                    crate::trajectory::BoundaryReason::IdleTimeout
                } else {
                    crate::trajectory::BoundaryReason::NewUserMessage
                };
                spawn_trajectory_result_emit(state, trajectory_id, reason);
            }
        }
    }

    crate::trajectory::TrajectoryContext {
        trajectory_id: uuid::Uuid::new_v4(),
        attempt_index: 1,
    }
}

pub(crate) async fn persist_request_event(
    state: &AppState,
    session_id: &str,
    repo: &str,
    user_content: &str,
    trajectory: crate::trajectory::TrajectoryContext,
    request_metadata: Option<Value>,
) -> Option<uuid::Uuid> {
    let raw_capture_enabled = state.capture_pool.is_some();
    let classification = crate::request_classification::classify_request_text(
        repo,
        session_id,
        user_content,
        None,
        "user_message",
    );
    let policy = orchestration_policy::derive_orchestration_policy(
        &classification,
        user_content,
        raw_capture_enabled,
    );
    let policy_metadata = orchestration_policy::compact_policy_metadata(&classification, &policy);

    let enriched_metadata = if let Some(ref obj) = request_metadata {
        if let Value::Object(map) = obj {
            let mut enriched = map.clone();
            enriched.insert("orchestration_policy".to_string(), policy_metadata.clone());
            Some(Value::Object(enriched))
        } else {
            let mut wrapper = serde_json::Map::new();
            wrapper.insert("orchestration_policy".to_string(), policy_metadata.clone());
            wrapper.insert("original_metadata".to_string(), obj.clone());
            Some(Value::Object(wrapper))
        }
    } else {
        let mut map = serde_json::Map::new();
        map.insert("orchestration_policy".to_string(), policy_metadata.clone());
        Some(Value::Object(map))
    };

    let metadata = crate::trajectory::make_request_metadata(enriched_metadata, trajectory);
    let req = AppendEventRequest {
        session_id: session_id.to_string(),
        repo: repo.to_string(),
        actor: Some("user".to_string()),
        event_type: "user_message".to_string(),
        summary: user_content.chars().take(500).collect(),
        evidence: None,
        metadata: Some(metadata),
        correlation_id: Some(trajectory.trajectory_id),
        parent_event_id: None,
        trajectory_id: Some(trajectory.trajectory_id),
        attempt_index: Some(trajectory.attempt_index),
        event_role: Some(crate::trajectory::EventRole::Request.as_str().to_string()),
        task: None,
        error_type: None,
        error_description: None,
    };
    let event = match db::event_from_append_request(&req) {
        Ok(event) => event,
        Err(e) => {
            tracing::warn!(repo, "failed to build trajectory request event: {e}");
            return None;
        }
    };
    let event_id = uuid::Uuid::parse_str(&event.id).ok();
    match db::insert_event(&state.pool, &event).await {
        Ok(()) => {
            let classification_event = db::AgentEvent {
                metadata: event.metadata.clone(),
                ..event.clone()
            };
            let classification =
                crate::request_classification::classify_request_event(&classification_event);
            let policy = orchestration_policy::derive_orchestration_policy(
                &classification,
                user_content,
                raw_capture_enabled,
            );
            if let Err(e) = orchestration_policy::persist_orchestration_policy(
                &state.pool,
                &classification,
                &policy,
            )
            .await
            {
                tracing::warn!(
                    target: "orchestration_policy",
                    repo,
                    event_id = %event.id,
                    "failed to persist orchestration policy: {e}"
                );
            }

            spawn_qdrant_index_event(state, event);
            spawn_feature_extraction(state, repo, session_id, Some(trajectory.trajectory_id));
            event_id
        }
        Err(e) => {
            tracing::warn!(repo, "failed to persist trajectory request event: {e}");
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn persist_model_response_event(
    state: &AppState,
    session_id: &str,
    repo: &str,
    assistant_content: &str,
    metadata: Value,
    trajectory: crate::trajectory::TrajectoryContext,
    parent_event_id: Option<uuid::Uuid>,
) -> Option<uuid::Uuid> {
    let req = AppendEventRequest {
        session_id: session_id.to_string(),
        repo: repo.to_string(),
        actor: Some("assistant".to_string()),
        event_type: "assistant_message".to_string(),
        summary: assistant_content.chars().take(500).collect(),
        evidence: None,
        metadata: Some(metadata),
        correlation_id: Some(trajectory.trajectory_id),
        parent_event_id,
        trajectory_id: Some(trajectory.trajectory_id),
        attempt_index: Some(trajectory.attempt_index),
        event_role: Some(
            crate::trajectory::EventRole::ModelResponse
                .as_str()
                .to_string(),
        ),
        task: None,
        error_type: None,
        error_description: None,
    };
    match db::append_event_from_request(&state.pool, &state.embedder, &state.qdrant_url, &req).await
    {
        Ok((event_id, _)) => {
            spawn_feature_extraction(state, repo, session_id, Some(trajectory.trajectory_id));
            uuid::Uuid::parse_str(&event_id).ok()
        }
        Err(e) => {
            tracing::warn!(
                repo,
                "failed to persist trajectory model_response event: {e}"
            );
            None
        }
    }
}
