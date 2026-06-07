use serde_json::Value;

use crate::background::spawn_bounded_background;
use crate::db;
use crate::orchestration_policy;
use crate::qdrant;
use crate::state::{AppState, AppendEventRequest};

pub(crate) fn spawn_feature_extraction(
    state: &AppState,
    repo: &str,
    session_id: &str,
    trajectory_id: Option<uuid::Uuid>,
) {
    if !state.feature_extraction_enabled {
        return;
    }
    let pool = state.pool.clone();
    let metrics = state.metrics.clone();
    let repo = repo.to_string();
    let session_id = session_id.to_string();
    spawn_bounded_background(state, "feature_extraction", async move {
        crate::feature_extraction::run_inline_extraction_best_effort(
            pool,
            metrics,
            repo,
            session_id,
            trajectory_id,
        )
        .await;
    });
}

fn spawn_qdrant_index_event(state: &AppState, event: db::AgentEvent) {
    let embedder = state.embedder.clone();
    let qdrant_url = state.qdrant_url.clone();
    spawn_bounded_background(state, "qdrant_index_event", async move {
        let event_id = event.id.clone();
        let event_type = event.event_type.clone();
        if let Err(e) = qdrant::store_event(&embedder, &qdrant_url, &event).await {
            tracing::warn!(
                event_id = %event_id,
                event_type = %event_type,
                "event stored in postgres but qdrant indexing failed: {e}"
            );
        }
    });
}

fn spawn_trajectory_result_emit(
    state: &AppState,
    trajectory_id: uuid::Uuid,
    reason: crate::trajectory::BoundaryReason,
) {
    let state_bg = state.clone();
    spawn_bounded_background(state, "trajectory_result_emit", async move {
        if let Err(e) = db::emit_trajectory_result_once(
            &state_bg.pool,
            &state_bg.embedder,
            &state_bg.qdrant_url,
            trajectory_id,
            Some(reason),
        )
        .await
        {
            tracing::warn!(
                trajectory_id = %trajectory_id,
                "failed to emit trajectory result: {e}"
            );
        }
    });
}

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

#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_tool_results_background(
    state: AppState,
    session_id: String,
    repo: String,
    task: String,
    correlation_id: uuid::Uuid,
    parent_event_id: Option<uuid::Uuid>,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    tool_results: Vec<crate::execution_feedback::CapturedToolResult>,
) {
    if !state.execution_feedback_enabled || tool_results.is_empty() {
        return;
    }

    let gate_state = state.clone();
    spawn_bounded_background(&gate_state, "tool_result_capture", async move {
        let ctx = crate::execution_feedback::ExecutionEventContext {
            session_id,
            repo: repo.clone(),
            task: task.clone(),
            actor: "validator".to_string(),
            correlation_id,
            parent_event_id,
            trajectory,
        };
        for result in tool_results {
            for event in crate::execution_feedback::events_for_tool_result(&ctx, &result) {
                if let Err(e) = db::append_execution_event(
                    &state.pool,
                    &state.embedder,
                    &state.qdrant_url,
                    event,
                )
                .await
                {
                    tracing::error!(
                        target: "execution_feedback",
                        repo = %repo,
                        task = %task,
                        "failed to write execution feedback event: {e}"
                    );
                }
            }
        }
        spawn_feature_extraction(
            &state,
            &repo,
            &ctx.session_id,
            trajectory.map(|trajectory| trajectory.trajectory_id),
        );
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn maybe_write_context_pack_event(
    state: &AppState,
    session_id: Option<&str>,
    repo: &str,
    task: &str,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    parent_event_id: Option<uuid::Uuid>,
    stats: &crate::state::ContextPackStats,
    token_budget: usize,
) -> Option<uuid::Uuid> {
    if !state.trajectory_capture_enabled {
        return None;
    }
    let (Some(session_id), Some(trajectory)) = (session_id, trajectory) else {
        return None;
    };
    let event = crate::trajectory::context_pack_event(
        session_id,
        repo,
        task,
        trajectory,
        parent_event_id,
        stats,
        token_budget,
    );
    let event_id = uuid::Uuid::parse_str(&event.id).ok();
    let state_bg = state.clone();
    let repo = repo.to_string();
    let task = task.to_string();
    spawn_bounded_background(state, "context_pack_event", async move {
        if let Err(e) = db::append_execution_event(
            &state_bg.pool,
            &state_bg.embedder,
            &state_bg.qdrant_url,
            event,
        )
        .await
        {
            tracing::warn!(
                repo = %repo,
                task = %task,
                "failed to write context_pack trajectory event: {e}"
            );
        }
    });
    event_id
}
