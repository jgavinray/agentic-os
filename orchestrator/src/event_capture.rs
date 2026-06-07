use serde_json::Value;

use crate::db;
use crate::state::{AppState, AppendEventRequest};

pub(crate) use crate::event_capture_background::spawn_feature_extraction;
pub(crate) use crate::event_capture_context::maybe_write_context_pack_event;
pub(crate) use crate::event_capture_exchange::persist_exchange_with_correlation;
pub(crate) use crate::event_capture_request::{
    begin_trajectory_for_request, persist_request_event,
};
pub(crate) use crate::event_capture_tool_results::capture_tool_results_background;

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
