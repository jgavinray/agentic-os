use serde_json::Value;
use std::sync::Arc;
use uuid::Uuid;

use crate::event_capture::{
    capture_tool_results_background, persist_exchange_with_correlation,
    persist_model_response_event,
};
use crate::state::AppState;
use crate::trajectory::TrajectoryContext;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn persist_anthropic_trajectory_response(
    state: &Arc<AppState>,
    session_id: &str,
    repo: &str,
    task: &str,
    assistant_content: &str,
    val: &Value,
    latency_ms: i64,
    request_metadata: Option<Value>,
    context_pack_id: Option<Uuid>,
    trajectory: TrajectoryContext,
    request_event_id: Option<Uuid>,
    tool_results: Vec<crate::execution_feedback::CapturedToolResult>,
) {
    let (input_tokens, output_tokens) = crate::trajectory::optional_token_counts_from_value(val);
    let metadata = crate::trajectory::model_response_metadata(
        &state.default_model,
        "litellm",
        input_tokens,
        output_tokens,
        Some(latency_ms),
        crate::trajectory::model_finish_reason(val),
        request_metadata,
        context_pack_id,
        Some(trajectory),
    );
    let assistant_event_id = persist_model_response_event(
        state,
        session_id,
        repo,
        assistant_content,
        metadata,
        trajectory,
        request_event_id,
    )
    .await;
    capture_tool_results_background(
        state.as_ref().clone(),
        session_id.to_string(),
        repo.to_string(),
        task.to_string(),
        trajectory.trajectory_id,
        assistant_event_id,
        Some(trajectory),
        tool_results,
    );
}

pub(crate) async fn persist_anthropic_exchange(
    state: &Arc<AppState>,
    repo: &str,
    task: &str,
    user_content: &str,
    assistant_content: &str,
    correlation_id: Option<Uuid>,
    tool_results: Vec<crate::execution_feedback::CapturedToolResult>,
) {
    match crate::db::find_or_create_session(&state.pool, repo, task, "agent").await {
        Ok(session_id) => {
            let assistant_event_id = persist_exchange_with_correlation(
                state,
                &session_id,
                repo,
                user_content,
                assistant_content,
                correlation_id,
                None,
            )
            .await;
            if let Some(correlation_id) = correlation_id {
                capture_tool_results_background(
                    state.as_ref().clone(),
                    session_id,
                    repo.to_string(),
                    task.to_string(),
                    correlation_id,
                    assistant_event_id,
                    None,
                    tool_results,
                );
            }
        }
        Err(e) => tracing::warn!("messages: find_or_create_session failed: {e}"),
    }
}
