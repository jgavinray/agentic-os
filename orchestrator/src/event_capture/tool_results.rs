use crate::background::spawn_bounded_background;
use crate::db;
use crate::event_capture::background::spawn_feature_extraction;
use crate::state::AppState;

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
