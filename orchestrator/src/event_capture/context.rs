use crate::background::spawn_bounded_background;
use crate::db;
use crate::state::AppState;

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
