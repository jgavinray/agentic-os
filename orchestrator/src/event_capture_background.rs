use crate::background::spawn_bounded_background;
use crate::db;
use crate::qdrant;
use crate::state::AppState;

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

pub(crate) fn spawn_qdrant_index_event(state: &AppState, event: db::AgentEvent) {
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

pub(crate) fn spawn_trajectory_result_emit(
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
