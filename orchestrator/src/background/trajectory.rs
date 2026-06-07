use std::sync::Arc;

use crate::db;
use crate::state::AppState;

pub async fn run_trajectory_idle_sweep(state: Arc<AppState>) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
    loop {
        interval.tick().await;
        if !state.trajectory_capture_enabled {
            continue;
        }
        let ids =
            match db::idle_trajectory_ids(&state.pool, state.trajectory_idle_timeout_sec, 10_000)
                .await
            {
                Ok(ids) => ids,
                Err(e) => {
                    tracing::warn!("trajectory idle sweep failed to list candidates: {e}");
                    continue;
                }
            };
        for trajectory_id in ids {
            if let Err(e) = db::emit_trajectory_result_once(
                &state.pool,
                &state.embedder,
                &state.qdrant_url,
                trajectory_id,
                Some(crate::trajectory::BoundaryReason::IdleTimeout),
            )
            .await
            {
                tracing::warn!(trajectory_id = %trajectory_id, "failed to emit idle trajectory result: {e}");
            }
        }
    }
}
