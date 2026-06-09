use std::sync::Arc;

use serde_json::Value;
use uuid::Uuid;

use crate::db;
use crate::event_capture::{begin_trajectory_for_request, persist_request_event};
use crate::state::AppState;
use crate::trajectory::TrajectoryContext;

pub(crate) async fn find_or_create_capture_session(
    state: &Arc<AppState>,
    repo: &str,
    task: &str,
) -> Option<String> {
    if !state.trajectory_capture_enabled {
        return None;
    }

    match db::find_or_create_session(&state.pool, repo, task, "agent").await {
        Ok(session_id) => Some(session_id),
        Err(e) => {
            tracing::warn!("find_or_create_session failed before trajectory capture: {e}");
            None
        }
    }
}

pub(crate) async fn begin_and_persist_request(
    state: &Arc<AppState>,
    session_id: Option<&str>,
    repo: &str,
    user_content: &str,
    request_metadata: Option<Value>,
) -> (Option<TrajectoryContext>, Option<Uuid>) {
    let Some(session_id) = session_id else {
        return (None, None);
    };

    let trajectory = begin_trajectory_for_request(state, session_id).await;
    let request_event_id = persist_request_event(
        state,
        session_id,
        repo,
        user_content,
        trajectory,
        request_metadata,
    )
    .await;

    (Some(trajectory), request_event_id)
}
