use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::auth::check_auth;
use crate::db;
use crate::handlers::spawn_feature_extraction;
use crate::state::{AppState, AppendEventRequest, CheckpointRequest};

#[tracing::instrument(name = "handler.checkpoint", skip(state, headers, req), fields(repo = %req.repo))]
pub async fn checkpoint(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<CheckpointRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let task_string = format!(
        "{}/{}",
        req.repo,
        req.summary.chars().take(50).collect::<String>()
    );
    let event = AppendEventRequest {
        session_id: req.session_id,
        repo: req.repo,
        actor: Some("checkpoint".to_string()),
        event_type: "checkpoint".to_string(),
        summary: req.summary,
        evidence: None,
        metadata: Some(serde_json::json!({
            "next_actions": req.next_actions.unwrap_or_default(),
            "open_questions": req.open_questions.unwrap_or_default(),
        })),
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: None,
        task: Some(task_string),
        error_type: None,
        error_description: None,
    };

    let (event_id, qdrant_indexed) = match db::append_event_from_request(
        &state.pool,
        &state.embedder,
        &state.qdrant_url,
        &event,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(
                    serde_json::json!({"error": "failed_to_checkpoint", "detail": e.to_string()}),
                ),
            )
                .into_response();
        }
    };
    spawn_feature_extraction(&state, &event.repo, &event.session_id, event.trajectory_id);

    axum::Json(serde_json::json!({"event_id": event_id, "qdrant_indexed": qdrant_indexed}))
        .into_response()
}
