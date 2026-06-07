use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use std::sync::Arc;

use crate::db;
use crate::handlers::{check_auth, spawn_feature_extraction};
use crate::state::{AppState, AppendEventRequest, StartSessionRequest, StartSessionResponse};

#[tracing::instrument(name = "handler.start_session", skip(state, headers, req), fields(repo = %req.repo, task = %req.task))]
pub async fn start_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<StartSessionRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let id = match db::start_session_from_request(&state.pool, &req).await {
        Ok(id) => id,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({
                    "error": "failed_to_create_session",
                    "detail": e.to_string()
                })),
            )
                .into_response();
        }
    };

    axum::Json(StartSessionResponse { session_id: id }).into_response()
}

#[tracing::instrument(name = "handler.append_event", skip(state, headers, req), fields(repo = %req.repo, event_type = %req.event_type))]
pub async fn append_event(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(mut req): axum::Json<AppendEventRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }
    if !state.trajectory_capture_enabled {
        req.trajectory_id = None;
        req.attempt_index = None;
        req.event_role = None;
        if let Some(metadata) = req.metadata.as_mut().and_then(Value::as_object_mut) {
            metadata.remove("trajectory_id");
            metadata.remove("attempt_index");
            metadata.remove("event_role");
        }
    }

    let (event_id, qdrant_indexed) =
        match db::append_event_from_request(&state.pool, &state.embedder, &state.qdrant_url, &req)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({
                        "error": "failed_to_append_event",
                        "detail": e.to_string()
                    })),
                )
                    .into_response();
            }
        };
    spawn_feature_extraction(&state, &req.repo, &req.session_id, req.trajectory_id);

    // Detect and record errors from the event.
    if let Some(error_type) = &req.error_type {
        let error_desc = req.error_description.as_deref().unwrap_or("");
        let task = req.task.as_deref().unwrap_or("");
        let severity = "medium";
        if let Err(e) = db::insert_error_record(
            &state.pool,
            &req.repo,
            task,
            error_type,
            error_desc,
            severity,
        )
        .await
        {
            tracing::warn!(%error_type, "failed to record error: {e}");
        }
    }

    axum::Json(serde_json::json!({"event_id": event_id, "qdrant_indexed": qdrant_indexed}))
        .into_response()
}
