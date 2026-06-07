use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::db;
use crate::handlers::{check_auth, spawn_feature_extraction};
use crate::state::AppState;

#[tracing::instrument(name = "handler.validations", skip(state, headers, req), fields(repo = %req.repo, task = %req.task))]
pub async fn validations(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<crate::execution_feedback::ValidationReportRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    if !state.execution_feedback_enabled {
        return axum::Json(crate::execution_feedback::ValidationReportResponse {
            captured: false,
            event_ids: vec![],
            qdrant_indexed: false,
        })
        .into_response();
    }

    let session_id = match &req.session_id {
        Some(session_id) => session_id.clone(),
        None => match db::find_or_create_session(
            &state.pool,
            &req.repo,
            &req.task,
            req.actor.as_deref().unwrap_or("validator"),
        )
        .await
        {
            Ok(session_id) => session_id,
            Err(e) => {
                tracing::error!(
                    target: "execution_feedback",
                    repo = %req.repo,
                    task = %req.task,
                    "failed to create validation session: {e}"
                );
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({
                        "error": "failed_to_create_validation_session",
                        "detail": e.to_string()
                    })),
                )
                    .into_response();
            }
        },
    };

    let ctx = crate::execution_feedback::ExecutionEventContext {
        session_id,
        repo: req.repo.clone(),
        task: req.task.clone(),
        actor: req.actor.clone().unwrap_or_else(|| "validator".to_string()),
        correlation_id: req.correlation_id.unwrap_or_else(uuid::Uuid::new_v4),
        parent_event_id: req.parent_event_id,
        trajectory: if state.trajectory_capture_enabled {
            req.trajectory_id
                .map(|trajectory_id| crate::trajectory::TrajectoryContext {
                    trajectory_id,
                    attempt_index: req.attempt_index.unwrap_or(1),
                })
        } else {
            None
        },
    };
    let events = match crate::execution_feedback::events_for_validation_report(&ctx, &req) {
        Ok(events) => events,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({"error": "invalid_validation_report", "detail": e})),
            )
                .into_response();
        }
    };
    let mut event_ids = Vec::new();
    let mut qdrant_indexed = true;
    for event in events {
        match db::append_execution_event(&state.pool, &state.embedder, &state.qdrant_url, event)
            .await
        {
            Ok((event_id, indexed)) => {
                event_ids.push(event_id);
                qdrant_indexed &= indexed;
            }
            Err(e) => {
                tracing::error!(
                    target: "execution_feedback",
                    repo = %req.repo,
                    task = %req.task,
                    "failed to write validation event: {e}"
                );
            }
        }
    }
    if !event_ids.is_empty() {
        spawn_feature_extraction(
            &state,
            &req.repo,
            &ctx.session_id,
            ctx.trajectory.map(|trajectory| trajectory.trajectory_id),
        );
    }

    axum::Json(crate::execution_feedback::ValidationReportResponse {
        captured: !event_ids.is_empty(),
        event_ids,
        qdrant_indexed,
    })
    .into_response()
}
