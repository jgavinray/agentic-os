use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::auth::check_auth;
use crate::context_packing::get_or_build_cached_context;
use crate::state::{
    AppState, ContextPackRequest, ContextPackResponse, TaskCategory, TaskContextConfig,
};

#[tracing::instrument(name = "handler.context_pack", skip(state, headers, req), fields(repo = %req.repo, task = %req.task))]
pub async fn context_pack(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<ContextPackRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let task_category = TaskCategory::from_task(&req.task);
    let task_config = TaskContextConfig::for_category(task_category);

    let cached = match get_or_build_cached_context(
        &state,
        &req.repo,
        &req.task,
        None,
        None,
        req.limit,
        &task_config,
        None,
        None,
    )
    .await
    {
        Ok(cached) => cached,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": "failed_to_fetch_context", "detail": e.to_string()})),
            ).into_response();
        }
    };

    axum::Json(ContextPackResponse {
        repo: req.repo,
        task: req.task,
        context: cached.context,
        memories: cached.memories,
    })
    .into_response()
}
