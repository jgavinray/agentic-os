use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::auth::check_auth;
use crate::db;
use crate::state::AppState;

#[derive(Debug, serde::Deserialize)]
pub struct ContextArtifactsQuery {
    pub repo: String,
    pub limit: Option<i64>,
    pub ledger_limit: Option<i64>,
}

#[tracing::instrument(name = "handler.context_artifacts", skip(state, headers))]
pub async fn context_artifacts(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<ContextArtifactsQuery>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let limit = query.limit.unwrap_or(25).clamp(1, 100);
    let ledger_limit = query.ledger_limit.unwrap_or(50).clamp(1, 200);
    let artifacts = match db::get_context_artifacts_for_repo(&state.pool, &query.repo, limit).await
    {
        Ok(artifacts) => artifacts,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({
                    "error": "failed_to_fetch_context_artifacts",
                    "detail": e.to_string()
                })),
            )
                .into_response();
        }
    };
    let ledger = match db::get_context_compiler_ledger(&state.pool, &query.repo, ledger_limit).await
    {
        Ok(ledger) => ledger,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({
                    "error": "failed_to_fetch_context_compiler_ledger",
                    "detail": e.to_string()
                })),
            )
                .into_response();
        }
    };

    axum::Json(serde_json::json!({
        "repo": query.repo,
        "artifacts": artifacts,
        "ledger": ledger,
    }))
    .into_response()
}
