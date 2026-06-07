use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use std::sync::Arc;

use crate::handlers::check_auth;
use crate::qdrant;
use crate::state::AppState;

#[tracing::instrument(name = "handler.search", skip(state, headers, req))]
pub async fn search(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<Value>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let query = req.get("q").and_then(|v| v.as_str()).unwrap_or("");
    let limit = req.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    let results = match qdrant::search(&state.embedder, &state.qdrant_url, query, limit).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("qdrant search failed: {e}");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(
                    serde_json::json!({"error": "search_unavailable", "detail": e.to_string()}),
                ),
            )
                .into_response();
        }
    };

    axum::Json(serde_json::json!({"results": results})).into_response()
}
