use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::handlers::check_auth;
use crate::state::AppState;
use crate::{db, telemetry};

#[derive(Debug, serde::Deserialize)]
pub struct CacheStatsQuery {
    pub repo: Option<String>,
    pub session_id: Option<String>,
}

#[tracing::instrument(name = "handler.cache_stats", skip(state, headers))]
pub async fn cache_stats(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<CacheStatsQuery>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let vllm = db::get_vllm_cache_stats(
        &state.pool,
        query.repo.as_deref(),
        query.session_id.as_deref(),
    )
    .await
    .unwrap_or_else(|e| {
        tracing::warn!("failed to fetch vLLM cache stats: {e}");
        db::VllmCacheStats::default()
    });
    axum::Json(serde_json::json!({
        "context_cache": state.cache.stats(),
        "vllm_prefix_cache": vllm,
        "filters": {
            "repo": query.repo,
            "session_id": query.session_id,
        }
    }))
    .into_response()
}

#[tracing::instrument(name = "handler.metrics", skip(state, headers))]
pub async fn metrics(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    telemetry::record_pool_gauges(&state.pool);
    telemetry::record_process_metrics();
    (
        telemetry::prometheus_content_type(),
        state.prometheus.render(),
    )
        .into_response()
}

#[tracing::instrument(name = "handler.metrics_json", skip(state, headers))]
pub async fn metrics_json(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    axum::Json(state.metrics.snapshot()).into_response()
}
