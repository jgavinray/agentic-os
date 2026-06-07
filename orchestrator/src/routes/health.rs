use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use std::sync::Arc;

use crate::auth::check_auth;
use crate::state::AppState;
use crate::{db, telemetry};

#[tracing::instrument(name = "handler.health")]
pub async fn health() -> axum::Json<Value> {
    axum::Json(serde_json::json!({"status": "ok"}))
}

#[tracing::instrument(name = "handler.health_live")]
pub async fn health_live() -> axum::Json<Value> {
    axum::Json(serde_json::json!({"status": "ok"}))
}

#[tracing::instrument(name = "handler.health_ready", skip(state))]
pub async fn health_ready(
    State(state): State<Arc<AppState>>,
) -> Result<axum::Json<Value>, (StatusCode, axum::Json<Value>)> {
    let http = state.http.clone();
    let mut healthy = Vec::new();
    let mut unhealthy = Vec::new();

    if db::check_ready(&state.pool).await.is_ok() {
        healthy.push("postgres");
    } else {
        unhealthy.push("postgres");
    }

    let qdrant_started = std::time::Instant::now();
    if http
        .get(format!("{}/collections", state.qdrant_url))
        .send()
        .await
        .map(|r| {
            let status = r.status();
            telemetry::record_qdrant_request(
                "health",
                qdrant_started.elapsed(),
                &status.as_u16().to_string(),
            );
            status.is_success()
        })
        .unwrap_or(false)
    {
        healthy.push("qdrant");
    } else {
        unhealthy.push("qdrant");
    }

    // LITELLM_URL already includes /v1 - no extra path segment needed here.
    if http
        .get(format!("{}/models", state.litellm_url))
        .send()
        .await
        .map(|r| r.status().is_success() || r.status() == 401)
        .unwrap_or(false)
    {
        healthy.push("litellm");
    } else {
        unhealthy.push("litellm");
    }

    if !unhealthy.is_empty() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(serde_json::json!({
                "status": "degraded",
                "healthy": healthy,
                "unhealthy": unhealthy,
            })),
        ));
    }

    Ok(axum::Json(serde_json::json!({
        "status": "ready",
        "services": healthy,
    })))
}

#[tracing::instrument(name = "handler.list_models", skip(state, headers))]
pub async fn list_models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let url = format!("{}/models", state.litellm_url);
    let started = std::time::Instant::now();
    match state
        .http
        .get(&url)
        .bearer_auth(&state.litellm_key)
        .send()
        .await
    {
        Ok(r) => {
            let status = r.status();
            telemetry::record_upstream_litellm(
                "models",
                started.elapsed(),
                &status.as_u16().to_string(),
            );
            if !status.is_success() {
                telemetry::record_upstream_litellm_error(
                    "models",
                    telemetry::upstream_error_kind(status),
                );
            }
            match r.json::<Value>().await {
                Ok(v) => (status, axum::Json(v)).into_response(),
                Err(_) => {
                    telemetry::record_upstream_litellm_error("models", "parse");
                    fallback_model_list(&state).into_response()
                }
            }
        }
        Err(e) => {
            telemetry::record_upstream_litellm("models", started.elapsed(), "error");
            telemetry::record_upstream_litellm_error("models", telemetry::reqwest_error_kind(&e));
            fallback_model_list(&state).into_response()
        }
    }
}

fn fallback_model_list(state: &AppState) -> axum::Json<Value> {
    axum::Json(serde_json::json!({
        "data": [{"id": state.default_model, "object": "model", "owned_by": "orchestrator"}],
        "object": "list"
    }))
}
