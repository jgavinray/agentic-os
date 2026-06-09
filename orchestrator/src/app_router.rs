use axum::http::HeaderValue;
use axum::middleware;
use axum::routing::{get, post};
use axum::Router;
use std::env;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::handlers;
use crate::state::AppState;
use crate::telemetry;

pub fn build_router(state: Arc<AppState>) -> Result<Router, anyhow::Error> {
    let cors = cors_layer_from_env()?;

    Ok(Router::new()
        .route("/health", get(handlers::health))
        .route("/health/live", get(handlers::health_live))
        .route("/health/ready", get(handlers::health_ready))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/messages", post(handlers::messages))
        .route("/v1/validations", post(handlers::validations))
        .route("/tools/authorize", post(handlers::authorize_tool))
        .route("/sessions/start", post(handlers::start_session))
        .route("/events/append", post(handlers::append_event))
        .route("/harness/guardrail", post(handlers::harness_guardrail))
        .route("/harness/outcomes", post(handlers::harness_outcome))
        .route(
            "/harness/litellm-callbacks",
            post(handlers::litellm_callback_payload),
        )
        .route("/context/pack", post(handlers::context_pack))
        .route("/context/artifacts", get(handlers::context_artifacts))
        .route("/cache/stats", get(handlers::cache_stats))
        .route("/metrics", get(handlers::metrics))
        .route("/metrics/json", get(handlers::metrics_json))
        .route("/summaries/checkpoint", post(handlers::checkpoint))
        .route("/search", post(handlers::search))
        .route_layer(middleware::from_fn(telemetry::http_metrics_middleware))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state))
}

fn cors_layer_from_env() -> Result<CorsLayer, anyhow::Error> {
    let allowed = env::var("ALLOWED_ORIGINS").unwrap_or_else(|_| "*".to_string());
    if allowed.trim() == "*" {
        tracing::warn!(
            target: "security",
            "ALLOWED_ORIGINS=* permits any browser origin; restrict it before exposing the orchestrator beyond a single-user local node"
        );
        return Ok(CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any));
    }

    let origins: Vec<HeaderValue> = allowed
        .split(',')
        .map(str::trim)
        .filter(|origin| !origin.is_empty())
        .map(HeaderValue::from_str)
        .collect::<Result<_, _>>()?;

    if origins.is_empty() {
        anyhow::bail!("ALLOWED_ORIGINS must be '*' or a comma-separated origin list");
    }

    Ok(CorsLayer::new()
        .allow_origin(origins)
        .allow_methods(Any)
        .allow_headers(Any))
}
