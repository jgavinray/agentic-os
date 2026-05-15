mod state;
mod db;
mod qdrant;
mod logging;
mod handlers;

use axum::routing::{get, post};
use axum::Router;
use std::env;
use std::sync::Arc;
use std::time::Duration;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::state::AppState;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    logging::init_logging();

    let db_url          = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let qdrant_url      = env::var("QDRANT_URL").expect("QDRANT_URL must be set");
    let litellm_url     = env::var("LITELLM_URL").expect("LITELLM_URL must be set");
    let litellm_key     = env::var("LITELLM_KEY").expect("LITELLM_KEY must be set");
    // BUG-12: canonical model name matches litellm-config.yaml
    // API_KEYS is comma-separated; API_KEY is the single-key fallback for backwards compat.
    let api_keys: Vec<String> = env::var("API_KEYS")
        .or_else(|_| env::var("API_KEY"))
        .unwrap_or_else(|_| "sk-local-orchestrator".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let default_model   = env::var("DEFAULT_MODEL").unwrap_or_else(|_| "qwen36-35b-heretic".to_string());
    let default_task    = env::var("DEFAULT_TASK").unwrap_or_else(|_| "engineering".to_string());
    let embedding_url   = env::var("EMBEDDING_URL").unwrap_or_else(|_| "http://embedding:80".to_string());

    let pool = db::create_pool(&db_url)?;
    db::init_schema(&pool).await?;
    qdrant::init(&qdrant_url).await?;

    // BUG-6: Explicit HTTP client timeouts. Two clients: one for normal requests
    // (full timeout budget), one for streaming (no overall timeout since completions
    // can legitimately run for many minutes).
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .pool_idle_timeout(Duration::from_secs(90))
        .timeout(Duration::from_secs(300))
        .build()?;

    let http_stream = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .pool_idle_timeout(Duration::from_secs(90))
        // No overall timeout — streaming responses are long-lived.
        .build()?;

    let state = Arc::new(AppState {
        pool,
        db_url,
        qdrant_url: qdrant_url.clone(),
        litellm_url,
        litellm_key,
        api_keys,
        default_model,
        default_task,
        embedding_url,
        http,
        http_stream,
    });

    // ADD-5: TraceLayer emits structured per-request logs (method, path, status, latency).
    let app = Router::new()
        .route("/health",               get(handlers::health))
        .route("/health/live",          get(handlers::health_live))
        .route("/health/ready",         get(handlers::health_ready))
        .route("/v1/models",            get(handlers::list_models))
        .route("/v1/chat/completions",  post(handlers::chat_completions))
        .route("/sessions/start",       post(handlers::start_session))
        .route("/events/append",        post(handlers::append_event))
        .route("/context/pack",         post(handlers::context_pack))
        .route("/summaries/checkpoint", post(handlers::checkpoint))
        .route("/search",               post(handlers::search))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr: std::net::SocketAddr = "0.0.0.0:8088".parse().unwrap();
    let listener = tokio::net::TcpListener::bind(addr).await?;

    tracing::info!("orchestrator listening on {addr}");

    // BUG-7: Graceful shutdown on SIGTERM or Ctrl+C so in-flight requests and
    // background persistence tasks complete before the process exits.
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c    => tracing::info!("received Ctrl+C, shutting down"),
        _ = terminate => tracing::info!("received SIGTERM, shutting down"),
    }
}
