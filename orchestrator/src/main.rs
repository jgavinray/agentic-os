mod state;
mod db;
mod qdrant;
mod logging;
mod handlers;

use axum::routing::{get, post};
use axum::Router;
use std::env;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::state::AppState;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Structured logging
    logging::init_logging();

    let db_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let qdrant_url = env::var("QDRANT_URL").expect("QDRANT_URL must be set");
    let litellm_url = env::var("LITELLM_URL").expect("LITELLM_URL must be set");
    let litellm_key = env::var("LITELLM_KEY").expect("LITELLM_KEY must be set");
    let api_key = env::var("API_KEY").unwrap_or_else(|_| "orchestrator-local".to_string());
    let default_model = env::var("DEFAULT_MODEL").unwrap_or_else(|_| "qwen3.6-35b-a3b".to_string());

    // Create Postgres pool
    let pool = db::create_pool(&db_url)?;

    // Initialize schema
    db::init_schema(&pool).await?;

    // Initialize Qdrant collection
    qdrant::init(&qdrant_url).await?;

    let state = Arc::new(AppState {
        pool,
        db_url,
        qdrant_url: qdrant_url.clone(),
        litellm_url,
        litellm_key,
        api_key,
        default_model,
        http: reqwest::Client::new(),
    });

    // Build router with all routes
    let app = Router::new()
        // Health checks (no auth)
        .route("/health", get(handlers::health))
        .route("/health/live", get(handlers::health_live))
        .route("/health/ready", get(handlers::health_ready))
        // OpenAI-compatible API (with auth)
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        // Internal orchestrator endpoints (with auth)
        .route("/sessions/start", post(handlers::start_session))
        .route("/events/append", post(handlers::append_event))
        .route("/context/pack", post(handlers::context_pack))
        .route("/summaries/checkpoint", post(handlers::checkpoint))
        // Semantic retrieval (with auth)
        .route("/search", post(handlers::search))
        // Layer
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr: std::net::SocketAddr = "0.0.0.0:8088".parse().unwrap();

    tracing::info!("orchestrator starting on {addr}");
    tracing::info!("health:   {}/health", addr);
    tracing::info!("chat:     {}/v1/chat/completions", addr);
    tracing::info!("models:   {}/v1/models", addr);

    axum::serve(
        tokio::net::TcpListener::bind(addr).await?,
        app,
    ).await?;

    Ok(())
}
