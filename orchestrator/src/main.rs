mod anthropic;
mod db;
mod embedder;
mod handlers;
mod hybrid;
mod logging;
mod migrations;
mod qdrant;
mod rate_limit;
mod sentiment;
mod state;
mod summarizer;
mod telemetry;

use axum::http::HeaderValue;
use axum::middleware;
use axum::routing::{get, post};
use axum::Router;
use std::env;
use std::sync::Arc;
use std::time::Duration;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::state::AppState;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let logging_guard = logging::init_logging()?;
    let prometheus = telemetry::install_recorder()?;

    let db_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let qdrant_url = env::var("QDRANT_URL").expect("QDRANT_URL must be set");
    let litellm_url = env::var("LITELLM_URL").expect("LITELLM_URL must be set");
    let litellm_key = env::var("LITELLM_KEY").expect("LITELLM_KEY must be set");
    // BUG-12: canonical model name matches litessh-prompt.md
    // API_KEYS is semicolon-delimited entries: `token,namespace;token2,namespace2`
    let api_keys: Vec<(String, String)> = env::var("API_KEYS")
        .unwrap_or_else(|_| "agent-os,agentic-os".to_string())
        .split(';')
        .map(|s| {
            let s = s.trim();
            let mut parts = s.splitn(2, ',');
            let token = parts.next().unwrap_or(s).trim().to_string();
            let namespace = parts.next().unwrap_or(&token).trim().to_string();
            (token, namespace)
        })
        .filter(|(t, _)| !t.is_empty())
        .collect();
    let default_model =
        env::var("DEFAULT_MODEL").unwrap_or_else(|_| "qwen36-35b-heretic".to_string());
    let default_task = env::var("DEFAULT_TASK").unwrap_or_else(|_| "engineering".to_string());
    let cache_ttl_ms = env::var("CONTEXT_CACHE_TTL_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(state::CONTEXT_CACHE_TTL_MS);
    let context_decay_rate = env::var("CONTEXT_DECAY_RATE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(state::DEFAULT_CONTEXT_DECAY_RATE);
    let rate_limit_per_minute = env::var("RATE_LIMIT_PER_MINUTE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    let rate_limit_burst = env::var("RATE_LIMIT_BURST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let embed_model_path = env::var("EMBED_MODEL_PATH").expect("EMBED_MODEL_PATH must be set");

    let pool = db::create_pool(&db_url)?;
    let single_writer = db::acquire_single_writer_lock(&pool).await?;
    migrations::run(&pool).await?;
    qdrant::init(&qdrant_url).await?;

    let embedder = Arc::new(
        embedder::Embedder::load(&embed_model_path)
            .expect("failed to load embedding model — run setup-models.sh first"),
    );

    let sentiment_classifier = std::env::var("SENTIMENT_MODEL_PATH").ok().and_then(|path| {
        let threshold = std::env::var("SENTIMENT_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.70_f32);
        match sentiment::SentimentClassifier::load(&path, threshold) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                tracing::warn!(
                    "sentiment classifier unavailable, negative feedback detection disabled: {e}"
                );
                None
            }
        }
    });

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

    let metrics = telemetry::MetricsRegistry::new();
    telemetry::prime_metrics(&metrics, &default_model, sentiment_classifier.is_some());

    let state = Arc::new(AppState {
        pool,
        sentiment: sentiment_classifier,
        qdrant_url: qdrant_url.clone(),
        litellm_url,
        litellm_key,
        api_keys,
        default_model,
        default_task,
        embedder,
        http,
        http_stream,
        cache: state::ContextCache::new(cache_ttl_ms),
        context_decay_rate,
        rate_limiter: rate_limit::RateLimiter::new(rate_limit_per_minute, rate_limit_burst),
        prometheus,
        metrics,
    });

    tokio::spawn(crate::summarizer::run(Arc::clone(&state)));

    // ADD-5: TraceLayer emits structured per-request logs (method, path, status, latency).
    let cors = cors_layer_from_env()?;

    let app = Router::new()
        .route("/health", get(handlers::health))
        .route("/health/live", get(handlers::health_live))
        .route("/health/ready", get(handlers::health_ready))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/messages", post(handlers::messages))
        .route("/sessions/start", post(handlers::start_session))
        .route("/events/append", post(handlers::append_event))
        .route("/context/pack", post(handlers::context_pack))
        .route("/cache/stats", get(handlers::cache_stats))
        .route("/metrics", get(handlers::metrics))
        .route("/metrics/json", get(handlers::metrics_json))
        .route("/summaries/checkpoint", post(handlers::checkpoint))
        .route("/search", post(handlers::search))
        .route_layer(middleware::from_fn(telemetry::http_metrics_middleware))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state);

    let addr: std::net::SocketAddr = "0.0.0.0:8088".parse().unwrap();
    let listener = tokio::net::TcpListener::bind(addr).await?;

    tracing::info!("orchestrator listening on {addr}");

    // BUG-7: Graceful shutdown on SIGTERM or Ctrl+C so in-flight requests and
    // background persistence tasks complete before the process exits.
    let server_result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await;

    single_writer.release().await;
    logging_guard.shutdown();
    server_result?;

    Ok(())
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

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
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
