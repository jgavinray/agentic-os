use orchestrator::{
    app_router, client_capture, db, handlers, logging, migrations, prompt_intervention_records,
    qdrant, rate_limit, request_classification, sampling, startup_backfill, startup_config,
    startup_runtime, state, summarizer, telemetry,
};
use std::sync::Arc;

use orchestrator::state::AppState;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let logging_guard = logging::init_logging()?;
    let prometheus = telemetry::install_recorder()?;
    let config = startup_config::load_startup_config()?;

    let pool = db::create_pool(&config.db_url)?;
    let capture_pool = if let Some(url) = config.capture_db_url.as_deref() {
        let pool = db::create_pool(url)?;
        client_capture::init(&pool).await?;
        prompt_intervention_records::init(&pool).await?;
        Some(pool)
    } else {
        None
    };
    let single_writer = db::acquire_single_writer_lock(&pool).await?;
    migrations::run(&pool).await?;
    if config.execution_feedback_enabled {
        db::warn_if_legacy_signature_backfill_pending(&pool).await?;
    }
    qdrant::init(&config.qdrant_url).await?;

    let embedder = startup_runtime::load_embedder(&config.embed_model_path);
    let sentiment_classifier = startup_runtime::load_sentiment_classifier_from_env();
    let (http, http_stream) =
        startup_runtime::build_http_clients(config.litellm_request_timeout_secs)?;

    let metrics = telemetry::MetricsRegistry::new();
    telemetry::prime_metrics(
        &metrics,
        &config.default_model,
        sentiment_classifier.is_some(),
    );

    let state = Arc::new(AppState {
        pool,
        capture_pool,
        sentiment: sentiment_classifier,
        qdrant_url: config.qdrant_url.clone(),
        litellm_url: config.litellm_url,
        litellm_key: config.litellm_key,
        summarizer_url: config.summarizer_url,
        summarizer_key: config.summarizer_key,
        summarizer_model: config.summarizer_model,
        total_recall_url: config.total_recall_url,
        vllm_metrics_url: config.vllm_metrics_url,
        summarizer_max_tokens: config.summarizer_max_tokens,
        api_keys: config.api_keys,
        default_model: config.default_model,
        default_task: config.default_task,
        embedder,
        http,
        http_stream,
        cache: state::ContextCache::new(config.cache_ttl_ms),
        context_decay_rate: config.context_decay_rate,
        rate_limiter: rate_limit::RateLimiter::new(
            config.rate_limit_per_minute,
            config.rate_limit_burst,
        ),
        execution_feedback_enabled: config.execution_feedback_enabled,
        trajectory_capture_enabled: config.trajectory_capture_enabled,
        trajectory_idle_timeout_sec: config.trajectory_idle_timeout_sec,
        failure_history_token_budget: config.failure_history_token_budget,
        feature_extraction_enabled: config.feature_extraction_enabled,
        operational_constraints_token_budget: config.operational_constraints_token_budget,
        background_work: Arc::new(tokio::sync::Semaphore::new(
            config.background_work_concurrency,
        )),
        sampling_config: config.sampling_config,
        sampling_policy: Arc::new(sampling::NoOpSamplingPolicy),
        request_live_policy_config: request_classification::live_policy_config_from_env(),
        prefix_cache_canary_enabled: config.prefix_cache_canary_enabled,
        prefix_cache_canary_namespace_allowlist: config.prefix_cache_canary_namespace_allowlist,
        tool_mediation_enabled: config.tool_mediation_enabled,
        prometheus,
        metrics,
    });

    startup_backfill::run_startup_harness_feedback_backfill(&state).await;
    startup_backfill::run_startup_request_classification_backfill(&state).await?;
    startup_backfill::run_startup_feature_backfill(&state).await?;

    if config.summarizer_enabled {
        tracing::info!(
            target: "summarizer",
            endpoint = %state.summarizer_url,
            model = %state.summarizer_model,
            max_tokens = state.summarizer_max_tokens,
            "summarizer loop enabled"
        );
        tokio::spawn(summarizer::run(Arc::clone(&state)));
    } else {
        tracing::info!("summarizer loop disabled by SUMMARIZER_ENABLED=false");
    }
    if state.trajectory_capture_enabled {
        tokio::spawn(handlers::run_trajectory_idle_sweep(Arc::clone(&state)));
    }
    startup_backfill::spawn_harness_feedback_background_repair(Arc::clone(&state));

    let app = app_router::build_router(state)?;

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
