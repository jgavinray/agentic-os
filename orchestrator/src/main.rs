use orchestrator::{
    app_router, client_capture, db, handlers, logging, migrations, qdrant, rate_limit,
    request_classification, sampling, startup_backfill, startup_env, startup_runtime, state,
    summarizer, telemetry,
};
use std::env;
use std::sync::Arc;

use orchestrator::state::AppState;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let logging_guard = logging::init_logging()?;
    let prometheus = telemetry::install_recorder()?;

    let db_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let capture_db_url = env::var("CAPTURE_DATABASE_URL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let qdrant_url = env::var("QDRANT_URL").expect("QDRANT_URL must be set");
    let litellm_url = env::var("LITELLM_URL")
        .expect("LITELLM_URL must be set")
        .trim_end_matches('/')
        .to_string();
    let litellm_key = env::var("LITELLM_KEY").expect("LITELLM_KEY must be set");
    // BUG-12: canonical model name matches litessh-prompt.md
    let api_keys = startup_env::api_keys_from_env();
    let default_model =
        env::var("DEFAULT_MODEL").unwrap_or_else(|_| "qwen36-35b-heretic".to_string());
    let summarizer_url = env::var("SUMMARIZER_BASE_URL")
        .unwrap_or_else(|_| litellm_url.clone())
        .trim_end_matches('/')
        .to_string();
    let summarizer_model = env::var("SUMMARIZER_MODEL").unwrap_or_else(|_| default_model.clone());
    let total_recall_url = startup_env::optional_trimmed_url_env("TOTAL_RECALL_URL");
    let vllm_metrics_url = startup_env::optional_trimmed_env("VLLM_METRICS_URL");
    let summarizer_key = startup_env::optional_trimmed_env("SUMMARIZER_KEY").or_else(|| {
        if summarizer_url == litellm_url {
            Some(litellm_key.clone())
        } else {
            None
        }
    });
    let summarizer_max_tokens =
        startup_env::parsed_env_min("SUMMARIZER_MAX_TOKENS", state::SUMMARIZER_MAX_TOKENS, 1);
    let default_task = env::var("DEFAULT_TASK").unwrap_or_else(|_| "engineering".to_string());
    let cache_ttl_ms = startup_env::parsed_env("CONTEXT_CACHE_TTL_MS", state::CONTEXT_CACHE_TTL_MS);
    let context_decay_rate =
        startup_env::parsed_env("CONTEXT_DECAY_RATE", state::DEFAULT_CONTEXT_DECAY_RATE);
    let rate_limit_per_minute = startup_env::parsed_env("RATE_LIMIT_PER_MINUTE", 60);
    let rate_limit_burst = startup_env::parsed_env("RATE_LIMIT_BURST", 30);
    let execution_feedback_enabled = startup_env::bool_env("EXECUTION_FEEDBACK_ENABLED", true);
    let trajectory_capture_enabled = startup_env::bool_env("TRAJECTORY_CAPTURE_ENABLED", true);
    let trajectory_idle_timeout_sec = startup_env::parsed_env(
        "TRAJECTORY_IDLE_TIMEOUT_SEC",
        orchestrator::trajectory::DEFAULT_TRAJECTORY_IDLE_TIMEOUT_SEC,
    );
    let sampling_config = sampling::SamplingConfig::from_env()?;
    let failure_history_token_budget = startup_env::parsed_env(
        "FAILURE_HISTORY_TOKEN_BUDGET",
        state::DEFAULT_FAILURE_HISTORY_TOKEN_BUDGET,
    );
    let feature_extraction_enabled =
        orchestrator::feature_extraction::feature_extraction_enabled_from_env();
    let tool_mediation_enabled = startup_env::bool_env("TOOL_MEDIATION_ENABLED", true);
    let prefix_cache_canary_enabled = startup_env::bool_env("PREFIX_CACHE_CANARY_ENABLED", false);
    let prefix_cache_canary_namespace_allowlist =
        startup_env::comma_list_env("PREFIX_CACHE_CANARY_NAMESPACE_ALLOWLIST");
    let summarizer_enabled = startup_env::bool_env("SUMMARIZER_ENABLED", true);
    let operational_constraints_token_budget =
        orchestrator::feature_extraction::operational_constraints_token_budget_from_env();
    let background_work_concurrency = startup_env::parsed_env_min(
        "BACKGROUND_WORK_CONCURRENCY",
        state::DEFAULT_BACKGROUND_WORK_CONCURRENCY,
        1,
    );
    let litellm_request_timeout_secs =
        startup_env::parsed_env_min("LITELLM_REQUEST_TIMEOUT_SECS", 300, 1);
    let embed_model_path = env::var("EMBED_MODEL_PATH").expect("EMBED_MODEL_PATH must be set");

    let pool = db::create_pool(&db_url)?;
    let capture_pool = if let Some(url) = capture_db_url.as_deref() {
        let pool = db::create_pool(url)?;
        client_capture::init(&pool).await?;
        Some(pool)
    } else {
        None
    };
    let single_writer = db::acquire_single_writer_lock(&pool).await?;
    migrations::run(&pool).await?;
    if execution_feedback_enabled {
        db::warn_if_legacy_signature_backfill_pending(&pool).await?;
    }
    qdrant::init(&qdrant_url).await?;

    let embedder = startup_runtime::load_embedder(&embed_model_path);
    let sentiment_classifier = startup_runtime::load_sentiment_classifier_from_env();
    let (http, http_stream) = startup_runtime::build_http_clients(litellm_request_timeout_secs)?;

    let metrics = telemetry::MetricsRegistry::new();
    telemetry::prime_metrics(&metrics, &default_model, sentiment_classifier.is_some());

    let state = Arc::new(AppState {
        pool,
        capture_pool,
        sentiment: sentiment_classifier,
        qdrant_url: qdrant_url.clone(),
        litellm_url,
        litellm_key,
        summarizer_url,
        summarizer_key,
        summarizer_model,
        total_recall_url,
        vllm_metrics_url,
        summarizer_max_tokens,
        api_keys,
        default_model,
        default_task,
        embedder,
        http,
        http_stream,
        cache: state::ContextCache::new(cache_ttl_ms),
        context_decay_rate,
        rate_limiter: rate_limit::RateLimiter::new(rate_limit_per_minute, rate_limit_burst),
        execution_feedback_enabled,
        trajectory_capture_enabled,
        trajectory_idle_timeout_sec,
        failure_history_token_budget,
        feature_extraction_enabled,
        operational_constraints_token_budget,
        background_work: Arc::new(tokio::sync::Semaphore::new(background_work_concurrency)),
        sampling_config,
        sampling_policy: Arc::new(sampling::NoOpSamplingPolicy),
        request_live_policy_config: request_classification::live_policy_config_from_env(),
        prefix_cache_canary_enabled,
        prefix_cache_canary_namespace_allowlist,
        tool_mediation_enabled,
        prometheus,
        metrics,
    });

    startup_backfill::run_startup_harness_feedback_backfill(&state).await;
    startup_backfill::run_startup_request_classification_backfill(&state).await?;
    startup_backfill::run_startup_feature_backfill(&state).await?;

    if summarizer_enabled {
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
