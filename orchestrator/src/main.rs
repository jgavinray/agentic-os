use axum::http::HeaderValue;
use axum::middleware;
use axum::routing::{get, post};
use axum::Router;
use orchestrator::{
    client_capture, db, embedder, feature_extraction, handlers, harness_feedback, logging,
    migrations, qdrant, rate_limit, request_classification, sampling, sentiment, state, summarizer,
    telemetry,
};
use std::env;
use std::sync::Arc;
use std::time::Duration;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

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
    let summarizer_url = env::var("SUMMARIZER_BASE_URL")
        .unwrap_or_else(|_| litellm_url.clone())
        .trim_end_matches('/')
        .to_string();
    let summarizer_model = env::var("SUMMARIZER_MODEL").unwrap_or_else(|_| default_model.clone());
    let total_recall_url = env::var("TOTAL_RECALL_URL")
        .ok()
        .map(|value| value.trim().trim_end_matches('/').to_string())
        .filter(|value| !value.is_empty());
    let vllm_metrics_url = env::var("VLLM_METRICS_URL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let summarizer_key = env::var("SUMMARIZER_KEY")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .or_else(|| {
            if summarizer_url == litellm_url {
                Some(litellm_key.clone())
            } else {
                None
            }
        });
    let summarizer_max_tokens = env::var("SUMMARIZER_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(state::SUMMARIZER_MAX_TOKENS)
        .max(1);
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
    let execution_feedback_enabled = env::var("EXECUTION_FEEDBACK_ENABLED")
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true);
    let trajectory_capture_enabled = env::var("TRAJECTORY_CAPTURE_ENABLED")
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true);
    let trajectory_idle_timeout_sec = env::var("TRAJECTORY_IDLE_TIMEOUT_SEC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(orchestrator::trajectory::DEFAULT_TRAJECTORY_IDLE_TIMEOUT_SEC);
    let sampling_config = sampling::SamplingConfig::from_env()?;
    let failure_history_token_budget = env::var("FAILURE_HISTORY_TOKEN_BUDGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(state::DEFAULT_FAILURE_HISTORY_TOKEN_BUDGET);
    let feature_extraction_enabled =
        orchestrator::feature_extraction::feature_extraction_enabled_from_env();
    let tool_mediation_enabled = env::var("TOOL_MEDIATION_ENABLED")
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true);
    let prefix_cache_canary_enabled = env::var("PREFIX_CACHE_CANARY_ENABLED")
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(false);
    let prefix_cache_canary_namespace_allowlist =
        env::var("PREFIX_CACHE_CANARY_NAMESPACE_ALLOWLIST")
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_string)
            .collect();
    let summarizer_enabled = env::var("SUMMARIZER_ENABLED")
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true);
    let operational_constraints_token_budget =
        orchestrator::feature_extraction::operational_constraints_token_budget_from_env();
    let background_work_concurrency = env::var("BACKGROUND_WORK_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(state::DEFAULT_BACKGROUND_WORK_CONCURRENCY)
        .max(1);
    let litellm_request_timeout_secs = env::var("LITELLM_REQUEST_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300)
        .max(1);
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
        .timeout(Duration::from_secs(litellm_request_timeout_secs))
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

    run_startup_harness_feedback_backfill(&state).await;
    run_startup_request_classification_backfill(&state).await?;
    run_startup_feature_backfill(&state).await?;

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
    spawn_harness_feedback_background_repair(Arc::clone(&state));

    // ADD-5: TraceLayer emits structured per-request logs (method, path, status, latency).
    let cors = cors_layer_from_env()?;

    let app = Router::new()
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

async fn run_startup_feature_backfill(state: &AppState) -> Result<(), anyhow::Error> {
    if !state.feature_extraction_enabled {
        tracing::info!(
            target: "feature_extraction",
            "startup feature backfill skipped because FEATURE_EXTRACTION_ENABLED=false"
        );
        return Ok(());
    }
    if !env_flag("FEATURE_STARTUP_BACKFILL_ENABLED", true) {
        tracing::info!(
            target: "feature_extraction",
            "startup feature backfill disabled by FEATURE_STARTUP_BACKFILL_ENABLED=false"
        );
        return Ok(());
    }

    let batch_size = positive_i64_env("FEATURE_STARTUP_BACKFILL_BATCH_SIZE", 500);
    let skip_bootstrap_tagging = env_flag("FEATURE_STARTUP_SKIP_BOOTSTRAP_TAGGING", false);
    let started = std::time::Instant::now();
    tracing::info!(
        target: "feature_extraction",
        batch_size,
        skip_bootstrap_tagging,
        "startup feature backfill started"
    );
    let report = feature_extraction::run_backfill(
        &state.pool,
        &feature_extraction::BackfillOptions {
            repo: None,
            session_id: None,
            trajectory_id: None,
            since: None,
            dry_run: false,
            batch_size,
            skip_bootstrap_tagging,
        },
        &feature_extraction::ExtractionConfig {
            feature_window_sec: feature_extraction::feature_window_sec_from_env(),
            constraint_freshness_window_sec:
                feature_extraction::constraint_freshness_window_sec_from_env(),
            max_operational_constraints: feature_extraction::max_operational_constraints_from_env(),
            evaluation_time: chrono::Utc::now(),
        },
    )
    .await
    .map_err(|e| {
        telemetry::record_feature_extraction_failure("extraction");
        tracing::error!(
            target: "feature_extraction",
            "startup feature backfill failed; refusing to serve traffic: {e}"
        );
        e
    })?;

    for _ in 0..report.persisted_records {
        telemetry::record_trajectory_feature_written(&state.metrics);
    }
    telemetry::record_feature_extraction_duration(started.elapsed());
    tracing::info!(
        target: "feature_extraction",
        events_scanned = report.events_scanned,
        bootstrap_updated = report.bootstrap_updated,
        feature_records = report.feature_records,
        persisted_records = report.persisted_records,
        elapsed_ms = started.elapsed().as_millis(),
        "startup feature backfill completed"
    );
    Ok(())
}

async fn run_startup_request_classification_backfill(
    state: &AppState,
) -> Result<(), anyhow::Error> {
    if !request_classification::request_classification_startup_backfill_enabled() {
        tracing::info!(
            target: "request_classification",
            "startup request classification backfill disabled by REQUEST_CLASSIFICATION_STARTUP_BACKFILL_ENABLED=false"
        );
        return Ok(());
    }

    let batch_size = request_classification::request_classification_startup_batch_size();
    let started = std::time::Instant::now();
    tracing::info!(
        target: "request_classification",
        batch_size,
        "startup request classification backfill started"
    );
    let report = request_classification::run_backfill(
        &state.pool,
        &request_classification::BackfillOptions {
            repo: None,
            session_id: None,
            since: None,
            dry_run: false,
            repair: false,
            repair_stale: true,
            batch_size,
        },
    )
    .await
    .map_err(|e| {
        tracing::error!(
            target: "request_classification",
            "startup request classification backfill failed; refusing to serve traffic: {e}"
        );
        e
    })?;

    tracing::info!(
        target: "request_classification",
        events_scanned = report.events_scanned,
        inserted = report.inserted,
        updated = report.updated,
        skipped = report.skipped,
        elapsed_ms = started.elapsed().as_millis(),
        "startup request classification backfill completed"
    );
    Ok(())
}

async fn run_startup_harness_feedback_backfill(state: &AppState) {
    if !env_flag("HARNESS_FEEDBACK_STARTUP_BACKFILL_ENABLED", true) {
        tracing::info!(
            target: "harness_feedback",
            "startup harness feedback repair disabled by HARNESS_FEEDBACK_STARTUP_BACKFILL_ENABLED=false"
        );
        return;
    }

    let batch_size = positive_i64_env("HARNESS_FEEDBACK_STARTUP_BACKFILL_BATCH_SIZE", 500);
    let started = std::time::Instant::now();
    tracing::info!(
        target: "harness_feedback",
        batch_size,
        "startup harness feedback repair started"
    );
    match harness_feedback::run_backfill(
        &state.pool,
        &harness_feedback::BackfillOptions {
            repo: None,
            session_id: None,
            since: None,
            dry_run: false,
            batch_size,
        },
    )
    .await
    {
        Ok(report) => {
            telemetry::record_harness_feedback_repair_run("success");
            tracing::info!(
                target: "harness_feedback",
                events_scanned = report.events_scanned,
                updated = report.updated,
                quarantined = report.quarantined,
                elapsed_ms = started.elapsed().as_millis(),
                "startup harness feedback repair completed"
            );
        }
        Err(e) => {
            telemetry::record_harness_feedback_repair_run("failure");
            tracing::warn!(
                target: "harness_feedback",
                "startup harness feedback repair failed; continuing because live classification remains active: {e}"
            );
        }
    }
}

fn spawn_harness_feedback_background_repair(state: Arc<AppState>) {
    if !env_flag("HARNESS_FEEDBACK_BACKGROUND_REPAIR_ENABLED", true) {
        tracing::info!(
            target: "harness_feedback",
            "background harness feedback repair disabled by HARNESS_FEEDBACK_BACKGROUND_REPAIR_ENABLED=false"
        );
        return;
    }

    let interval_sec = positive_i64_env("HARNESS_FEEDBACK_REPAIR_INTERVAL_SEC", 300) as u64;
    let lookback_sec = positive_i64_env(
        "HARNESS_FEEDBACK_REPAIR_LOOKBACK_SEC",
        (interval_sec.saturating_mul(2)).max(60) as i64,
    );
    let batch_size = positive_i64_env("HARNESS_FEEDBACK_REPAIR_BATCH_SIZE", 500);
    tracing::info!(
        target: "harness_feedback",
        interval_sec,
        lookback_sec,
        batch_size,
        "background harness feedback repair enabled"
    );

    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(interval_sec)).await;
            let since = chrono::Utc::now() - chrono::Duration::seconds(lookback_sec);
            let started = std::time::Instant::now();
            match harness_feedback::run_backfill(
                &state.pool,
                &harness_feedback::BackfillOptions {
                    repo: None,
                    session_id: None,
                    since: Some(since),
                    dry_run: false,
                    batch_size,
                },
            )
            .await
            {
                Ok(report) => {
                    telemetry::record_harness_feedback_repair_run("success");
                    tracing::info!(
                        target: "harness_feedback",
                        events_scanned = report.events_scanned,
                        updated = report.updated,
                        quarantined = report.quarantined,
                        elapsed_ms = started.elapsed().as_millis(),
                        "background harness feedback repair completed"
                    );
                }
                Err(e) => {
                    telemetry::record_harness_feedback_repair_run("failure");
                    tracing::warn!(
                        target: "harness_feedback",
                        "background harness feedback repair failed: {e}"
                    );
                }
            }
        }
    });
}

fn env_flag(name: &str, default: bool) -> bool {
    env::var(name)
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(default)
}

fn positive_i64_env(name: &str, default: i64) -> i64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
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
