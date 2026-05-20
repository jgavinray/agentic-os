use axum::extract::MatchedPath;
use axum::extract::Request;
use axum::http::{header, HeaderMap};
use axum::middleware::Next;
use axum::response::Response;
use deadpool_postgres::Pool;
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use serde::Serialize;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const HTTP_BUCKETS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];
const UPSTREAM_BUCKETS: &[f64] = &[
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
];
const IO_BUCKETS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
];
const EMBEDDER_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0,
];
const TOKEN_BUCKETS: &[f64] = &[
    16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0,
];

#[derive(Debug, Clone, Serialize, Default)]
pub struct MetricsSnapshot {
    pub context_pack_requests: u64,
    pub context_cache_hits: u64,
    pub context_cache_misses: u64,
    pub context_cache_replacements: u64,
    pub context_pack_build_ms_total: u64,
    pub context_pack_chars_total: u64,
    pub context_pack_tokens_estimate_total: u64,
    pub l0_items_injected: u64,
    pub l1_items_injected: u64,
    pub l2_items_injected: u64,
    pub l3_items_injected: u64,
    pub failed_attempts_injected: u64,
    pub remediations_injected: u64,
    pub failure_history_items_injected: u64,
    pub operational_constraints_injected: u64,
    pub retrieval_semantic_hits: u64,
    pub retrieval_fts_hits: u64,
    pub retrieval_deduped_hits: u64,
    pub processed_tokens: u64,
    pub cached_tokens: u64,
    pub generated_tokens: u64,
    pub promotion_attempts: u64,
    pub promotion_accepted: u64,
    pub promotion_rejected: u64,
    pub memory_source_items: u64,
    pub memory_source_items_with_sources: u64,
    pub memory_source_coverage: f64,
    pub stale_cache_invalidations: u64,
    pub execution_artifacts: u64,
    pub failure_signatures: u64,
    pub patch_lifecycle_events: u64,
    pub validation_results: u64,
    pub remediation_reuse: u64,
    pub trajectory_results: u64,
    pub trajectory_attempts: u64,
    pub trajectory_validation_failures: u64,
    pub trajectory_input_tokens: u64,
    pub trajectory_output_tokens: u64,
    pub trajectory_features: u64,
    pub feature_extraction_failures: u64,
    pub feature_tag_schema_version_unknown: u64,
}

#[derive(Clone)]
pub struct MetricsRegistry {
    inner: Arc<RwLock<MetricsSnapshot>>,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(MetricsSnapshot::default())),
        }
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        self.inner.read().unwrap().clone()
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn install_recorder() -> Result<PrometheusHandle, anyhow::Error> {
    let handle = PrometheusBuilder::new()
        .set_buckets_for_metric(
            Matcher::Full("http_request_duration_seconds".to_string()),
            HTTP_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("upstream_litellm_duration_seconds".to_string()),
            UPSTREAM_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("summarizer_upstream_duration_seconds".to_string()),
            UPSTREAM_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("stream_first_token_seconds".to_string()),
            UPSTREAM_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("stream_duration_seconds".to_string()),
            UPSTREAM_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("db_query_duration_seconds".to_string()),
            IO_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("qdrant_request_duration_seconds".to_string()),
            IO_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("embedder_inference_duration_seconds".to_string()),
            EMBEDDER_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("embedder_input_tokens".to_string()),
            TOKEN_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("sentiment_inference_duration_seconds".to_string()),
            EMBEDDER_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("summarizer_duration_seconds".to_string()),
            UPSTREAM_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("context_pack_build_duration_seconds".to_string()),
            HTTP_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("context_pack_tokens_estimate".to_string()),
            TOKEN_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("feature_extraction_duration_seconds".to_string()),
            IO_BUCKETS,
        )?
        .install_recorder()?;
    describe_metrics();
    Ok(handle)
}

fn describe_metrics() {
    describe_counter!(
        "http_requests_total",
        "HTTP requests served by the orchestrator."
    );
    describe_histogram!(
        "http_request_duration_seconds",
        "HTTP request latency in seconds."
    );
    describe_gauge!(
        "http_requests_in_flight",
        "HTTP requests currently in flight."
    );
    describe_counter!("auth_attempts_total", "API authentication attempts.");
    describe_counter!(
        "upstream_litellm_requests_total",
        "Requests sent from the orchestrator to LiteLLM."
    );
    describe_histogram!(
        "upstream_litellm_duration_seconds",
        "LiteLLM upstream request latency in seconds."
    );
    describe_counter!(
        "upstream_litellm_errors_total",
        "LiteLLM upstream request errors by path and kind."
    );
    describe_counter!(
        "summarizer_upstream_requests_total",
        "Requests sent from the orchestrator to the summarizer endpoint."
    );
    describe_histogram!(
        "summarizer_upstream_duration_seconds",
        "Summarizer upstream request latency in seconds."
    );
    describe_counter!(
        "summarizer_upstream_errors_total",
        "Summarizer upstream request errors by path and kind."
    );
    describe_histogram!(
        "stream_first_token_seconds",
        "Seconds from request handling to first streamed byte."
    );
    describe_histogram!(
        "stream_duration_seconds",
        "Total stream duration in seconds."
    );
    describe_counter!(
        "stream_disconnects_total",
        "Streams interrupted before completion."
    );
    describe_gauge!("db_pool_size", "Configured Postgres pool capacity.");
    describe_gauge!(
        "db_pool_available",
        "Postgres pool connections currently available."
    );
    describe_gauge!(
        "db_pool_waiters",
        "Tasks waiting for a Postgres pool connection."
    );
    describe_histogram!(
        "db_query_duration_seconds",
        "Postgres query latency in seconds."
    );
    describe_counter!("db_query_errors_total", "Postgres query failures.");
    describe_counter!("qdrant_requests_total", "Qdrant API requests.");
    describe_histogram!(
        "qdrant_request_duration_seconds",
        "Qdrant API latency in seconds."
    );
    describe_counter!(
        "embedder_inferences_total",
        "Embedding model inference calls."
    );
    describe_histogram!(
        "embedder_inference_duration_seconds",
        "Embedding model inference latency in seconds."
    );
    describe_histogram!("embedder_input_tokens", "Tokenized embedder input length.");
    describe_counter!(
        "sentiment_inferences_total",
        "Sentiment classifier inferences by verdict."
    );
    describe_histogram!(
        "sentiment_inference_duration_seconds",
        "Sentiment classifier inference latency in seconds."
    );
    describe_counter!("summarizer_ticks_total", "Summarizer loop ticks.");
    describe_counter!(
        "summarizer_candidates_found_total",
        "Summarizer candidates found by target level."
    );
    describe_counter!(
        "summarizer_summaries_written_total",
        "Summarizer write attempts by target level and result."
    );
    describe_histogram!(
        "summarizer_duration_seconds",
        "Summarizer pass latency in seconds."
    );
    describe_counter!(
        "context_pack_requests_total",
        "Context pack build requests."
    );
    describe_counter!("context_pack_cache_hits_total", "Context pack cache hits.");
    describe_counter!(
        "context_pack_cache_misses_total",
        "Context pack cache misses."
    );
    describe_counter!(
        "context_cache_replacements_total",
        "Older context cache versions replaced after refresh."
    );
    describe_histogram!(
        "context_pack_build_duration_seconds",
        "Context pack build latency in seconds."
    );
    describe_histogram!(
        "context_pack_tokens_estimate",
        "Estimated context pack token count."
    );
    describe_counter!(
        "context_pack_items_injected_total",
        "Context pack items injected by memory layer."
    );
    describe_counter!("retrieval_hits_total", "Retrieval hits by source.");
    describe_counter!(
        "inference_tokens_total",
        "Inference tokens by kind and model."
    );
    describe_counter!(
        "context_cache_stale_invalidations_total",
        "Context cache invalidations triggered by new memory."
    );
    describe_counter!(
        "execution_artifacts_total",
        "Structured execution artifact events written."
    );
    describe_counter!(
        "failure_signatures_total",
        "Canonical failure fingerprints recorded."
    );
    describe_counter!(
        "patch_lifecycle_total",
        "Patch lifecycle outcomes recorded."
    );
    describe_counter!(
        "validation_results_total",
        "Deterministic validation outcomes recorded."
    );
    describe_counter!(
        "remediation_reuse_total",
        "Prior remediations surfaced in context packs."
    );
    describe_counter!(
        "task_retries_total",
        "Historical retry behavior by task type, outcome, and trigger category."
    );
    describe_gauge!(
        "task_retries",
        "Currently active retry chains across tasks."
    );
    describe_counter!("memory_promotions_total", "Memory promotion decisions.");
    describe_gauge!(
        "memory_source_coverage",
        "Fraction of promoted memory with sources."
    );
    describe_counter!(
        "rate_limited_total",
        "Requests rejected by the per-key rate limiter."
    );
    describe_counter!(
        "sampling_param_overrides_total",
        "Sampling parameter override hook executions by bounded parameter and reason."
    );
    describe_counter!(
        "trajectory_results_total",
        "Completed trajectory results by bounded operational status."
    );
    describe_counter!(
        "trajectory_attempts_total",
        "Attempts observed across completed trajectories."
    );
    describe_counter!(
        "trajectory_validation_failures_total",
        "Failed trajectory validations by bounded validator type."
    );
    describe_counter!(
        "trajectory_tokens_total",
        "Trajectory model tokens by direction."
    );
    describe_counter!(
        "trajectory_features_total",
        "Operational feature records written or updated."
    );
    describe_counter!(
        "feature_failure_classes_total",
        "Operational feature failure classes observed."
    );
    describe_counter!(
        "operational_constraints_injected_total",
        "Operational constraints injected into context packs."
    );
    describe_counter!(
        "operational_constraints_suppressed_total",
        "Operational constraints suppressed by bounded deterministic reasons."
    );
    describe_histogram!(
        "feature_extraction_duration_seconds",
        "Inline feature extraction latency in seconds."
    );
    describe_counter!(
        "feature_extraction_failures_total",
        "Feature extraction failures by stage."
    );
    describe_counter!(
        "feature_tag_schema_version_unknown_total",
        "Detection tags skipped due to unknown schema versions."
    );
    describe_counter!(
        "process_cpu_seconds_total",
        "CPU seconds consumed by this process."
    );
    describe_gauge!(
        "process_resident_memory_bytes",
        "Resident memory used by this process."
    );
    describe_gauge!(
        "process_start_time_seconds",
        "Process start time since Unix epoch."
    );
    describe_gauge!(
        "process_open_fds",
        "Open file descriptors for this process."
    );
}

pub fn prime_metrics(registry: &MetricsRegistry, default_model: &str, sentiment_loaded: bool) {
    for result in ["accepted", "rejected"] {
        counter!("auth_attempts_total", "result" => result).increment(0);
        counter!("memory_promotions_total", "result" => result).increment(0);
    }
    for endpoint in [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/models",
        "/sessions/start",
        "/events/append",
        "/v1/validations",
        "/context/pack",
        "/search",
        "/metrics",
    ] {
        gauge!("http_requests_in_flight", "endpoint" => endpoint).set(0.0);
    }
    for path in ["chat_completions", "messages", "models"] {
        counter!("upstream_litellm_requests_total", "path" => path, "status" => "none")
            .increment(0);
        histogram!("upstream_litellm_duration_seconds", "path" => path).record(0.0);
        for kind in ["timeout", "connection", "4xx", "5xx", "parse"] {
            counter!("upstream_litellm_errors_total", "path" => path, "kind" => kind).increment(0);
        }
    }
    for path in ["chat_completions"] {
        counter!("summarizer_upstream_requests_total", "path" => path, "status" => "none")
            .increment(0);
        histogram!("summarizer_upstream_duration_seconds", "path" => path).record(0.0);
        for kind in ["timeout", "connection", "4xx", "5xx", "parse"] {
            counter!("summarizer_upstream_errors_total", "path" => path, "kind" => kind)
                .increment(0);
        }
    }
    for path in ["chat_completions", "messages"] {
        histogram!("stream_first_token_seconds", "path" => path).record(0.0);
        histogram!("stream_duration_seconds", "path" => path).record(0.0);
        counter!("stream_disconnects_total", "path" => path, "reason" => "client_disconnect")
            .increment(0);
    }
    for op in [
        "create_session",
        "find_or_create_session",
        "insert_event",
        "count_events_for_repo",
        "get_context_evidence",
        "search_events_fts",
        "hydrate_active_search_hits",
        "get_event_chain",
        "get_trajectory",
        "get_trajectory_attempts",
        "get_trajectory_result",
        "latest_trajectory_event_for_session",
        "idle_trajectory_ids",
        "insert_trajectory_result_event",
        "get_failure_history",
        "insert_error_record",
        "get_active_errors",
        "record_token_usage",
        "check_ready",
    ] {
        histogram!("db_query_duration_seconds", "op" => op).record(0.0);
        counter!("db_query_errors_total", "op" => op).increment(0);
    }
    for op in ["upsert", "search", "health", "create_collection"] {
        counter!("qdrant_requests_total", "op" => op, "status" => "none").increment(0);
        histogram!("qdrant_request_duration_seconds", "op" => op).record(0.0);
    }
    counter!("embedder_inferences_total").increment(0);
    histogram!("embedder_inference_duration_seconds").record(0.0);
    histogram!("embedder_input_tokens").record(0.0);
    if sentiment_loaded {
        for verdict in ["negative", "non_negative"] {
            counter!("sentiment_inferences_total", "verdict" => verdict).increment(0);
        }
        histogram!("sentiment_inference_duration_seconds").record(0.0);
    }
    for level in ["1", "2", "3"] {
        counter!("summarizer_candidates_found_total", "target_level" => level).increment(0);
        histogram!("summarizer_duration_seconds", "target_level" => level).record(0.0);
        for result in ["success", "failure"] {
            counter!(
                "summarizer_summaries_written_total",
                "target_level" => level,
                "result" => result
            )
            .increment(0);
        }
    }
    counter!("summarizer_ticks_total").increment(0);
    counter!("context_pack_requests_total").increment(0);
    counter!("context_pack_cache_hits_total").increment(0);
    counter!("context_pack_cache_misses_total").increment(0);
    histogram!("context_pack_build_duration_seconds").record(0.0);
    histogram!("context_pack_tokens_estimate").record(0.0);
    for layer in ["l0", "l1", "l2", "l3", "failed_attempt", "remediation"] {
        counter!("context_pack_items_injected_total", "layer" => layer).increment(0);
    }
    counter!("context_pack_items_injected_total", "layer" => "failure_history").increment(0);
    counter!("context_pack_items_injected_total", "layer" => "operational_constraints")
        .increment(0);
    counter!("context_cache_replacements_total").increment(0);
    for source in ["semantic", "fts", "deduped"] {
        counter!("retrieval_hits_total", "source" => source).increment(0);
    }
    for kind in ["processed", "cached", "generated"] {
        counter!("inference_tokens_total", "kind" => kind, "model" => default_model.to_string())
            .increment(0);
    }
    counter!("context_cache_stale_invalidations_total").increment(0);
    for event_type in crate::execution_feedback::EXECUTION_EVENT_TYPES {
        for success in ["true", "false"] {
            counter!(
                "execution_artifacts_total",
                "event_type" => event_type,
                "success" => success
            )
            .increment(0);
        }
    }
    for outcome in ["applied", "rejected", "reverted"] {
        counter!("patch_lifecycle_total", "outcome" => outcome).increment(0);
    }
    for validator in [
        "cargo",
        "pytest",
        "npm test",
        "eslint",
        "tsc",
        "mypy",
        "ruff",
        "terraform",
        "kubectl",
        "other",
    ] {
        for result in ["pass", "fail"] {
            counter!(
                "validation_results_total",
                "validator" => validator,
                "result" => result
            )
            .increment(0);
        }
    }
    gauge!("task_retries").set(0.0);
    for task_type in ["coding", "infra", "recall", "general"] {
        for outcome in ["succeeded", "abandoned", "still_active"] {
            for trigger_category in [
                "borrow_checker",
                "import_error",
                "type_error",
                "parse_error",
                "unknown",
                "none",
            ] {
                counter!(
                    "task_retries_total",
                    "task_type" => task_type,
                    "outcome" => outcome,
                    "trigger_category" => trigger_category
                )
                .increment(0);
            }
        }
    }
    gauge!("memory_source_coverage").set(registry.snapshot().memory_source_coverage);
    counter!(
        "sampling_param_overrides_total",
        "parameter" => crate::sampling::PARAM_NONE,
        "reason" => crate::sampling::REASON_NOOP
    )
    .increment(0);
    for parameter in [
        crate::sampling::PARAM_TEMPERATURE,
        crate::sampling::PARAM_TOP_P,
        crate::sampling::PARAM_MAX_TOKENS,
        crate::sampling::PARAM_SEED,
    ] {
        counter!(
            "sampling_param_overrides_total",
            "parameter" => parameter,
            "reason" => crate::sampling::REASON_OVERRIDDEN_BY_ORCHESTRATOR
        )
        .increment(0);
    }
    for status in crate::trajectory::FINAL_STATUSES {
        counter!("trajectory_results_total", "status" => status).increment(0);
    }
    counter!("trajectory_attempts_total").increment(0);
    for validator_type in crate::execution_feedback::VALIDATOR_TYPES {
        counter!("trajectory_validation_failures_total", "validator_type" => validator_type)
            .increment(0);
    }
    for direction in ["input", "output"] {
        counter!("trajectory_tokens_total", "direction" => direction).increment(0);
    }
    counter!("trajectory_features_total").increment(0);
    for failure_class in crate::feature_extraction::FEATURE_FAILURE_CLASSES {
        counter!("feature_failure_classes_total", "failure_class" => failure_class).increment(0);
    }
    for constraint_type in crate::feature_extraction::OPERATIONAL_CONSTRAINT_TYPES {
        counter!(
            "operational_constraints_injected_total",
            "constraint_type" => constraint_type
        )
        .increment(0);
        for reason in crate::feature_extraction::OPERATIONAL_SUPPRESSION_REASONS {
            counter!(
                "operational_constraints_suppressed_total",
                "constraint_type" => constraint_type,
                "reason" => reason
            )
            .increment(0);
        }
    }
    histogram!("feature_extraction_duration_seconds").record(0.0);
    for stage in ["extraction", "constraint_build", "persistence"] {
        counter!("feature_extraction_failures_total", "stage" => stage).increment(0);
    }
    counter!("feature_tag_schema_version_unknown_total").increment(0);
}

pub async fn http_metrics_middleware(req: Request, next: Next) -> Response {
    let endpoint = req
        .extensions()
        .get::<MatchedPath>()
        .map(|path| path.as_str().to_string())
        .unwrap_or_else(|| req.uri().path().to_string());
    let method = req.method().as_str().to_string();
    let started = Instant::now();

    gauge!("http_requests_in_flight", "endpoint" => endpoint.clone()).increment(1.0);
    let response = next.run(req).await;
    gauge!("http_requests_in_flight", "endpoint" => endpoint.clone()).decrement(1.0);

    let status = response.status().as_u16().to_string();
    counter!(
        "http_requests_total",
        "endpoint" => endpoint.clone(),
        "method" => method.clone(),
        "status" => status
    )
    .increment(1);
    histogram!(
        "http_request_duration_seconds",
        "endpoint" => endpoint,
        "method" => method
    )
    .record(started.elapsed().as_secs_f64());

    response
}

pub fn prometheus_content_type() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        "text/plain; version=0.0.4".parse().unwrap(),
    );
    headers
}

pub fn record_auth_attempt(accepted: bool) {
    let result = if accepted { "accepted" } else { "rejected" };
    counter!("auth_attempts_total", "result" => result).increment(1);
}

pub fn record_context_pack(registry: &MetricsRegistry, stats: &crate::state::ContextPackStats) {
    counter!("context_pack_requests_total").increment(1);
    if stats.cache_hit {
        counter!("context_pack_cache_hits_total").increment(1);
    } else {
        counter!("context_pack_cache_misses_total").increment(1);
    }
    histogram!("context_pack_build_duration_seconds").record(stats.build_ms as f64 / 1000.0);
    histogram!("context_pack_tokens_estimate").record(stats.context_tokens_estimate as f64);
    record_items("l0", stats.l0_items_injected);
    record_items("l1", stats.l1_items_injected);
    record_items("l2", stats.l2_items_injected);
    record_items("l3", stats.l3_items_injected);
    record_items("failed_attempt", stats.failed_attempts_injected);
    record_items("remediation", stats.remediations_injected);
    record_items("failure_history", stats.failure_history_items_injected);
    record_items(
        "operational_constraints",
        stats.operational_constraints_injected,
    );
    for signature in &stats.failure_history_remediation_signatures {
        counter!(
            "remediation_reuse_total",
            "signature" => crate::execution_feedback::bounded_failure_signature_label(signature)
        )
        .increment(1);
    }
    counter!("retrieval_hits_total", "source" => "semantic")
        .increment(stats.retrieval_semantic_hits as u64);
    counter!("retrieval_hits_total", "source" => "fts").increment(stats.retrieval_fts_hits as u64);
    counter!("retrieval_hits_total", "source" => "deduped")
        .increment(stats.retrieval_deduped_hits as u64);

    let mut metrics = registry.inner.write().unwrap();
    metrics.context_pack_requests += 1;
    if stats.cache_hit {
        metrics.context_cache_hits += 1;
    } else {
        metrics.context_cache_misses += 1;
    }
    metrics.context_pack_build_ms_total += stats.build_ms;
    metrics.context_pack_chars_total += stats.context_chars as u64;
    metrics.context_pack_tokens_estimate_total += stats.context_tokens_estimate as u64;
    metrics.l0_items_injected += stats.l0_items_injected as u64;
    metrics.l1_items_injected += stats.l1_items_injected as u64;
    metrics.l2_items_injected += stats.l2_items_injected as u64;
    metrics.l3_items_injected += stats.l3_items_injected as u64;
    metrics.failed_attempts_injected += stats.failed_attempts_injected as u64;
    metrics.remediations_injected += stats.remediations_injected as u64;
    metrics.failure_history_items_injected += stats.failure_history_items_injected as u64;
    metrics.operational_constraints_injected += stats.operational_constraints_injected as u64;
    metrics.remediation_reuse += stats.failure_history_remediation_signatures.len() as u64;
    metrics.retrieval_semantic_hits += stats.retrieval_semantic_hits as u64;
    metrics.retrieval_fts_hits += stats.retrieval_fts_hits as u64;
    metrics.retrieval_deduped_hits += stats.retrieval_deduped_hits as u64;
}

pub fn record_execution_artifact(event: &crate::db::AgentEvent) {
    if !crate::execution_feedback::EXECUTION_EVENT_TYPES.contains(&event.event_type.as_str()) {
        return;
    }

    // Only persisted execution events reach here. Keep labels bounded to the
    // canonical event taxonomy and validator/signature normalizers.
    let success = event
        .metadata
        .get("success")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let success_label = if success { "true" } else { "false" };
    counter!(
        "execution_artifacts_total",
        "event_type" => event.event_type.clone(),
        "success" => success_label
    )
    .increment(1);

    let payload = &event.metadata["payload"];
    if !success && crate::execution_feedback::is_failure_outcome_event_type(&event.event_type) {
        let signature = payload["signature"].as_str().unwrap_or("unknown");
        let category = crate::execution_feedback::retry_trigger_category_from_payload(payload);
        counter!(
            "failure_signatures_total",
            "signature" => crate::execution_feedback::bounded_failure_signature_label(signature),
            "category" => category
        )
        .increment(1);

        if event.event_type != crate::execution_feedback::EVENT_TYPE_TOOL_RESULT {
            let task = event
                .metadata
                .get("task")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let task_type = crate::execution_feedback::task_retry_type(task);
            // This is a coarse process-local gauge for outstanding failure
            // chains. It is intentionally bounded by normalized outcome writes
            // and remediations, not by task names, to avoid user-controlled labels.
            gauge!("task_retries").increment(1.0);
            counter!(
                "task_retries_total",
                "task_type" => task_type,
                "outcome" => "still_active",
                "trigger_category" => category
            )
            .increment(1);
        }
    }

    match event.event_type.as_str() {
        crate::execution_feedback::EVENT_TYPE_PATCH_RESULT => {
            let outcome = match event.metadata["payload"]["outcome"].as_str() {
                Some("applied") => "applied",
                Some("reverted") => "reverted",
                _ => "rejected",
            };
            counter!("patch_lifecycle_total", "outcome" => outcome).increment(1);
        }
        crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT
        | crate::execution_feedback::EVENT_TYPE_TEST_RESULT
        | crate::execution_feedback::EVENT_TYPE_LINT_RESULT
        | crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT => {
            // Different artifact types name the validator differently. Collapse
            // them into the bounded validator label set before recording.
            let validator = payload["validator_name"]
                .as_str()
                .or_else(|| payload["framework"].as_str())
                .or_else(|| payload["tool_name"].as_str())
                .or_else(|| payload["target"].as_str())
                .unwrap_or("other");
            let result = if success { "pass" } else { "fail" };
            counter!(
                "validation_results_total",
                "validator" => crate::execution_feedback::bounded_validator_label(validator),
                "result" => result
            )
            .increment(1);
            if !success
                && event.event_role.as_deref()
                    == Some(crate::trajectory::EventRole::Validation.as_str())
            {
                let validator_type = bounded_validator_type_label(
                    payload["validator_type"].as_str().unwrap_or("other"),
                );
                counter!(
                    "trajectory_validation_failures_total",
                    "validator_type" => validator_type
                )
                .increment(1);
            }
        }
        crate::execution_feedback::EVENT_TYPE_REMEDIATION => {
            let task = event
                .metadata
                .get("task")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let signature = payload["signature"].as_str();
            gauge!("task_retries").decrement(1.0);
            counter!(
                "task_retries_total",
                "task_type" => crate::execution_feedback::task_retry_type(task),
                "outcome" => "succeeded",
                "trigger_category" => crate::execution_feedback::retry_trigger_category_from_signature(signature)
            )
            .increment(1);
        }
        _ => {}
    }
}

fn bounded_validator_type_label(value: &str) -> &'static str {
    match value {
        "compile" => "compile",
        "test" => "test",
        "lint" => "lint",
        "type_check" => "type_check",
        "schema" => "schema",
        "static_analysis" => "static_analysis",
        "other" => "other",
        _ => "other",
    }
}

pub fn record_trajectory_result(summary: &crate::trajectory::TrajectoryResultSummary) {
    counter!(
        "trajectory_results_total",
        "status" => summary.final_status.as_str()
    )
    .increment(1);
    counter!("trajectory_attempts_total").increment(summary.final_attempt_index.max(1) as u64);
    counter!("trajectory_tokens_total", "direction" => "input")
        .increment(summary.total_input_tokens.max(0) as u64);
    counter!("trajectory_tokens_total", "direction" => "output")
        .increment(summary.total_output_tokens.max(0) as u64);
}

pub fn record_trajectory_feature_written(registry: &MetricsRegistry) {
    counter!("trajectory_features_total").increment(1);
    registry.inner.write().unwrap().trajectory_features += 1;
}

pub fn record_feature_failure_class(failure_class: &str) {
    let failure_class = bounded_feature_failure_class(failure_class);
    counter!("feature_failure_classes_total", "failure_class" => failure_class).increment(1);
}

pub fn record_operational_constraint_injected(constraint_type: &str) {
    let constraint_type = bounded_operational_constraint_type(constraint_type);
    counter!("operational_constraints_injected_total", "constraint_type" => constraint_type)
        .increment(1);
}

pub fn record_operational_constraint_suppressed(constraint_type: &str, reason: &str) {
    let constraint_type = bounded_operational_constraint_type(constraint_type);
    let reason = bounded_operational_suppression_reason(reason);
    counter!(
        "operational_constraints_suppressed_total",
        "constraint_type" => constraint_type,
        "reason" => reason
    )
    .increment(1);
}

pub fn record_feature_extraction_duration(elapsed: Duration) {
    histogram!("feature_extraction_duration_seconds").record(elapsed.as_secs_f64());
}

pub fn record_feature_extraction_failure(stage: &'static str) {
    let stage = bounded_feature_stage(stage);
    counter!("feature_extraction_failures_total", "stage" => stage).increment(1);
}

pub fn record_unknown_tag_schema_version() {
    counter!("feature_tag_schema_version_unknown_total").increment(1);
}

fn bounded_feature_failure_class(value: &str) -> &'static str {
    match value {
        "tool_loop" => "tool_loop",
        "user_interruption" => "user_interruption",
        "missing_auth" => "missing_auth",
        "wrong_endpoint" => "wrong_endpoint",
        "summarization_failure" => "summarization_failure",
        "migration_failure" => "migration_failure",
        "context_pack_empty" => "context_pack_empty",
        "context_pack_truncated" => "context_pack_truncated",
        "high_input_tokens" => "high_input_tokens",
        "slow_upstream_model" => "slow_upstream_model",
        "empty_tool_use_message" => "empty_tool_use_message",
        "abandoned_before_model" => "abandoned_before_model",
        "single_model_abandoned_no_tools" => "single_model_abandoned_no_tools",
        "summarizer_shared_upstream" => "summarizer_shared_upstream",
        "other" => "other",
        _ => "other",
    }
}

fn bounded_operational_constraint_type(value: &str) -> &'static str {
    match value {
        "use_known_auth" => "use_known_auth",
        "use_known_endpoint" => "use_known_endpoint",
        "use_known_migration_fix" => "use_known_migration_fix",
        "avoid_tool_loop" => "avoid_tool_loop",
        "fix_context_retrieval" => "fix_context_retrieval",
        "reduce_context_bloat" => "reduce_context_bloat",
        "separate_summarizer_upstream" => "separate_summarizer_upstream",
        "handle_user_interruption" => "handle_user_interruption",
        "handle_summarization_failure" => "handle_summarization_failure",
        _ => "handle_summarization_failure",
    }
}

fn bounded_operational_suppression_reason(value: &str) -> &'static str {
    match value {
        "freshness_window" => "freshness_window",
        "recovery_detected" => "recovery_detected",
        "priority_cap" => "priority_cap",
        "token_budget" => "token_budget",
        _ => "token_budget",
    }
}

fn bounded_feature_stage(value: &str) -> &'static str {
    match value {
        "extraction" => "extraction",
        "constraint_build" => "constraint_build",
        "persistence" => "persistence",
        _ => "extraction",
    }
}

fn record_items(layer: &'static str, count: usize) {
    counter!("context_pack_items_injected_total", "layer" => layer).increment(count as u64);
}

pub fn record_tokens(registry: &MetricsRegistry, usage: &crate::state::TokenUsage, model: &str) {
    if usage.is_empty() {
        return;
    }
    counter!("inference_tokens_total", "kind" => "processed", "model" => model.to_string())
        .increment(usage.processed_tokens);
    counter!("inference_tokens_total", "kind" => "cached", "model" => model.to_string())
        .increment(usage.cached_tokens);
    counter!("inference_tokens_total", "kind" => "generated", "model" => model.to_string())
        .increment(usage.generated_tokens);

    let mut metrics = registry.inner.write().unwrap();
    metrics.processed_tokens += usage.processed_tokens;
    metrics.cached_tokens += usage.cached_tokens;
    metrics.generated_tokens += usage.generated_tokens;
}

pub fn record_cache_invalidation(registry: &MetricsRegistry) {
    counter!("context_cache_stale_invalidations_total").increment(1);
    registry.inner.write().unwrap().stale_cache_invalidations += 1;
}

pub fn record_context_cache_replacement(registry: &MetricsRegistry, replaced: usize) {
    counter!("context_cache_replacements_total").increment(replaced as u64);
    registry.inner.write().unwrap().context_cache_replacements += replaced as u64;
}

pub fn record_promotion(registry: &MetricsRegistry, accepted: bool, has_sources: bool) {
    let result = if accepted { "accepted" } else { "rejected" };
    counter!("memory_promotions_total", "result" => result).increment(1);

    let mut metrics = registry.inner.write().unwrap();
    metrics.promotion_attempts += 1;
    if accepted {
        metrics.promotion_accepted += 1;
    } else {
        metrics.promotion_rejected += 1;
    }
    metrics.memory_source_items += 1;
    if has_sources {
        metrics.memory_source_items_with_sources += 1;
    }
    metrics.memory_source_coverage =
        metrics.memory_source_items_with_sources as f64 / metrics.memory_source_items as f64;
    gauge!("memory_source_coverage").set(metrics.memory_source_coverage);
}

pub fn record_rate_limited(key_hash: &str) {
    counter!("rate_limited_total", "key_hash" => key_hash.to_string()).increment(1);
}

pub fn record_sampling_param_override(parameter: &'static str, reason: &'static str) {
    counter!("sampling_param_overrides_total", "parameter" => parameter, "reason" => reason)
        .increment(1);
}

pub fn record_db_query(op: &'static str, elapsed: Duration, success: bool) {
    histogram!("db_query_duration_seconds", "op" => op).record(elapsed.as_secs_f64());
    if !success {
        counter!("db_query_errors_total", "op" => op).increment(1);
    }
}

pub fn record_qdrant_request(op: &'static str, elapsed: Duration, status: &str) {
    histogram!("qdrant_request_duration_seconds", "op" => op).record(elapsed.as_secs_f64());
    counter!("qdrant_requests_total", "op" => op, "status" => status.to_string()).increment(1);
}

pub fn record_upstream_litellm(path: &'static str, elapsed: Duration, status: &str) {
    histogram!("upstream_litellm_duration_seconds", "path" => path).record(elapsed.as_secs_f64());
    counter!(
        "upstream_litellm_requests_total",
        "path" => path,
        "status" => status.to_string()
    )
    .increment(1);
}

pub fn record_upstream_litellm_error(path: &'static str, kind: &'static str) {
    counter!("upstream_litellm_errors_total", "path" => path, "kind" => kind).increment(1);
}

pub fn record_upstream_summarizer(path: &'static str, elapsed: Duration, status: &str) {
    histogram!("summarizer_upstream_duration_seconds", "path" => path)
        .record(elapsed.as_secs_f64());
    counter!(
        "summarizer_upstream_requests_total",
        "path" => path,
        "status" => status.to_string()
    )
    .increment(1);
}

pub fn record_upstream_summarizer_error(path: &'static str, kind: &'static str) {
    counter!("summarizer_upstream_errors_total", "path" => path, "kind" => kind).increment(1);
}

pub fn upstream_error_kind(status: reqwest::StatusCode) -> &'static str {
    if status.is_client_error() {
        "4xx"
    } else if status.is_server_error() {
        "5xx"
    } else {
        "parse"
    }
}

pub fn reqwest_error_kind(err: &reqwest::Error) -> &'static str {
    if err.is_timeout() {
        "timeout"
    } else if err.is_connect() {
        "connection"
    } else if err.is_status() {
        "5xx"
    } else {
        "parse"
    }
}

pub fn record_embedder_input_tokens(tokens: usize) {
    histogram!("embedder_input_tokens").record(tokens as f64);
}

pub fn record_embedder_inference(elapsed: Duration, success: bool) {
    if success {
        counter!("embedder_inferences_total").increment(1);
    }
    histogram!("embedder_inference_duration_seconds").record(elapsed.as_secs_f64());
}

pub fn record_sentiment(verdict: bool, elapsed: Duration) {
    let verdict = if verdict { "negative" } else { "non_negative" };
    counter!("sentiment_inferences_total", "verdict" => verdict).increment(1);
    histogram!("sentiment_inference_duration_seconds").record(elapsed.as_secs_f64());
}

pub fn record_summarizer_tick() {
    counter!("summarizer_ticks_total").increment(1);
}

pub fn record_summarizer_candidate(target_level: i32, count: usize) {
    counter!(
        "summarizer_candidates_found_total",
        "target_level" => target_level.to_string()
    )
    .increment(count as u64);
}

pub fn record_summarizer_duration(target_level: i32, elapsed: Duration) {
    histogram!(
        "summarizer_duration_seconds",
        "target_level" => target_level.to_string()
    )
    .record(elapsed.as_secs_f64());
}

pub fn record_summarizer_written(target_level: i32, success: bool) {
    let result = if success { "success" } else { "failure" };
    counter!(
        "summarizer_summaries_written_total",
        "target_level" => target_level.to_string(),
        "result" => result
    )
    .increment(1);
}

pub struct StreamTracker {
    path: &'static str,
    started: Instant,
    first_token_seen: bool,
    completed: bool,
}

impl StreamTracker {
    pub fn new(path: &'static str, started: Instant) -> Self {
        Self {
            path,
            started,
            first_token_seen: false,
            completed: false,
        }
    }

    pub fn first_token(&mut self) {
        if !self.first_token_seen {
            histogram!("stream_first_token_seconds", "path" => self.path)
                .record(self.started.elapsed().as_secs_f64());
            self.first_token_seen = true;
        }
    }

    pub fn finish(&mut self) {
        self.completed = true;
        histogram!("stream_duration_seconds", "path" => self.path)
            .record(self.started.elapsed().as_secs_f64());
    }

    pub fn fail(&mut self, reason: &'static str) {
        self.completed = true;
        histogram!("stream_duration_seconds", "path" => self.path)
            .record(self.started.elapsed().as_secs_f64());
        counter!("stream_disconnects_total", "path" => self.path, "reason" => reason).increment(1);
    }
}

impl Drop for StreamTracker {
    fn drop(&mut self) {
        if !self.completed {
            histogram!("stream_duration_seconds", "path" => self.path)
                .record(self.started.elapsed().as_secs_f64());
            counter!(
                "stream_disconnects_total",
                "path" => self.path,
                "reason" => "client_disconnect"
            )
            .increment(1);
        }
    }
}

pub fn record_pool_gauges(pool: &Pool) {
    let status = pool.status();
    gauge!("db_pool_size").set(status.max_size as f64);
    gauge!("db_pool_available").set(status.available as f64);
    gauge!("db_pool_waiters").set(status.waiting as f64);
}

pub fn record_process_metrics() {
    let metrics = process_metrics();
    counter!("process_cpu_seconds_total").absolute(metrics.cpu_seconds.floor() as u64);
    gauge!("process_resident_memory_bytes").set(metrics.resident_memory_bytes as f64);
    gauge!("process_start_time_seconds").set(metrics.start_time_seconds);
    gauge!("process_open_fds").set(metrics.open_fds as f64);
}

#[derive(Default)]
struct ProcessMetrics {
    cpu_seconds: f64,
    resident_memory_bytes: u64,
    start_time_seconds: f64,
    open_fds: u64,
}

fn process_metrics() -> ProcessMetrics {
    #[cfg(target_os = "linux")]
    {
        linux_process_metrics()
    }
    #[cfg(not(target_os = "linux"))]
    {
        ProcessMetrics {
            start_time_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or_default(),
            ..Default::default()
        }
    }
}

#[cfg(target_os = "linux")]
fn linux_process_metrics() -> ProcessMetrics {
    const CLOCK_TICKS_PER_SECOND: f64 = 100.0;
    const PAGE_SIZE_BYTES: u64 = 4096;

    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let after_comm = stat.rsplit_once(") ").map(|(_, rest)| rest).unwrap_or("");
    let fields: Vec<&str> = after_comm.split_whitespace().collect();
    let utime = fields
        .get(11)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_default();
    let stime = fields
        .get(12)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_default();
    let start_ticks = fields
        .get(19)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_default();
    let cpu_seconds = (utime + stime) / CLOCK_TICKS_PER_SECOND;

    let resident_memory_bytes = std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .nth(1)
                .and_then(|pages| pages.parse::<u64>().ok())
        })
        .map(|pages| pages * PAGE_SIZE_BYTES)
        .unwrap_or_default();

    let boot_time = std::fs::read_to_string("/proc/stat")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|line| {
                line.strip_prefix("btime ")
                    .and_then(|value| value.parse::<f64>().ok())
            })
        })
        .unwrap_or_default();

    let open_fds = std::fs::read_dir("/proc/self/fd")
        .map(|entries| entries.count() as u64)
        .unwrap_or_default();

    ProcessMetrics {
        cpu_seconds,
        resident_memory_bytes,
        start_time_seconds: boot_time + start_ticks / CLOCK_TICKS_PER_SECOND,
        open_fds,
    }
}
