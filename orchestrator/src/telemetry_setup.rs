use metrics::{describe_counter, describe_gauge, describe_histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};

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
        "vllm_prefix_cache_tokens_total",
        "vLLM prefix-cache query and hit token deltas observed around orchestrated requests."
    );
    describe_counter!(
        "vllm_prompt_tokens_by_cache_source_total",
        "vLLM prompt token deltas by backend cache source observed around orchestrated requests."
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
        "harness_feedback_signals_total",
        "Deterministic harness feedback signals recorded on persisted events."
    );
    describe_counter!(
        "harness_feedback_quarantined_total",
        "Persisted events marked as excluded from future context memory."
    );
    describe_counter!(
        "harness_feedback_learning_records_total",
        "Harness feedback records available for learning-oriented analysis."
    );
    describe_counter!(
        "harness_feedback_repair_runs_total",
        "Harness feedback startup and background repair runs by bounded result."
    );
    describe_counter!(
        "harness_guardrail_decisions_total",
        "Deterministic runtime guardrail decisions by bounded action and reason."
    );
    describe_counter!(
        "request_classification_backfill_runs_total",
        "Request classification startup and operator backfill runs by bounded result."
    );
    describe_counter!(
        "request_classifications_written_total",
        "Request classification persistence attempts by bounded result."
    );
    describe_counter!(
        "request_classification_unknown_labels_total",
        "Request classification rows with bounded unknown labels by field."
    );
    describe_counter!(
        "request_classifications_total",
        "Request classifications by bounded intent, domain, and route."
    );
    describe_counter!(
        "request_route_recommendations_total",
        "Shadow route recommendations by bounded route."
    );
    describe_counter!(
        "request_risk_flags_total",
        "Request classification risk flags by bounded risk."
    );
    describe_counter!(
        "request_complexity_total",
        "Request classification complexity by bounded complexity."
    );
    describe_counter!(
        "request_live_policy_actions_total",
        "Feature-flagged live request policy actions by bounded action and reason."
    );
    describe_counter!(
        "request_live_policy_bypassed_total",
        "Requests bypassing live request policy by bounded reason."
    );
    describe_counter!(
        "tool_mediation_decisions_total",
        "Deterministic tool menu and tool-call decisions by bounded decision and reason."
    );
    describe_counter!(
        "tool_mediation_tools_total",
        "Tools observed by deterministic mediation by bounded action and capability."
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
