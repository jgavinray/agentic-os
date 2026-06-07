use axum::extract::MatchedPath;
use axum::extract::Request;
use axum::http::{header, HeaderMap};
use axum::middleware::Next;
use axum::response::Response;
use metrics::{counter, gauge, histogram};
use serde::Serialize;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

pub use crate::telemetry_model_services::{
    record_embedder_inference, record_embedder_input_tokens, record_sentiment,
    record_summarizer_candidate, record_summarizer_duration, record_summarizer_tick,
    record_summarizer_written,
};
pub use crate::telemetry_request_classification::{
    record_request_classification, record_request_classification_backfill_run,
    record_request_classification_unknown_label, record_request_classification_write,
    record_request_complexity, record_request_live_policy_action,
    record_request_live_policy_bypass, record_request_risk_flag,
    record_request_route_recommendation,
};
pub use crate::telemetry_runtime_metrics::{record_pool_gauges, record_process_metrics};
pub use crate::telemetry_setup::{install_recorder, prime_metrics};
pub use crate::telemetry_streaming::StreamTracker;
pub use crate::telemetry_tool_mediation::{record_tool_authorization, record_tool_menu_outcome};
pub use crate::telemetry_upstream::{
    record_upstream_litellm, record_upstream_litellm_error, record_upstream_summarizer,
    record_upstream_summarizer_error, reqwest_error_kind, upstream_error_kind,
};

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
    pub tool_mediation_decisions: u64,
    pub tool_mediation_tools: u64,
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

    pub(crate) fn increment_tool_mediation_decisions(&self) {
        self.inner.write().unwrap().tool_mediation_decisions += 1;
    }

    pub(crate) fn increment_tool_mediation_tools(&self) {
        self.inner.write().unwrap().tool_mediation_tools += 1;
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
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

pub fn record_harness_feedback_signal(signal_type: &str) {
    let signal_type = crate::harness_feedback::bounded_signal_type(signal_type);
    counter!("harness_feedback_signals_total", "signal_type" => signal_type).increment(1);
}

pub fn record_harness_feedback_quarantine(reason: &str) {
    let reason = crate::harness_feedback::bounded_quarantine_reason(reason);
    counter!("harness_feedback_quarantined_total", "reason" => reason).increment(1);
}

pub fn record_harness_feedback_learning(status: &str) {
    let status = crate::harness_feedback::bounded_learning_status(status);
    counter!("harness_feedback_learning_records_total", "status" => status).increment(1);
}

pub fn record_harness_feedback_repair_run(result: &'static str) {
    let result = match result {
        "success" => "success",
        "failure" => "failure",
        _ => "failure",
    };
    counter!("harness_feedback_repair_runs_total", "result" => result).increment(1);
}

pub fn record_harness_guardrail_decision(action: &str, reason: &str) {
    let action = crate::harness_feedback::bounded_guardrail_action(action);
    let reason = crate::harness_feedback::bounded_guardrail_reason(reason);
    counter!(
        "harness_guardrail_decisions_total",
        "action" => action,
        "reason" => reason
    )
    .increment(1);
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

pub fn record_vllm_cache_delta(delta: &crate::vllm_metrics::VllmCacheDelta, model: &str) {
    counter!(
        "vllm_prefix_cache_tokens_total",
        "kind" => "queries",
        "model" => model.to_string()
    )
    .increment(delta.prefix_cache_queries_delta.max(0) as u64);
    counter!(
        "vllm_prefix_cache_tokens_total",
        "kind" => "hits",
        "model" => model.to_string()
    )
    .increment(delta.prefix_cache_hits_delta.max(0) as u64);
    for (source, value) in [
        ("total", delta.prompt_tokens_total_delta),
        ("cached", delta.prompt_tokens_cached_delta),
        ("local_compute", delta.prompt_tokens_local_compute_delta),
        ("local_cache_hit", delta.prompt_tokens_local_cache_hit_delta),
        (
            "external_kv_transfer",
            delta.prompt_tokens_external_kv_delta,
        ),
    ] {
        counter!(
            "vllm_prompt_tokens_by_cache_source_total",
            "source" => source,
            "model" => model.to_string()
        )
        .increment(value.max(0) as u64);
    }
}

pub fn record_vllm_cache_token_flow(
    input_tokens: u64,
    output_tokens: u64,
    provider_cache: crate::litellm::ProviderCacheCounters,
    model: &str,
) {
    for (kind, value) in [
        ("request_input", input_tokens),
        ("request_output", output_tokens),
        (
            "provider_cached",
            provider_cache.provider_cached_tokens.max(0) as u64,
        ),
        (
            "provider_cache_created",
            provider_cache.provider_cache_created_tokens.max(0) as u64,
        ),
        (
            "provider_cache_read",
            provider_cache.provider_cache_read_tokens.max(0) as u64,
        ),
    ] {
        counter!(
            "vllm_prompt_tokens_by_cache_source_total",
            "source" => kind,
            "model" => model.to_string()
        )
        .increment(value);
    }
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
