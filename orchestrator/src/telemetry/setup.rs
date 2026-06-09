use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::PrometheusHandle;

use crate::telemetry::MetricsRegistry;
use crate::telemetry_descriptions::describe_metrics;
use crate::telemetry_prime_feedback::{
    prime_execution_feedback_metrics, prime_feature_feedback_metrics, prime_trajectory_metrics,
};
use crate::telemetry_prime_runtime::{
    prime_dependency_metrics, prime_request_entrypoint_metrics, prime_upstream_metrics,
};
use crate::telemetry_recorder::install_prometheus_recorder;

pub fn install_recorder() -> Result<PrometheusHandle, anyhow::Error> {
    let handle = install_prometheus_recorder()?;
    describe_metrics();
    Ok(handle)
}

pub fn prime_metrics(registry: &MetricsRegistry, default_model: &str, sentiment_loaded: bool) {
    prime_request_entrypoint_metrics();
    prime_upstream_metrics();
    prime_dependency_metrics(sentiment_loaded);
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
    prime_execution_feedback_metrics();
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
    prime_trajectory_metrics();
    prime_feature_feedback_metrics();
    crate::telemetry_request_policy_prime::prime_request_policy_metrics();
}
