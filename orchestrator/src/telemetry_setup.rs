use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::PrometheusHandle;

use crate::telemetry::MetricsRegistry;
use crate::telemetry_descriptions::describe_metrics;
use crate::telemetry_recorder::install_prometheus_recorder;

pub fn install_recorder() -> Result<PrometheusHandle, anyhow::Error> {
    let handle = install_prometheus_recorder()?;
    describe_metrics();
    Ok(handle)
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
        "/harness/guardrail",
        "/tools/authorize",
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
    for signal_type in crate::harness_feedback::HARNESS_SIGNAL_TYPES {
        counter!("harness_feedback_signals_total", "signal_type" => signal_type).increment(0);
    }
    for reason in crate::harness_feedback::HARNESS_QUARANTINE_REASONS {
        counter!("harness_feedback_quarantined_total", "reason" => reason).increment(0);
    }
    for status in crate::harness_feedback::HARNESS_LEARNING_STATUSES {
        counter!("harness_feedback_learning_records_total", "status" => status).increment(0);
    }
    for result in ["success", "failure"] {
        counter!("harness_feedback_repair_runs_total", "result" => result).increment(0);
    }
    for action in crate::harness_feedback::HARNESS_GUARDRAIL_ACTIONS {
        for reason in crate::harness_feedback::HARNESS_GUARDRAIL_REASONS {
            counter!(
                "harness_guardrail_decisions_total",
                "action" => action,
                "reason" => reason
            )
            .increment(0);
        }
    }
    crate::telemetry_request_policy_prime::prime_request_policy_metrics();
}
