use metrics::{counter, histogram};
use std::time::Duration;

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
