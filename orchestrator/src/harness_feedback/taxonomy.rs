use std::collections::BTreeSet;

pub const HARNESS_FEEDBACK_SCHEMA_VERSION: u32 = 1;

pub const HARNESS_SIGNAL_TYPES: [&str; 27] = [
    "benchmark_trace",
    "invalid_path",
    "repeated_invalid_path",
    "path_typo_context_ledger",
    "poisoned_path_attempt",
    "requested_path_mismatch",
    "read_missing_path",
    "edit_failed",
    "gate_blocked",
    "gate_recovery_failed",
    "gate_recovered",
    "forbidden_tool_fallback",
    "command_boundary_violation",
    "patch_invalid",
    "fake_success_claim",
    "false_path_success",
    "success_without_change",
    "missing_required_validation",
    "no_progress_timeout",
    "diagnosis_success",
    "hard_stop_compliance",
    "continued_after_stop",
    "scope_violation",
    "misleading_green_tests",
    "memory_poison_indicator",
    "self_correction_after_violation",
    "unknown",
];

pub const HARNESS_QUARANTINE_REASONS: [&str; 9] = [
    "benchmark_session",
    "poisoned_path",
    "repeated_invalid_path",
    "fake_success_claim",
    "tool_recovery_failed",
    "contract_violation",
    "patch_invalid",
    "manual_quarantine",
    "memory_poison_indicator",
];

pub const HARNESS_LEARNING_STATUSES: [&str; 3] = ["recorded", "quarantined", "ignored"];
pub const HARNESS_GUARDRAIL_ACTIONS: [&str; 4] = ["allow", "warn", "block", "terminate"];
pub const HARNESS_GUARDRAIL_REASONS: [&str; 9] = [
    "none",
    "poisoned_path",
    "repeated_invalid_path",
    "fake_success_claim",
    "tool_recovery_failed",
    "contract_violation",
    "patch_invalid",
    "memory_poison_indicator",
    "benchmark_session",
];

pub(crate) const MEMORY_POLICY_INCLUDE: &str = "include";
pub(crate) const MEMORY_POLICY_EXCLUDE: &str = "exclude_from_context";
pub const GUARDRAIL_ACTION_ALLOW: &str = "allow";
pub const GUARDRAIL_ACTION_WARN: &str = "warn";
pub const GUARDRAIL_ACTION_BLOCK: &str = "block";
pub const GUARDRAIL_ACTION_TERMINATE: &str = "terminate";
pub(crate) const GUARDRAIL_REASON_NONE: &str = "none";

pub(crate) fn guardrail_action_for_signals(signals: &BTreeSet<&'static str>) -> &'static str {
    if signals.contains("continued_after_stop") {
        return GUARDRAIL_ACTION_TERMINATE;
    }
    if signals.contains("poisoned_path_attempt")
        || signals.contains("path_typo_context_ledger")
        || signals.contains("requested_path_mismatch")
        || signals.contains("false_path_success")
        || signals.contains("forbidden_tool_fallback")
        || signals.contains("command_boundary_violation")
        || signals.contains("patch_invalid")
        || signals.contains("success_without_change")
        || signals.contains("missing_required_validation")
    {
        return GUARDRAIL_ACTION_BLOCK;
    }
    if signals.contains("misleading_green_tests")
        || signals.contains("self_correction_after_violation")
    {
        return GUARDRAIL_ACTION_WARN;
    }
    GUARDRAIL_ACTION_ALLOW
}

pub fn bounded_signal_type(value: &str) -> &'static str {
    match value {
        "benchmark_trace" | "benchmark" => "benchmark_trace",
        "invalid_path" => "invalid_path",
        "repeated_invalid_path" => "repeated_invalid_path",
        "path_typo_context_ledger" | "context_leder" | "context_ledler" => {
            "path_typo_context_ledger"
        }
        "poisoned_path_attempt" => "poisoned_path_attempt",
        "requested_path_mismatch" => "requested_path_mismatch",
        "read_missing_path" | "missing_path" => "read_missing_path",
        "edit_failed" => "edit_failed",
        "gate_blocked" | "gateguard_blocked" => "gate_blocked",
        "gate_recovery_failed" => "gate_recovery_failed",
        "gate_recovered" => "gate_recovered",
        "forbidden_tool_fallback" => "forbidden_tool_fallback",
        "command_boundary_violation" => "command_boundary_violation",
        "patch_invalid" | "invalid_patch" => "patch_invalid",
        "fake_success_claim" => "fake_success_claim",
        "false_path_success" => "false_path_success",
        "success_without_change" => "success_without_change",
        "missing_required_validation" => "missing_required_validation",
        "no_progress_timeout" | "no_progress" => "no_progress_timeout",
        "diagnosis_success" => "diagnosis_success",
        "hard_stop_compliance" => "hard_stop_compliance",
        "continued_after_stop" => "continued_after_stop",
        "scope_violation" => "scope_violation",
        "misleading_green_tests" => "misleading_green_tests",
        "memory_poison_indicator" => "memory_poison_indicator",
        "self_correction_after_violation" => "self_correction_after_violation",
        _ => "unknown",
    }
}

pub fn bounded_quarantine_reason(value: &str) -> &'static str {
    match value {
        "benchmark_session" => "benchmark_session",
        "poisoned_path" => "poisoned_path",
        "repeated_invalid_path" => "repeated_invalid_path",
        "fake_success_claim" => "fake_success_claim",
        "tool_recovery_failed" => "tool_recovery_failed",
        "contract_violation" => "contract_violation",
        "patch_invalid" => "patch_invalid",
        "manual_quarantine" => "manual_quarantine",
        "memory_poison_indicator" => "memory_poison_indicator",
        _ => "manual_quarantine",
    }
}

pub fn bounded_learning_status(value: &str) -> &'static str {
    match value {
        "recorded" => "recorded",
        "quarantined" => "quarantined",
        "ignored" => "ignored",
        _ => "ignored",
    }
}

pub fn bounded_guardrail_action(value: &str) -> &'static str {
    match value {
        "allow" => GUARDRAIL_ACTION_ALLOW,
        "warn" => GUARDRAIL_ACTION_WARN,
        "block" => GUARDRAIL_ACTION_BLOCK,
        "terminate" => GUARDRAIL_ACTION_TERMINATE,
        _ => GUARDRAIL_ACTION_BLOCK,
    }
}

pub fn bounded_guardrail_reason(value: &str) -> &'static str {
    match value {
        "none" => GUARDRAIL_REASON_NONE,
        "benchmark_session" => "benchmark_session",
        "poisoned_path" => "poisoned_path",
        "repeated_invalid_path" => "repeated_invalid_path",
        "fake_success_claim" => "fake_success_claim",
        "tool_recovery_failed" => "tool_recovery_failed",
        "contract_violation" => "contract_violation",
        "patch_invalid" => "patch_invalid",
        "memory_poison_indicator" => "memory_poison_indicator",
        _ => "memory_poison_indicator",
    }
}
