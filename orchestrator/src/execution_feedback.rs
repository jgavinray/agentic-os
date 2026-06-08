//! Deterministic execution feedback capture.
//!
//! This module deliberately avoids LLM calls. It turns observed tool output and
//! explicit validation reports into structured `agent_events` payloads using
//! regexes, fixed rule ordering, and small parser helpers. The resulting events
//! can then flow through the existing Postgres, Qdrant, FTS, and metrics paths.

use crate::db::AgentEvent;
use serde_json::json;
use uuid::Uuid;

pub use crate::execution_feedback_chain::{group_by_parent, patch_validation_ids_resolve};
pub use crate::execution_feedback_events::{
    build_execution_event, is_failure_outcome_event_type, ExecutionEventContext,
    ExecutionEventKind, EVENT_TYPE_COMPILE_RESULT, EVENT_TYPE_LINT_RESULT, EVENT_TYPE_PATCH_RESULT,
    EVENT_TYPE_REMEDIATION, EVENT_TYPE_TEST_RESULT, EVENT_TYPE_TOOL_RESULT,
    EVENT_TYPE_VALIDATION_RESULT, EXECUTION_EVENT_TYPES, FAILURE_OUTCOME_EVENT_TYPES,
};
use crate::execution_feedback_fingerprints::inline_signature_payload_from_fingerprint;
pub use crate::execution_feedback_fingerprints::{
    backfill_inline_signature_metadata, extract_failure_signatures, fingerprint,
    inline_signature_payload, outcome_raw_text_from_payload, signature_category, summarize_text,
    FailureFingerprint, FINGERPRINT_VERSION,
};
pub use crate::execution_feedback_labels::{
    bounded_failure_category_label, bounded_failure_signature_label, bounded_validator_label,
    retry_trigger_category_from_payload, retry_trigger_category_from_signature, task_retry_type,
};
use crate::execution_feedback_parsing::{first_number_before, infer_success};
pub use crate::execution_feedback_parsing::{
    tool_results_from_sse, tool_results_from_value, CapturedToolResult,
};
use crate::execution_feedback_payloads::{
    compile_error_count, compile_result_payload_with_fingerprint,
    lint_result_payload_with_fingerprint, normalize_validation_payload,
    test_result_payload_with_fingerprint, tool_result_payload_with_fingerprint, warning_count,
};
pub use crate::execution_feedback_payloads::{
    compile_result_payload, lint_result_payload, patch_result_payload, remediation_payload,
    test_result_payload, tool_result_payload, validation_result_payload,
};
pub use crate::execution_feedback_reports::{
    events_for_validation_report, ValidationReportRequest, ValidationReportResponse,
};
use crate::execution_feedback_validators::validator_type_str;
pub use crate::execution_feedback_validators::{
    classify_validator, validate_validator_type, ValidationKind, ValidatorSpec, VALIDATOR_TYPES,
};

pub fn events_for_tool_result(
    ctx: &ExecutionEventContext,
    result: &CapturedToolResult,
) -> Vec<AgentEvent> {
    let tool_success = infer_success(result.exit_code, &result.content);
    let tool_fp = (!tool_success).then(|| fingerprint(&result.content));
    let tool_event = build_execution_event(
        ctx,
        ExecutionEventKind::ToolResult,
        tool_success,
        tool_result_payload_with_fingerprint(result, tool_success, tool_fp.as_ref()),
    );
    let mut events = vec![tool_event.clone()];

    if let Some(spec) = classify_validator(&result.tool_name, &result.content) {
        // A validator tool produces two linked memories: the observed tool run
        // and the normalized compile/test/lint/validation result derived from it.
        let validation_ctx = ctx.child_of(Uuid::parse_str(&tool_event.id).unwrap());
        let (kind, payload, success) = match spec.kind {
            ValidationKind::Compile | ValidationKind::TypeCheck => {
                let language = if spec.validator == "tsc" {
                    "typescript"
                } else if spec.validator == "mypy" {
                    "python"
                } else {
                    "rust"
                };
                let error_count = compile_error_count(&result.content);
                let warning_count = warning_count(&result.content);
                let success = result.exit_code == 0 && error_count == 0;
                let mut payload = compile_result_payload_with_fingerprint(
                    language,
                    spec.validator,
                    result.exit_code,
                    error_count,
                    warning_count,
                    success,
                    tool_fp.as_ref(),
                );
                payload["validator_type"] = json!(validator_type_str(spec.kind));
                payload["duration_ms"] = json!(result.duration_ms);
                (ExecutionEventKind::CompileResult, payload, success)
            }
            ValidationKind::Test => {
                let passed = first_number_before("passed", &result.content).unwrap_or(0);
                let failed = first_number_before("failed", &result.content).unwrap_or(0);
                let skipped = first_number_before("skipped", &result.content)
                    .or_else(|| first_number_before("ignored", &result.content))
                    .unwrap_or(0);
                let total = first_number_before("total", &result.content)
                    .unwrap_or(passed + failed + skipped);
                let success = result.exit_code == 0 && failed == 0;
                let mut payload = test_result_payload_with_fingerprint(
                    spec.validator,
                    total,
                    passed,
                    failed,
                    skipped,
                    success,
                    tool_fp.as_ref(),
                    &result.content,
                );
                payload["duration_ms"] = json!(result.duration_ms);
                (ExecutionEventKind::TestResult, payload, success)
            }
            ValidationKind::Lint => {
                let error_count = first_number_before("error", &result.content)
                    .or_else(|| first_number_before("errors", &result.content))
                    .unwrap_or_else(|| compile_error_count(&result.content) as u64);
                let warning_count = first_number_before("warning", &result.content)
                    .or_else(|| first_number_before("warnings", &result.content))
                    .unwrap_or_else(|| warning_count(&result.content) as u64);
                let success = result.exit_code == 0 && error_count == 0;
                let mut payload = lint_result_payload_with_fingerprint(
                    spec.validator,
                    error_count,
                    warning_count,
                    success,
                    tool_fp.as_ref(),
                    &result.content,
                );
                payload["duration_ms"] = json!(result.duration_ms);
                (ExecutionEventKind::LintResult, payload, success)
            }
            ValidationKind::Schema | ValidationKind::StaticAnalysis | ValidationKind::Other => {
                let success = infer_success(result.exit_code, &result.content);
                let failure_reason = if success {
                    String::new()
                } else {
                    summarize_text(&result.content, 500)
                };
                let payload = inline_signature_payload_from_fingerprint(
                    json!({
                        "validator_name": spec.validator,
                        "pass": success,
                        "failure_reason": failure_reason,
                    }),
                    success,
                    tool_fp.as_ref(),
                );
                let payload = normalize_validation_payload(
                    payload,
                    spec.validator,
                    validator_type_str(spec.kind),
                    spec.validator,
                    success,
                    Some(result.exit_code),
                    Some(result.duration_ms),
                    (!success).then(|| summarize_text(&result.content, 500)),
                );
                (ExecutionEventKind::ValidationResult, payload, success)
            }
        };
        let validation_event = build_execution_event(&validation_ctx, kind, success, payload);
        events.push(validation_event);
    }

    events
}

#[cfg(test)]
#[path = "execution_feedback_capture_tests.rs"]
mod capture_tests;

#[cfg(test)]
#[path = "execution_feedback_event_tests.rs"]
mod event_tests;

#[cfg(test)]
#[path = "execution_feedback_fingerprint_tests.rs"]
mod fingerprint_tests;
