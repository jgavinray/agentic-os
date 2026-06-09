use serde_json::json;

use crate::execution_feedback_events::{ExecutionEventKind, EVENT_TYPE_TOOL_RESULT};
use crate::execution_feedback_fingerprints::{
    fingerprint, inline_signature_payload_from_fingerprint, summarize_text, FailureFingerprint,
};
use crate::execution_feedback_parsing::{first_number_before, infer_success, CapturedToolResult};
use crate::execution_feedback_payloads::{
    compile_error_count, compile_result_payload_with_fingerprint,
    lint_result_payload_with_fingerprint, normalize_validation_payload,
    test_result_payload_with_fingerprint, tool_result_payload_with_fingerprint, warning_count,
};
use crate::execution_feedback_validators::{
    classify_validator, validator_type_str, ValidationKind, ValidatorSpec,
};

pub(crate) struct ToolEventDerivation {
    pub(crate) tool_kind: ExecutionEventKind,
    pub(crate) tool_success: bool,
    pub(crate) tool_fingerprint: Option<FailureFingerprint>,
    pub(crate) validator_event: Option<ValidatorEventDerivation>,
}

pub(crate) struct ValidatorEventDerivation {
    pub(crate) kind: ExecutionEventKind,
    pub(crate) payload: serde_json::Value,
    pub(crate) success: bool,
}

pub(crate) fn derive_tool_event(result: &CapturedToolResult) -> ToolEventDerivation {
    let tool_success = infer_success(result.exit_code, &result.content);
    let tool_fingerprint = (!tool_success).then(|| fingerprint(&result.content));
    let validator_event = classify_validator(&result.tool_name, &result.content)
        .map(|spec| derive_validator_event(result, &spec, tool_fingerprint.as_ref()));

    ToolEventDerivation {
        tool_kind: ExecutionEventKind::ToolResult,
        tool_success,
        tool_fingerprint,
        validator_event,
    }
}

pub(crate) fn tool_payload(
    result: &CapturedToolResult,
    derivation: &ToolEventDerivation,
) -> serde_json::Value {
    debug_assert_eq!(derivation.tool_kind.as_str(), EVENT_TYPE_TOOL_RESULT);
    tool_result_payload_with_fingerprint(
        result,
        derivation.tool_success,
        derivation.tool_fingerprint.as_ref(),
    )
}

fn derive_validator_event(
    result: &CapturedToolResult,
    spec: &ValidatorSpec,
    tool_fingerprint: Option<&FailureFingerprint>,
) -> ValidatorEventDerivation {
    match spec.kind {
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
                tool_fingerprint,
            );
            payload["validator_type"] = json!(validator_type_str(spec.kind));
            payload["duration_ms"] = json!(result.duration_ms);
            ValidatorEventDerivation {
                kind: ExecutionEventKind::CompileResult,
                payload,
                success,
            }
        }
        ValidationKind::Test => {
            let passed = first_number_before("passed", &result.content).unwrap_or(0);
            let failed = first_number_before("failed", &result.content).unwrap_or(0);
            let skipped = first_number_before("skipped", &result.content)
                .or_else(|| first_number_before("ignored", &result.content))
                .unwrap_or(0);
            let total =
                first_number_before("total", &result.content).unwrap_or(passed + failed + skipped);
            let success = result.exit_code == 0 && failed == 0;
            let mut payload = test_result_payload_with_fingerprint(
                spec.validator,
                total,
                passed,
                failed,
                skipped,
                success,
                tool_fingerprint,
                &result.content,
            );
            payload["duration_ms"] = json!(result.duration_ms);
            ValidatorEventDerivation {
                kind: ExecutionEventKind::TestResult,
                payload,
                success,
            }
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
                tool_fingerprint,
                &result.content,
            );
            payload["duration_ms"] = json!(result.duration_ms);
            ValidatorEventDerivation {
                kind: ExecutionEventKind::LintResult,
                payload,
                success,
            }
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
                tool_fingerprint,
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
            ValidatorEventDerivation {
                kind: ExecutionEventKind::ValidationResult,
                payload,
                success,
            }
        }
    }
}
