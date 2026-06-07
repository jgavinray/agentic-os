use serde_json::{json, Value};

use crate::execution_feedback_fingerprints::{
    fingerprint, inline_signature_payload, inline_signature_payload_from_fingerprint,
    summarize_text, FailureFingerprint,
};
use crate::execution_feedback_parsing::{failure_summaries, first_number_before};
pub(crate) use crate::execution_feedback_validation_payloads::{
    compile_error_count, normalize_validation_payload, warning_count,
};

pub fn compile_result_payload(
    language: &str,
    target: &str,
    exit_code: i32,
    content: &str,
) -> Value {
    let error_count = compile_error_count(content);
    let warning_count = warning_count(content);
    let success = exit_code == 0 && error_count == 0;
    let fp = (!success).then(|| fingerprint(content));
    compile_result_payload_with_fingerprint(
        language,
        target,
        exit_code,
        error_count,
        warning_count,
        success,
        fp.as_ref(),
    )
}

pub(crate) fn compile_result_payload_with_fingerprint(
    language: &str,
    target: &str,
    exit_code: i32,
    error_count: usize,
    warning_count: usize,
    success: bool,
    fp: Option<&FailureFingerprint>,
) -> Value {
    let payload = inline_signature_payload_from_fingerprint(
        json!({
        "language": language,
        "target": target,
        "exit_code": exit_code,
        "error_count": error_count,
        "warning_count": warning_count,
        }),
        success,
        fp,
    );
    normalize_validation_payload(
        payload,
        target,
        "compile",
        target,
        success,
        Some(exit_code),
        None,
        fp.map(|fp| fp.raw_excerpt.clone()),
    )
}

pub fn test_result_payload(framework: &str, content: &str) -> Value {
    let passed = first_number_before("passed", content).unwrap_or(0);
    let failed = first_number_before("failed", content).unwrap_or(0);
    let skipped = first_number_before("skipped", content)
        .or_else(|| first_number_before("ignored", content))
        .unwrap_or(0);
    let total = first_number_before("total", content).unwrap_or(passed + failed + skipped);
    let success = failed == 0;
    let fp = (!success).then(|| fingerprint(content));
    test_result_payload_with_fingerprint(
        framework,
        total,
        passed,
        failed,
        skipped,
        success,
        fp.as_ref(),
        content,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn test_result_payload_with_fingerprint(
    framework: &str,
    total: u64,
    passed: u64,
    failed: u64,
    skipped: u64,
    success: bool,
    fp: Option<&FailureFingerprint>,
    content: &str,
) -> Value {
    let payload = inline_signature_payload_from_fingerprint(
        json!({
        "framework": framework,
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "failure_summaries": failure_summaries(content),
        }),
        success,
        fp,
    );
    normalize_validation_payload(
        payload,
        framework,
        "test",
        framework,
        success,
        None,
        None,
        fp.map(|fp| fp.raw_excerpt.clone()),
    )
}

pub fn lint_result_payload(tool_name: &str, content: &str) -> Value {
    let error_count = first_number_before("error", content)
        .or_else(|| first_number_before("errors", content))
        .unwrap_or_else(|| compile_error_count(content) as u64);
    let warning_count = first_number_before("warning", content)
        .or_else(|| first_number_before("warnings", content))
        .unwrap_or_else(|| warning_count(content) as u64);
    let success = error_count == 0;
    let fp = (!success).then(|| fingerprint(content));
    lint_result_payload_with_fingerprint(
        tool_name,
        error_count,
        warning_count,
        success,
        fp.as_ref(),
        content,
    )
}

pub(crate) fn lint_result_payload_with_fingerprint(
    tool_name: &str,
    error_count: u64,
    warning_count: u64,
    success: bool,
    fp: Option<&FailureFingerprint>,
    content: &str,
) -> Value {
    let payload = inline_signature_payload_from_fingerprint(
        json!({
        "tool_name": tool_name,
        "error_count": error_count,
        "warning_count": warning_count,
        "findings": failure_summaries(content),
        }),
        success,
        fp,
    );
    normalize_validation_payload(
        payload,
        tool_name,
        "lint",
        tool_name,
        success,
        None,
        None,
        fp.map(|fp| fp.raw_excerpt.clone()),
    )
}

pub fn validation_result_payload(validator_name: &str, pass: bool, failure_reason: &str) -> Value {
    let payload = inline_signature_payload(
        json!({
        "validator_name": validator_name,
        "pass": pass,
        "failure_reason": failure_reason,
        }),
        pass,
        failure_reason,
    );
    normalize_validation_payload(
        payload,
        validator_name,
        "other",
        validator_name,
        pass,
        None,
        None,
        (!pass).then(|| summarize_text(failure_reason, 500)),
    )
}
