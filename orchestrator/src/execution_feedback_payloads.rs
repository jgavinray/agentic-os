use regex::Regex;
use serde_json::{json, Value};
use std::sync::OnceLock;
use uuid::Uuid;

use crate::execution_feedback_fingerprints::{
    fingerprint, inline_signature_payload, inline_signature_payload_from_fingerprint,
    summarize_text, FailureFingerprint,
};
use crate::execution_feedback_parsing::{
    failure_summaries, first_number_before, infer_success, CapturedToolResult,
};

fn regex_cell(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("fingerprint regex must compile"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn normalize_validation_payload(
    mut payload: Value,
    validator_name: &str,
    validator_type: &str,
    command: &str,
    success: bool,
    exit_code: Option<i32>,
    duration_ms: Option<u64>,
    failure_excerpt: Option<String>,
) -> Value {
    let Some(obj) = payload.as_object_mut() else {
        return payload;
    };
    obj.entry("validator_name".to_string())
        .or_insert_with(|| json!(validator_name));
    obj.entry("validator_type".to_string())
        .or_insert_with(|| json!(validator_type));
    obj.entry("command".to_string())
        .or_insert_with(|| json!(command));
    obj.insert("success".to_string(), json!(success));
    obj.entry("exit_code".to_string())
        .or_insert_with(|| exit_code.map(Value::from).unwrap_or(Value::Null));
    obj.entry("duration_ms".to_string())
        .or_insert_with(|| duration_ms.map(Value::from).unwrap_or(Value::Null));
    let failure_signature = obj.get("signature").and_then(Value::as_str);
    obj.insert(
        "failure_signature".to_string(),
        failure_signature
            .filter(|signature| *signature != "unknown")
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    obj.entry("failure_excerpt".to_string()).or_insert_with(|| {
        failure_excerpt
            .filter(|excerpt| !excerpt.is_empty())
            .map(Value::from)
            .unwrap_or(Value::Null)
    });
    payload
}

pub fn tool_result_payload(result: &CapturedToolResult) -> Value {
    let success = infer_success(result.exit_code, &result.content);
    let fp = (!success).then(|| fingerprint(&result.content));
    tool_result_payload_with_fingerprint(result, success, fp.as_ref())
}

pub(crate) fn tool_result_payload_with_fingerprint(
    result: &CapturedToolResult,
    success: bool,
    fp: Option<&FailureFingerprint>,
) -> Value {
    inline_signature_payload_from_fingerprint(
        json!({
        "tool_name": result.tool_name,
        "exit_code": result.exit_code,
        "stdout_summary": result.stdout_summary,
        "stderr_summary": result.stderr_summary,
        "duration_ms": result.duration_ms,
        }),
        success,
        fp,
    )
}

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

#[allow(dead_code)]
pub fn patch_result_payload(
    files_touched: Vec<String>,
    outcome: &str,
    validation_event_ids: Vec<Uuid>,
) -> Value {
    json!({
        "files_touched": files_touched,
        "lines_added": 0,
        "lines_removed": 0,
        "patch_applied": outcome == "applied",
        "patch_reverted": outcome == "reverted",
        "outcome": outcome,
        "validation_event_ids": validation_event_ids,
    })
}

#[allow(dead_code)]
pub fn remediation_payload(
    failure_event_id: Uuid,
    resolving_event_id: Uuid,
    signature: &str,
) -> Value {
    json!({
        "failure_event_id": failure_event_id,
        "resolving_event_id": resolving_event_id,
        "signature": signature,
    })
}

pub(crate) fn compile_error_count(content: &str) -> usize {
    static RE: OnceLock<Regex> = OnceLock::new();
    regex_cell(&RE, r"(?m)^\s*error(?:\[[A-Za-z0-9]+\])?:")
        .find_iter(content)
        .count()
}

pub(crate) fn warning_count(content: &str) -> usize {
    static RE: OnceLock<Regex> = OnceLock::new();
    regex_cell(&RE, r"(?m)^\s*warning(?:\[.*?\])?:?")
        .find_iter(content)
        .count()
}
