use serde_json::{json, Value};
use uuid::Uuid;

use crate::execution_feedback_fingerprints::{
    fingerprint, inline_signature_payload_from_fingerprint, FailureFingerprint,
};
use crate::execution_feedback_parsing::{infer_success, CapturedToolResult};
pub(crate) use crate::execution_feedback_validation_result_payloads::{
    compile_error_count, compile_result_payload_with_fingerprint,
    lint_result_payload_with_fingerprint, normalize_validation_payload,
    test_result_payload_with_fingerprint, warning_count,
};
pub use crate::execution_feedback_validation_result_payloads::{
    compile_result_payload, lint_result_payload, test_result_payload, validation_result_payload,
};

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
