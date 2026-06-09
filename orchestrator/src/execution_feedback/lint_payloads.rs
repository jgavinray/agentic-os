use serde_json::{json, Value};

use crate::execution_feedback_fingerprints::{
    fingerprint, inline_signature_payload_from_fingerprint, FailureFingerprint,
};
use crate::execution_feedback_parsing::{failure_summaries, first_number_before};
use crate::execution_feedback_validation_payloads::{
    compile_error_count, normalize_validation_payload, warning_count,
};

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
