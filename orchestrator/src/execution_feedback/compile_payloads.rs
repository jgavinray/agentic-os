use serde_json::{json, Value};

use crate::execution_feedback_fingerprints::{
    fingerprint, inline_signature_payload_from_fingerprint, FailureFingerprint,
};
use crate::execution_feedback_validation_payloads::{
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
