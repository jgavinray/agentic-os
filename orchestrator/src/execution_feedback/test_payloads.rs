use serde_json::{json, Value};

use crate::execution_feedback_fingerprints::{
    fingerprint, inline_signature_payload_from_fingerprint, FailureFingerprint,
};
use crate::execution_feedback_parsing::{failure_summaries, first_number_before};
use crate::execution_feedback_validation_payloads::normalize_validation_payload;

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
