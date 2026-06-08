use serde_json::{json, Value};

pub use crate::execution_feedback_compile_payloads::compile_result_payload;
pub(crate) use crate::execution_feedback_compile_payloads::compile_result_payload_with_fingerprint;
use crate::execution_feedback_fingerprints::{inline_signature_payload, summarize_text};
pub use crate::execution_feedback_lint_payloads::lint_result_payload;
pub(crate) use crate::execution_feedback_lint_payloads::lint_result_payload_with_fingerprint;
pub use crate::execution_feedback_test_payloads::test_result_payload;
pub(crate) use crate::execution_feedback_test_payloads::test_result_payload_with_fingerprint;
pub(crate) use crate::execution_feedback_validation_payloads::{
    compile_error_count, normalize_validation_payload, warning_count,
};

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
