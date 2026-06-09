use serde_json::{json, Value};

use crate::execution_feedback_fingerprints::{fingerprint, signature_category, FailureFingerprint};

pub(crate) fn inline_signature_payload_from_fingerprint(
    mut payload: Value,
    success: bool,
    fingerprint: Option<&FailureFingerprint>,
) -> Value {
    let Some(obj) = payload.as_object_mut() else {
        return payload;
    };

    if success {
        obj.insert("signature".to_string(), Value::Null);
        obj.insert("signature_category".to_string(), Value::Null);
        obj.insert("fingerprint_version".to_string(), Value::Null);
        return payload;
    }

    let fp = fingerprint.expect("failed outcome payloads require a fingerprint");
    obj.insert("signature".to_string(), json!(fp.signature));
    obj.insert(
        "signature_category".to_string(),
        json!(signature_category(fp)),
    );
    obj.insert("fingerprint_version".to_string(), json!(fp.version));
    payload
}

pub fn inline_signature_payload(payload: Value, success: bool, raw_error_text: &str) -> Value {
    if success {
        return inline_signature_payload_from_fingerprint(payload, true, None);
    }

    let fp = fingerprint(raw_error_text);
    inline_signature_payload_from_fingerprint(payload, false, Some(&fp))
}

pub fn outcome_raw_text_from_payload(
    payload: &Value,
    fallback_raw_excerpt: Option<&str>,
) -> String {
    let mut parts = Vec::new();
    if let Some(raw) = fallback_raw_excerpt.filter(|raw| !raw.trim().is_empty()) {
        parts.push(raw.to_string());
    }
    if let Some(reason) = payload.get("failure_reason").and_then(Value::as_str) {
        parts.push(reason.to_string());
    }
    if let Some(stderr) = payload.get("stderr_summary").and_then(Value::as_str) {
        parts.push(stderr.to_string());
    }
    if let Some(stdout) = payload.get("stdout_summary").and_then(Value::as_str) {
        parts.push(stdout.to_string());
    }
    for key in ["failure_summaries", "findings"] {
        if let Some(values) = payload.get(key).and_then(Value::as_array) {
            for value in values {
                if let Some(text) = value.as_str() {
                    parts.push(text.to_string());
                }
            }
        }
    }

    if parts.iter().any(|part| !part.trim().is_empty()) {
        parts.join("\n")
    } else {
        payload.to_string()
    }
}

pub fn backfill_inline_signature_metadata(
    metadata: &Value,
    fallback_raw_excerpt: Option<&str>,
) -> Option<Value> {
    let payload = metadata.get("payload")?;
    let raw = outcome_raw_text_from_payload(payload, fallback_raw_excerpt);
    let mut updated = metadata.clone();
    updated["payload"] = inline_signature_payload(payload.clone(), false, &raw);
    Some(updated)
}
