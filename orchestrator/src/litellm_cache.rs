use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::litellm::CachePolicySnapshot;

pub fn exact_cache_decision(
    endpoint: &str,
    req: &Value,
    disabled_by_config: bool,
) -> CachePolicySnapshot {
    let bypass = if disabled_by_config {
        Some("disabled_by_config")
    } else if endpoint != "chat_completions" && endpoint != "messages" {
        Some("unsupported_endpoint")
    } else if req.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        Some("streaming")
    } else if req.get("tools").is_some() {
        Some("tools_present")
    } else if req.get("functions").is_some() {
        Some("functions_present")
    } else if req.get("tool_choice").is_some() {
        Some("tool_choice_present")
    } else if req.get("function_call").is_some() {
        Some("function_call_present")
    } else if req.get("n").and_then(Value::as_u64).is_some_and(|n| n > 1) {
        Some("n_gt_1")
    } else if req.get("temperature").is_none() {
        Some("temperature_missing")
    } else if req
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
        != 0.0
    {
        Some("temperature_nonzero")
    } else {
        None
    };
    CachePolicySnapshot {
        cache_backend: Some("litellm_exact_response_cache".to_string()),
        cache_policy_enabled: bypass.is_none(),
        cache_bypass_reason: bypass.map(str::to_string),
    }
}

pub fn context_pack_hash(context: &str) -> String {
    let stable = json!({ "context": context });
    let bytes = serde_json::to_vec(&stable).expect("context hash serialization cannot fail");
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

pub fn context_prefix_hashes(context: &str) -> (Option<String>, Option<String>) {
    let (stable_prefix, dynamic_tail) = split_context_prefix_tail(context);
    (
        (!stable_prefix.is_empty()).then(|| context_pack_hash(stable_prefix)),
        (!dynamic_tail.is_empty()).then(|| context_pack_hash(dynamic_tail)),
    )
}

pub fn split_context_prefix_tail(context: &str) -> (&str, &str) {
    if let Some(idx) = context.find("\nRepository: ") {
        context.split_at(idx + 1)
    } else if context.starts_with("Repository: ") {
        ("", context)
    } else {
        (context, "")
    }
}

pub fn sse_chunk_has_non_empty_data(chunk: &[u8]) -> bool {
    let text = String::from_utf8_lossy(chunk);
    text.lines().any(|line| {
        let Some(data) = line.strip_prefix("data:") else {
            return !line.trim().is_empty();
        };
        let data = data.trim();
        !data.is_empty() && data != "[DONE]"
    })
}
