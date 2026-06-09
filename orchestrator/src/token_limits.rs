use serde_json::Value;

use crate::state::{configured_default_max_tokens, configured_max_max_tokens, MAX_MAX_TOKENS};

pub(crate) fn enforce_min_max_tokens(req: &mut Value) {
    let requested = req.get("max_tokens").and_then(|v| v.as_u64());
    let default_max_tokens = configured_default_max_tokens();
    let max_max_tokens = configured_max_max_tokens();
    let effective = requested
        .unwrap_or(default_max_tokens)
        .clamp(1, max_max_tokens);
    let requested_for_log = requested.unwrap_or(0);
    if requested_for_log != effective {
        tracing::warn!(
            requested_max_tokens = requested_for_log,
            effective_max_tokens = effective,
            default_max_tokens = default_max_tokens,
            max_max_tokens = max_max_tokens,
            "clamped max_tokens for backend capacity"
        );
    }
    req["max_tokens"] = Value::from(effective);
}

fn number_after_marker(text: &str, marker: &str) -> Option<u64> {
    let start = text.find(marker)? + marker.len();
    let digits: String = text[start..]
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

pub(crate) fn context_window_retry_max_tokens(error_body: &str) -> Option<u64> {
    let max_context = number_after_marker(error_body, "maximum context length is")?;
    let requested_output = number_after_marker(error_body, "requested")?;
    let input_tokens = number_after_marker(error_body, "prompt contains at least")?;
    let remaining = max_context.checked_sub(input_tokens)?;
    if remaining == 0 || remaining >= requested_output {
        return None;
    }
    // Leave a little space because vLLM reports "at least" for prompt tokens.
    Some(remaining.saturating_sub(256).max(1).min(MAX_MAX_TOKENS))
}

pub(crate) fn set_max_tokens(req: &mut Value, max_tokens: u64) {
    req["max_tokens"] = Value::from(max_tokens);
}
