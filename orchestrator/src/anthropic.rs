use serde_json::Value;

/// Extract the last user message text content from an Anthropic request for persistence.
pub(crate) fn extract_user_content_from_anthropic(req: &Value) -> String {
    req.get("messages")
        .and_then(|v| v.as_array())
        .and_then(|msgs| {
            msgs.iter()
                .rfind(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        })
        .map(|m| match m.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(blocks)) => blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        })
        .unwrap_or_default()
        .chars()
        .take(500)
        .collect()
}
