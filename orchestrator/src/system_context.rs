use serde_json::Value;

pub(crate) fn inject_system_context(payload: &mut Value, context: &str) {
    if let Some(messages) = payload.get_mut("messages").and_then(|v| v.as_array_mut()) {
        if let Some(first) = messages.first_mut() {
            if first.get("role").and_then(|r| r.as_str()) == Some("system") {
                let existing = first["content"].as_str().unwrap_or("").to_string();
                first["content"] = Value::String(format!("{existing}\n\n---\n{context}"));
                return;
            }
        }
        messages.insert(0, serde_json::json!({"role": "system", "content": context}));
    }
}

pub(crate) fn anthropic_text_block(text: impl Into<String>) -> Value {
    serde_json::json!({"type": "text", "text": text.into()})
}

fn anthropic_cacheable_text_block(text: impl Into<String>) -> Value {
    serde_json::json!({
        "type": "text",
        "text": text.into(),
        "cache_control": {"type": "ephemeral"}
    })
}

pub(crate) fn existing_anthropic_system_blocks(payload: &Value) -> Vec<Value> {
    match payload.get("system") {
        Some(Value::String(s)) if !s.trim().is_empty() => vec![anthropic_text_block(s.clone())],
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| {
                if block.get("type").and_then(|value| value.as_str()) == Some("text")
                    && block
                        .get("text")
                        .and_then(|value| value.as_str())
                        .is_some_and(|text| !text.trim().is_empty())
                {
                    Some(block.clone())
                } else {
                    None
                }
            })
            .collect(),
        _ => vec![],
    }
}

pub(crate) fn inject_system_context_anthropic(payload: &mut Value, context: &str) {
    let (stable_prefix, dynamic_tail) = if context.starts_with("== Stable Context Artifacts ==") {
        crate::litellm::split_context_prefix_tail(context)
    } else {
        ("", context)
    };
    let mut blocks = existing_anthropic_system_blocks(payload);
    if !stable_prefix.trim().is_empty() {
        blocks.push(anthropic_cacheable_text_block(stable_prefix.to_string()));
    }
    if !dynamic_tail.trim().is_empty() {
        blocks.push(anthropic_text_block(dynamic_tail.to_string()));
    }
    if blocks.is_empty() {
        return;
    }
    payload["system"] = Value::Array(blocks);
}
