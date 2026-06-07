use serde_json::Value;

use crate::state::TokenUsage;

pub(crate) fn extract_assistant_from_anthropic_response(resp: &Value) -> String {
    resp.get("content")
        .and_then(|v| v.as_array())
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|b| {
                    if b["type"].as_str() == Some("text") {
                        b["text"].as_str()
                    } else {
                        None
                    }
                })
                .collect::<String>()
        })
        .unwrap_or_default()
        .chars()
        .take(500)
        .collect()
}

pub(crate) fn extract_token_usage_from_anthropic_sse(raw: &str) -> TokenUsage {
    let mut usage = TokenUsage::default();
    for line in raw.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        match v["type"].as_str() {
            Some("message_start") => {
                let u = &v["message"]["usage"];
                usage.processed_tokens = u["input_tokens"].as_u64().unwrap_or(0);
                usage.cached_tokens = u["cache_read_input_tokens"].as_u64().unwrap_or(0);
            }
            Some("message_delta") => {
                let u = &v["usage"];
                usage.generated_tokens = u["output_tokens"].as_u64().unwrap_or(0);
            }
            _ => {}
        }
    }
    usage
}

pub(crate) fn extract_assistant_from_anthropic_sse(raw: &str) -> String {
    let mut content = String::new();
    for line in raw.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if v["type"].as_str() == Some("content_block_delta")
            && v["delta"]["type"].as_str() == Some("text_delta")
        {
            if let Some(text) = v["delta"]["text"].as_str() {
                content.push_str(text);
            }
        }
    }
    content.chars().take(500).collect()
}

pub(crate) fn extract_assistant_from_sse(raw: &str) -> String {
    let mut content = String::new();
    for line in raw.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data.trim() == "[DONE]" {
                break;
            }
            if let Ok(v) = serde_json::from_str::<Value>(data) {
                if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
                    content.push_str(delta);
                }
            }
        }
    }
    content
}

pub(crate) fn extract_token_usage_from_sse(raw: &str) -> TokenUsage {
    let mut usage = TokenUsage::default();
    for line in raw.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data.trim() == "[DONE]" {
            break;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        let chunk_usage = TokenUsage::from_openai_value(&value);
        usage.processed_tokens = usage.processed_tokens.max(chunk_usage.processed_tokens);
        usage.cached_tokens = usage.cached_tokens.max(chunk_usage.cached_tokens);
        usage.generated_tokens = usage.generated_tokens.max(chunk_usage.generated_tokens);
    }
    usage
}

pub(crate) fn optional_token_usage_from_sse(raw: &str) -> (Option<i64>, Option<i64>) {
    let mut input = None;
    let mut output = None;
    for line in raw.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data.trim() == "[DONE]" {
            break;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        let (chunk_input, chunk_output) =
            crate::trajectory::optional_token_counts_from_value(&value);
        input = chunk_input.or(input);
        output = chunk_output.or(output);
    }
    (input, output)
}
