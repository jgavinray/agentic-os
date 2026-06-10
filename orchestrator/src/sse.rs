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

/// Extract the final stop/finish reason from a buffered SSE stream.
///
/// Anthropic streams carry `delta.stop_reason` on `message_delta` events;
/// OpenAI streams carry `choices[0].finish_reason`. The last non-null value
/// wins, matching non-streaming semantics.
pub(crate) fn optional_stop_reason_from_sse(raw: &str) -> Option<String> {
    let mut stop_reason = None;
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
        let chunk_reason = value
            .get("delta")
            .and_then(|delta| delta.get("stop_reason"))
            .and_then(Value::as_str)
            .or_else(|| {
                value
                    .get("choices")
                    .and_then(Value::as_array)
                    .and_then(|choices| choices.first())
                    .and_then(|choice| choice.get("finish_reason"))
                    .and_then(Value::as_str)
            });
        if let Some(reason) = chunk_reason {
            stop_reason = Some(reason.to_string());
        }
    }
    stop_reason
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

#[cfg(test)]
mod stop_reason_tests {
    use super::*;

    #[test]
    fn extracts_anthropic_stop_reason_from_message_delta() {
        let raw = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":12}}\n";
        assert_eq!(
            optional_stop_reason_from_sse(raw).as_deref(),
            Some("end_turn")
        );
    }

    #[test]
    fn extracts_openai_finish_reason_and_last_value_wins() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n",
            "data: [DONE]\n",
        );
        assert_eq!(optional_stop_reason_from_sse(raw).as_deref(), Some("stop"));
    }

    #[test]
    fn missing_stop_reason_returns_none() {
        let raw = "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n";
        assert_eq!(optional_stop_reason_from_sse(raw), None);
    }
}
