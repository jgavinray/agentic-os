use bytes::Bytes;
use serde_json::{json, Value};

// ── Error ───────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct TranslationError {
    pub http_status: u16,
    pub error_type: &'static str,
    pub message: String,
}

impl TranslationError {
    fn invalid(msg: impl Into<String>) -> Self {
        Self {
            http_status: 400,
            error_type: "invalid_request_error",
            message: msg.into(),
        }
    }
}

// ── Request translation: Anthropic → OpenAI ─────────────────────

/// Translate an Anthropic Messages request body into an OpenAI chat/completions body.
/// Returns Err if required fields are missing or unsupported content blocks are present.
pub(crate) fn anthropic_to_openai(req: Value) -> Result<Value, TranslationError> {
    // Validate required fields.
    if req.get("model").is_none() {
        return Err(TranslationError::invalid("missing required field: model"));
    }
    if req.get("max_tokens").is_none() {
        return Err(TranslationError::invalid("missing required field: max_tokens"));
    }
    let messages = req
        .get("messages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| TranslationError::invalid("missing required field: messages"))?
        .clone();

    // Build OpenAI messages array, starting with optional system message.
    let mut oai_messages: Vec<Value> = Vec::new();

    if let Some(system) = req.get("system") {
        let system_text = extract_system_text(system)?;
        oai_messages.push(json!({"role": "system", "content": system_text}));
    }

    // Translate each message's content.
    for msg in &messages {
        let role = msg
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("user");
        let content = normalize_content(msg.get("content"))?;
        oai_messages.push(json!({"role": role, "content": content}));
    }

    // Build the OpenAI request.
    let mut oai = json!({
        "model": req["model"],
        "messages": oai_messages,
        "max_tokens": req["max_tokens"],
    });

    // Pass-through fields.
    for field in ["temperature", "top_p", "stream"] {
        if let Some(v) = req.get(field) {
            oai[field] = v.clone();
        }
    }

    // stop_sequences → stop
    if let Some(v) = req.get("stop_sequences") {
        oai["stop"] = v.clone();
    }

    // top_k and metadata are silently dropped.

    Ok(oai)
}

fn extract_system_text(system: &Value) -> Result<String, TranslationError> {
    if let Some(s) = system.as_str() {
        return Ok(s.to_string());
    }
    if let Some(arr) = system.as_array() {
        return concat_text_blocks(arr);
    }
    Err(TranslationError::invalid(
        "system must be a string or array of text blocks",
    ))
}

fn concat_text_blocks(blocks: &[Value]) -> Result<String, TranslationError> {
    let mut out = String::new();
    for block in blocks {
        let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if block_type != "text" {
            return Err(TranslationError::invalid(
                "non-text content blocks are not supported",
            ));
        }
        if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
            out.push_str(t);
        }
    }
    Ok(out)
}

/// Normalize a message content field to a plain string.
fn normalize_content(content: Option<&Value>) -> Result<String, TranslationError> {
    match content {
        None => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(blocks)) => concat_text_blocks(blocks),
        Some(_) => Err(TranslationError::invalid("invalid content type")),
    }
}

// ── Persistence helper ──────────────────────────────────────────

/// Extract the last user message content from an Anthropic request for persistence.
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

// ── Response translation: OpenAI → Anthropic (non-streaming) ───

/// Map an OpenAI finish_reason to an Anthropic stop_reason.
fn map_finish_reason(reason: Option<&str>) -> &'static str {
    match reason {
        Some("length") => "max_tokens",
        _ => "end_turn",
    }
}

/// Translate an OpenAI chat/completions response body to an Anthropic message response.
pub(crate) fn openai_to_anthropic_response(resp: Value, model: &str) -> Value {
    let id = resp
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| {
            if s.starts_with("msg_") {
                s.to_string()
            } else {
                format!("msg_{s}")
            }
        })
        .unwrap_or_else(|| format!("msg_{}", uuid::Uuid::new_v4()));

    let text = resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let finish_reason = resp["choices"][0]["finish_reason"].as_str();
    let stop_reason = map_finish_reason(finish_reason);

    let input_tokens = resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    let output_tokens = resp["usage"]["completion_tokens"].as_u64().unwrap_or(0);

    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    })
}

// ── SSE translation: OpenAI → Anthropic (streaming) ────────────

pub(crate) struct SseTranslationState {
    pub msg_id: String,
    pub model: String,
    pub header_emitted: bool,
    pub output_tokens: u64,
    pub stop_reason: String,
}

impl SseTranslationState {
    pub(crate) fn new(model: String) -> Self {
        Self {
            msg_id: format!("msg_{}", uuid::Uuid::new_v4()),
            model,
            header_emitted: false,
            output_tokens: 0,
            stop_reason: "end_turn".to_string(),
        }
    }
}

/// Format one Anthropic SSE event as bytes.
pub(crate) fn anthropic_sse_event(event_name: &str, data: &Value) -> Bytes {
    let s = format!(
        "event: {}\ndata: {}\n\n",
        event_name,
        serde_json::to_string(data).unwrap_or_default()
    );
    Bytes::from(s)
}

/// Translate one complete SSE line from the OpenAI upstream into zero or more
/// Anthropic SSE event bytes. The caller must split incoming bytes into lines
/// and call this for each line (blank lines are ignored).
pub(crate) fn translate_openai_sse_chunk(
    line: &str,
    state: &mut SseTranslationState,
) -> Vec<Bytes> {
    let mut out = Vec::new();

    let data = match line.strip_prefix("data: ") {
        Some(d) => d.trim(),
        None => return out,
    };

    // Terminal event — emit the closing Anthropic sequence.
    if data == "[DONE]" {
        if !state.header_emitted {
            out.extend(emit_header(state));
        }
        out.push(anthropic_sse_event(
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": 0}),
        ));
        let stop_reason = state.stop_reason.clone();
        out.push(anthropic_sse_event(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                "usage": {"output_tokens": state.output_tokens}
            }),
        ));
        out.push(anthropic_sse_event(
            "message_stop",
            &json!({"type": "message_stop"}),
        ));
        return out;
    }

    let chunk: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return out,
    };

    // Track finish_reason and usage from the upstream chunk.
    if let Some(r) = chunk["choices"][0]["finish_reason"].as_str() {
        state.stop_reason = map_finish_reason(Some(r)).to_string();
    }
    if let Some(n) = chunk["usage"]["completion_tokens"].as_u64() {
        state.output_tokens = n;
    }

    let delta_text = chunk["choices"][0]["delta"]["content"]
        .as_str()
        .unwrap_or("");

    // Emit header events on the first chunk.
    if !state.header_emitted {
        out.extend(emit_header(state));
    }

    if !delta_text.is_empty() {
        out.push(anthropic_sse_event(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": delta_text}
            }),
        ));
    }

    out
}

fn emit_header(state: &mut SseTranslationState) -> Vec<Bytes> {
    state.header_emitted = true;
    vec![
        anthropic_sse_event(
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": state.msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": state.model,
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
            }),
        ),
        anthropic_sse_event(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            }),
        ),
    ]
}

// ── Unit tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn translate(req: Value) -> Value {
        anthropic_to_openai(req).expect("translation should succeed")
    }

    #[test]
    fn system_string_is_prepended_as_system_message() {
        let req = json!({
            "model": "m", "max_tokens": 10,
            "system": "be helpful",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let oai = translate(req);
        let msgs = oai["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "be helpful");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn system_array_blocks_are_concatenated() {
        let req = json!({
            "model": "m", "max_tokens": 10,
            "system": [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}],
            "messages": [{"role": "user", "content": "hi"}]
        });
        let oai = translate(req);
        assert_eq!(oai["messages"][0]["content"], "foobar");
    }

    #[test]
    fn content_array_text_blocks_concatenated() {
        let req = json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "world"}
            ]}]
        });
        let oai = translate(req);
        assert_eq!(oai["messages"][0]["content"], "hello world");
    }

    #[test]
    fn non_text_content_block_returns_400_error() {
        let req = json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {}}
            ]}]
        });
        let err = anthropic_to_openai(req).unwrap_err();
        assert_eq!(err.http_status, 400);
        assert_eq!(err.error_type, "invalid_request_error");
    }

    #[test]
    fn stop_sequences_mapped_to_stop() {
        let req = json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "stop_sequences": ["END", "STOP"]
        });
        let oai = translate(req);
        assert_eq!(oai["stop"], json!(["END", "STOP"]));
        assert!(oai.get("stop_sequences").is_none());
    }

    #[test]
    fn top_k_and_metadata_dropped() {
        let req = json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "top_k": 40,
            "metadata": {"user_id": "u1"}
        });
        let oai = translate(req);
        assert!(oai.get("top_k").is_none());
        assert!(oai.get("metadata").is_none());
    }

    #[test]
    fn missing_model_returns_error() {
        let req = json!({"max_tokens": 10, "messages": []});
        assert!(anthropic_to_openai(req).is_err());
    }

    #[test]
    fn missing_max_tokens_returns_error() {
        let req = json!({"model": "m", "messages": []});
        assert!(anthropic_to_openai(req).is_err());
    }

    #[test]
    fn finish_reason_stop_maps_to_end_turn() {
        let resp = json!({"id":"x","choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}});
        let out = openai_to_anthropic_response(resp, "m");
        assert_eq!(out["stop_reason"], "end_turn");
    }

    #[test]
    fn finish_reason_length_maps_to_max_tokens() {
        let resp = json!({"id":"x","choices":[{"message":{"content":"hi"},"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":2}});
        let out = openai_to_anthropic_response(resp, "m");
        assert_eq!(out["stop_reason"], "max_tokens");
    }

    #[test]
    fn finish_reason_null_maps_to_end_turn() {
        let resp = json!({"id":"x","choices":[{"message":{"content":"hi"},"finish_reason":null}],"usage":{"prompt_tokens":1,"completion_tokens":2}});
        let out = openai_to_anthropic_response(resp, "m");
        assert_eq!(out["stop_reason"], "end_turn");
    }

    #[test]
    fn prompt_tokens_mapped_to_input_tokens() {
        let resp = json!({"id":"x","choices":[{"message":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3}});
        let out = openai_to_anthropic_response(resp, "m");
        assert_eq!(out["usage"]["input_tokens"], 5);
        assert_eq!(out["usage"]["output_tokens"], 3);
    }

    #[test]
    fn id_prefixed_with_msg_when_missing_prefix() {
        let resp = json!({"id":"abc123","choices":[{"message":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0}});
        let out = openai_to_anthropic_response(resp, "m");
        assert!(out["id"].as_str().unwrap().starts_with("msg_"));
    }

    #[test]
    fn id_not_double_prefixed_when_already_msg_() {
        let resp = json!({"id":"msg_abc","choices":[{"message":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0}});
        let out = openai_to_anthropic_response(resp, "m");
        assert_eq!(out["id"], "msg_abc");
    }

    #[test]
    fn sse_translate_emits_header_on_first_chunk() {
        let mut state = SseTranslationState::new("m".to_string());
        let line = r#"data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#;
        let events = translate_openai_sse_chunk(line, &mut state);
        // message_start + content_block_start + content_block_delta
        assert!(events.len() >= 3);
        assert!(state.header_emitted);
    }

    #[test]
    fn sse_translate_done_emits_closing_sequence() {
        let mut state = SseTranslationState::new("m".to_string());
        translate_openai_sse_chunk(
            r#"data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
            &mut state,
        );
        let events = translate_openai_sse_chunk("data: [DONE]", &mut state);
        let all = events
            .iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect::<String>();
        assert!(all.contains("content_block_stop"));
        assert!(all.contains("message_delta"));
        assert!(all.contains("message_stop"));
    }

    #[test]
    fn sse_translate_finish_reason_length_produces_max_tokens() {
        let mut state = SseTranslationState::new("m".to_string());
        translate_openai_sse_chunk(
            r#"data: {"choices":[{"delta":{"content":"x"},"finish_reason":"length"}]}"#,
            &mut state,
        );
        assert_eq!(state.stop_reason, "max_tokens");
    }

    #[test]
    fn sse_translate_non_data_lines_ignored() {
        let mut state = SseTranslationState::new("m".to_string());
        let events = translate_openai_sse_chunk(": keep-alive", &mut state);
        assert!(events.is_empty());
    }
}
