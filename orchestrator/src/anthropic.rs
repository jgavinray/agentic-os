use bytes::Bytes;
use serde_json::{json, Value};
use std::collections::HashMap;

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
pub(crate) fn anthropic_to_openai(req: Value) -> Result<Value, TranslationError> {
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

    let mut oai_messages: Vec<Value> = Vec::new();

    if let Some(system) = req.get("system") {
        let system_text = extract_system_text(system)?;
        oai_messages.push(json!({"role": "system", "content": system_text}));
    }

    for msg in &messages {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        oai_messages.extend(translate_message(role, msg.get("content"))?);
    }

    let mut oai = json!({
        "model": req["model"],
        "messages": oai_messages,
        "max_tokens": req["max_tokens"],
    });

    for field in ["temperature", "top_p", "stream"] {
        if let Some(v) = req.get(field) {
            oai[field] = v.clone();
        }
    }

    if let Some(v) = req.get("stop_sequences") {
        oai["stop"] = v.clone();
    }

    // Translate tools: input_schema → parameters
    if let Some(tools) = req.get("tools").and_then(|v| v.as_array()) {
        let oai_tools: Vec<Value> = tools.iter().map(translate_tool_def).collect();
        oai["tools"] = json!(oai_tools);
    }

    if let Some(tc) = req.get("tool_choice") {
        oai["tool_choice"] = translate_tool_choice(tc);
    }

    // top_k and metadata are silently dropped.
    Ok(oai)
}

/// Translate one Anthropic message into one or more OpenAI messages.
/// Returns multiple messages when tool_result blocks are present.
fn translate_message(role: &str, content: Option<&Value>) -> Result<Vec<Value>, TranslationError> {
    match content {
        None => Ok(vec![json!({"role": role, "content": ""})]),
        Some(Value::String(s)) => Ok(vec![json!({"role": role, "content": s})]),
        Some(Value::Array(blocks)) => {
            if role == "assistant" {
                translate_assistant_content(blocks)
            } else {
                translate_user_content(blocks)
            }
        }
        Some(_) => Err(TranslationError::invalid("invalid content type")),
    }
}

/// Assistant messages may contain text + tool_use blocks → one OAI message with tool_calls.
fn translate_assistant_content(blocks: &[Value]) -> Result<Vec<Value>, TranslationError> {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    for block in blocks {
        match block.get("type").and_then(|v| v.as_str()).unwrap_or("") {
            "text" => {
                if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                    text_parts.push(t.to_string());
                }
            }
            "tool_use" => {
                let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let input = block.get("input").cloned().unwrap_or(json!({}));
                let arguments = serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string());
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments}
                }));
            }
            other => {
                return Err(TranslationError::invalid(format!(
                    "unsupported block type in assistant message: {other}"
                )))
            }
        }
    }

    let mut msg = json!({"role": "assistant", "content": text_parts.join("")});
    if !tool_calls.is_empty() {
        msg["tool_calls"] = json!(tool_calls);
    }
    Ok(vec![msg])
}

/// User messages may contain text + tool_result blocks.
/// Each tool_result becomes a separate {role:"tool"} message.
fn translate_user_content(blocks: &[Value]) -> Result<Vec<Value>, TranslationError> {
    let mut out: Vec<Value> = Vec::new();
    let mut text_parts: Vec<String> = Vec::new();

    for block in blocks {
        match block.get("type").and_then(|v| v.as_str()).unwrap_or("") {
            "text" => {
                if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                    text_parts.push(t.to_string());
                }
            }
            "tool_result" => {
                if !text_parts.is_empty() {
                    out.push(json!({"role": "user", "content": text_parts.join("")}));
                    text_parts.clear();
                }
                let tool_use_id = block.get("tool_use_id").and_then(|v| v.as_str()).unwrap_or("");
                let content = match block.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(bs)) => bs
                        .iter()
                        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => String::new(),
                };
                out.push(json!({"role": "tool", "tool_call_id": tool_use_id, "content": content}));
            }
            "image" => {
                return Err(TranslationError::invalid(
                    "image content blocks are not supported",
                ))
            }
            other => {
                return Err(TranslationError::invalid(format!(
                    "unsupported content block type: {other}"
                )))
            }
        }
    }

    if !text_parts.is_empty() {
        out.push(json!({"role": "user", "content": text_parts.join("")}));
    }
    if out.is_empty() {
        out.push(json!({"role": "user", "content": ""}));
    }
    Ok(out)
}

fn translate_tool_def(tool: &Value) -> Value {
    let name = tool.get("name").cloned().unwrap_or(json!(""));
    let params = tool
        .get("input_schema")
        .cloned()
        .unwrap_or(json!({"type": "object", "properties": {}}));
    let mut function = json!({"name": name, "parameters": params});
    if let Some(desc) = tool.get("description") {
        function["description"] = desc.clone();
    }
    json!({"type": "function", "function": function})
}

fn translate_tool_choice(tc: &Value) -> Value {
    match tc.get("type").and_then(|v| v.as_str()) {
        Some("auto") => json!("auto"),
        Some("any") => json!("required"),
        Some("tool") => {
            let name = tc.get("name").and_then(|v| v.as_str()).unwrap_or("");
            json!({"type": "function", "function": {"name": name}})
        }
        _ => json!("auto"),
    }
}

fn extract_system_text(system: &Value) -> Result<String, TranslationError> {
    if let Some(s) = system.as_str() {
        return Ok(s.to_string());
    }
    if let Some(arr) = system.as_array() {
        let mut out = String::new();
        for block in arr {
            if block.get("type").and_then(|v| v.as_str()) != Some("text") {
                return Err(TranslationError::invalid(
                    "system must be a string or array of text blocks",
                ));
            }
            if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                out.push_str(t);
            }
        }
        return Ok(out);
    }
    Err(TranslationError::invalid(
        "system must be a string or array of text blocks",
    ))
}

// ── Persistence helper ──────────────────────────────────────────

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

// ── Response translation: OpenAI → Anthropic (non-streaming) ───

fn map_finish_reason(reason: Option<&str>) -> &'static str {
    match reason {
        Some("length") => "max_tokens",
        Some("tool_calls") => "tool_use",
        _ => "end_turn",
    }
}

/// Translate an OpenAI chat/completions response into an Anthropic message response.
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

    let finish_reason = resp["choices"][0]["finish_reason"].as_str();
    let stop_reason = map_finish_reason(finish_reason);
    let input_tokens = resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    let output_tokens = resp["usage"]["completion_tokens"].as_u64().unwrap_or(0);

    let mut content: Vec<Value> = Vec::new();

    // Text content block (only if non-empty).
    let text = match &resp["choices"][0]["message"]["content"] {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Array(blocks) => {
            let joined = blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<String>();
            if joined.is_empty() { None } else { Some(joined) }
        }
        _ => None,
    };
    if let Some(t) = text {
        content.push(json!({"type": "text", "text": t}));
    }

    // Tool use content blocks.
    if let Some(tool_calls) = resp["choices"][0]["message"]["tool_calls"].as_array() {
        for tc in tool_calls {
            let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let name = tc["function"].get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = tc["function"].get("arguments").and_then(|v| v.as_str()).unwrap_or("{}");
            let input: Value = serde_json::from_str(args).unwrap_or(json!({}));
            content.push(json!({"type": "tool_use", "id": id, "name": name, "input": input}));
        }
    }

    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }

    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
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
    // Maps OpenAI tool_call index → Anthropic content block index.
    tool_block_map: HashMap<u64, u32>,
    // Next available Anthropic content block index.
    next_block_index: u32,
    // Whether the text block has been opened.
    text_block_open: bool,
}

impl SseTranslationState {
    pub(crate) fn new(model: String) -> Self {
        Self {
            msg_id: format!("msg_{}", uuid::Uuid::new_v4()),
            model,
            header_emitted: false,
            output_tokens: 0,
            stop_reason: "end_turn".to_string(),
            tool_block_map: HashMap::new(),
            next_block_index: 0,
            text_block_open: false,
        }
    }
}

/// Format one Anthropic SSE event as bytes.
pub(crate) fn anthropic_sse_event(event_name: &str, data: &Value) -> Bytes {
    Bytes::from(format!(
        "event: {}\ndata: {}\n\n",
        event_name,
        serde_json::to_string(data).unwrap_or_default()
    ))
}

/// Translate one complete SSE line from the OpenAI upstream into Anthropic SSE bytes.
pub(crate) fn translate_openai_sse_chunk(
    line: &str,
    state: &mut SseTranslationState,
) -> Vec<Bytes> {
    let mut out = Vec::new();

    let data = match line.strip_prefix("data: ") {
        Some(d) => d.trim(),
        None => return out,
    };

    if data == "[DONE]" {
        if !state.header_emitted {
            out.push(emit_message_start(state));
        }
        // If no blocks were opened at all, open and close a text block.
        if state.next_block_index == 0 {
            out.push(anthropic_sse_event(
                "content_block_start",
                &json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
            ));
            state.next_block_index = 1;
        }
        // Close all open blocks in order.
        for i in 0..state.next_block_index {
            out.push(anthropic_sse_event(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": i}),
            ));
        }
        let stop_reason = state.stop_reason.clone();
        out.push(anthropic_sse_event(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                "usage": {"output_tokens": state.output_tokens}
            }),
        ));
        out.push(anthropic_sse_event("message_stop", &json!({"type": "message_stop"})));
        return out;
    }

    let chunk: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return out,
    };

    if let Some(r) = chunk["choices"][0]["finish_reason"].as_str() {
        state.stop_reason = map_finish_reason(Some(r)).to_string();
    }
    if let Some(n) = chunk["usage"]["completion_tokens"].as_u64() {
        state.output_tokens = n;
    }

    if !state.header_emitted {
        out.push(emit_message_start(state));
    }

    let delta = &chunk["choices"][0]["delta"];

    // Text delta.
    let text = delta.get("content").and_then(|v| v.as_str()).unwrap_or("");
    if !text.is_empty() {
        if !state.text_block_open {
            let idx = state.next_block_index;
            state.next_block_index += 1;
            state.text_block_open = true;
            out.push(anthropic_sse_event(
                "content_block_start",
                &json!({"type":"content_block_start","index":idx,"content_block":{"type":"text","text":""}}),
            ));
        }
        // Text block index is 0 when it's the first opened.
        let text_idx = if state.text_block_open && state.tool_block_map.is_empty() {
            0u32
        } else {
            // Find the index we assigned to the text block — it's always the first opened.
            0u32
        };
        out.push(anthropic_sse_event(
            "content_block_delta",
            &json!({"type":"content_block_delta","index":text_idx,"delta":{"type":"text_delta","text":text}}),
        ));
    }

    // Tool call deltas.
    if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tool_calls {
            let oai_idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);

            if !state.tool_block_map.contains_key(&oai_idx) {
                let block_idx = state.next_block_index;
                state.next_block_index += 1;
                state.tool_block_map.insert(oai_idx, block_idx);

                let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let name = tc["function"].get("name").and_then(|v| v.as_str()).unwrap_or("");
                out.push(anthropic_sse_event(
                    "content_block_start",
                    &json!({
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}
                    }),
                ));
            }

            let block_idx = state.tool_block_map[&oai_idx];
            let args = tc["function"].get("arguments").and_then(|v| v.as_str()).unwrap_or("");
            if !args.is_empty() {
                out.push(anthropic_sse_event(
                    "content_block_delta",
                    &json!({
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "input_json_delta", "partial_json": args}
                    }),
                ));
            }
        }
    }

    out
}

fn emit_message_start(state: &mut SseTranslationState) -> Bytes {
    state.header_emitted = true;
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
    )
}

// ── Unit tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn translate(req: Value) -> Value {
        anthropic_to_openai(req).expect("translation should succeed")
    }

    // ── anthropic_to_openai: basics ──────────────────────────────

    #[test]
    fn system_string_is_prepended_as_system_message() {
        let oai = translate(json!({
            "model": "m", "max_tokens": 10,
            "system": "be helpful",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let msgs = oai["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "be helpful");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn system_array_blocks_are_concatenated() {
        let oai = translate(json!({
            "model": "m", "max_tokens": 10,
            "system": [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}],
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(oai["messages"][0]["content"], "foobar");
    }

    #[test]
    fn content_array_text_blocks_concatenated() {
        let oai = translate(json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "world"}
            ]}]
        }));
        assert_eq!(oai["messages"][0]["content"], "hello world");
    }

    #[test]
    fn image_content_block_returns_400_error() {
        let err = anthropic_to_openai(json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]
        }))
        .unwrap_err();
        assert_eq!(err.http_status, 400);
        assert_eq!(err.error_type, "invalid_request_error");
    }

    #[test]
    fn stop_sequences_mapped_to_stop() {
        let oai = translate(json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "stop_sequences": ["END", "STOP"]
        }));
        assert_eq!(oai["stop"], json!(["END", "STOP"]));
        assert!(oai.get("stop_sequences").is_none());
    }

    #[test]
    fn top_k_and_metadata_dropped() {
        let oai = translate(json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "top_k": 40, "metadata": {"user_id": "u1"}
        }));
        assert!(oai.get("top_k").is_none());
        assert!(oai.get("metadata").is_none());
    }

    #[test]
    fn missing_model_returns_error() {
        assert!(anthropic_to_openai(json!({"max_tokens": 10, "messages": []})).is_err());
    }

    #[test]
    fn missing_max_tokens_returns_error() {
        assert!(anthropic_to_openai(json!({"model": "m", "messages": []})).is_err());
    }

    // ── Tool definitions ─────────────────────────────────────────

    #[test]
    fn tools_input_schema_translated_to_parameters() {
        let oai = translate(json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "name": "bash",
                "description": "Run a shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]
                }
            }]
        }));
        let tool = &oai["tools"][0];
        assert_eq!(tool["type"], "function");
        assert_eq!(tool["function"]["name"], "bash");
        assert_eq!(tool["function"]["description"], "Run a shell command");
        assert!(tool["function"].get("input_schema").is_none());
        assert_eq!(tool["function"]["parameters"]["properties"]["command"]["type"], "string");
    }

    #[test]
    fn tool_choice_auto_passes_through() {
        let oai = translate(json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "auto"}
        }));
        assert_eq!(oai["tool_choice"], "auto");
    }

    #[test]
    fn tool_choice_any_becomes_required() {
        let oai = translate(json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "any"}
        }));
        assert_eq!(oai["tool_choice"], "required");
    }

    #[test]
    fn tool_choice_specific_tool() {
        let oai = translate(json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "tool", "name": "bash"}
        }));
        assert_eq!(oai["tool_choice"]["type"], "function");
        assert_eq!(oai["tool_choice"]["function"]["name"], "bash");
    }

    // ── Tool use in messages ─────────────────────────────────────

    #[test]
    fn assistant_tool_use_block_becomes_tool_calls() {
        let oai = translate(json!({
            "model": "m", "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "run ls"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_1", "name": "bash", "input": {"command": "ls"}}
                ]}
            ]
        }));
        let msgs = oai["messages"].as_array().unwrap();
        let asst = &msgs[1];
        assert_eq!(asst["role"], "assistant");
        assert_eq!(asst["tool_calls"][0]["id"], "call_1");
        assert_eq!(asst["tool_calls"][0]["function"]["name"], "bash");
        let args: Value = serde_json::from_str(
            asst["tool_calls"][0]["function"]["arguments"].as_str().unwrap()
        ).unwrap();
        assert_eq!(args["command"], "ls");
    }

    #[test]
    fn assistant_mixed_text_and_tool_use() {
        let oai = translate(json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "assistant", "content": [
                {"type": "text", "text": "Let me check"},
                {"type": "tool_use", "id": "c1", "name": "bash", "input": {"command": "pwd"}}
            ]}]
        }));
        let asst = &oai["messages"][0];
        assert_eq!(asst["content"], "Let me check");
        assert_eq!(asst["tool_calls"][0]["id"], "c1");
    }

    #[test]
    fn user_tool_result_becomes_tool_role_message() {
        let oai = translate(json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "/home/user"}
            ]}]
        }));
        let msgs = oai["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["tool_call_id"], "call_1");
        assert_eq!(msgs[0]["content"], "/home/user");
    }

    #[test]
    fn user_mixed_tool_result_and_text_produces_multiple_messages() {
        let oai = translate(json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "output"},
                {"type": "text", "text": "looks good"}
            ]}]
        }));
        let msgs = oai["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "looks good");
    }

    // ── openai_to_anthropic_response ─────────────────────────────

    #[test]
    fn finish_reason_stop_maps_to_end_turn() {
        let out = openai_to_anthropic_response(
            json!({"id":"x","choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}),
            "m",
        );
        assert_eq!(out["stop_reason"], "end_turn");
    }

    #[test]
    fn finish_reason_length_maps_to_max_tokens() {
        let out = openai_to_anthropic_response(
            json!({"id":"x","choices":[{"message":{"content":"hi"},"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}),
            "m",
        );
        assert_eq!(out["stop_reason"], "max_tokens");
    }

    #[test]
    fn finish_reason_null_maps_to_end_turn() {
        let out = openai_to_anthropic_response(
            json!({"id":"x","choices":[{"message":{"content":"hi"},"finish_reason":null}],"usage":{"prompt_tokens":1,"completion_tokens":2}}),
            "m",
        );
        assert_eq!(out["stop_reason"], "end_turn");
    }

    #[test]
    fn finish_reason_tool_calls_maps_to_tool_use() {
        let out = openai_to_anthropic_response(
            json!({"id":"x","choices":[{"message":{"content":null,"tool_calls":[
                {"id":"c1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}
            ]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":3}}),
            "m",
        );
        assert_eq!(out["stop_reason"], "tool_use");
        assert_eq!(out["content"][0]["type"], "tool_use");
        assert_eq!(out["content"][0]["name"], "bash");
        assert_eq!(out["content"][0]["input"]["command"], "ls");
    }

    #[test]
    fn prompt_tokens_mapped_to_input_tokens() {
        let out = openai_to_anthropic_response(
            json!({"id":"x","choices":[{"message":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3}}),
            "m",
        );
        assert_eq!(out["usage"]["input_tokens"], 5);
        assert_eq!(out["usage"]["output_tokens"], 3);
    }

    #[test]
    fn id_prefixed_with_msg_when_missing_prefix() {
        let out = openai_to_anthropic_response(
            json!({"id":"abc","choices":[{"message":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0}}),
            "m",
        );
        assert!(out["id"].as_str().unwrap().starts_with("msg_"));
    }

    #[test]
    fn id_not_double_prefixed_when_already_msg_() {
        let out = openai_to_anthropic_response(
            json!({"id":"msg_abc","choices":[{"message":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0}}),
            "m",
        );
        assert_eq!(out["id"], "msg_abc");
    }

    // ── SSE translation ──────────────────────────────────────────

    #[test]
    fn sse_translate_emits_message_start_on_first_chunk() {
        let mut state = SseTranslationState::new("m".to_string());
        let events = translate_openai_sse_chunk(
            r#"data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
            &mut state,
        );
        let all = concat_events(&events);
        assert!(all.contains("message_start"));
        assert!(state.header_emitted);
    }

    #[test]
    fn sse_translate_text_opens_text_block_and_emits_delta() {
        let mut state = SseTranslationState::new("m".to_string());
        let events = translate_openai_sse_chunk(
            r#"data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}"#,
            &mut state,
        );
        let all = concat_events(&events);
        assert!(all.contains("content_block_start"));
        assert!(all.contains("text_delta"));
        assert!(all.contains("hello"));
    }

    #[test]
    fn sse_translate_done_emits_closing_sequence() {
        let mut state = SseTranslationState::new("m".to_string());
        translate_openai_sse_chunk(
            r#"data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
            &mut state,
        );
        let events = translate_openai_sse_chunk("data: [DONE]", &mut state);
        let all = concat_events(&events);
        assert!(all.contains("content_block_stop"));
        assert!(all.contains("message_delta"));
        assert!(all.contains("message_stop"));
    }

    #[test]
    fn sse_translate_tool_call_opens_tool_use_block() {
        let mut state = SseTranslationState::new("m".to_string());
        let events = translate_openai_sse_chunk(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}"#,
            &mut state,
        );
        let all = concat_events(&events);
        assert!(all.contains("content_block_start"));
        assert!(all.contains("tool_use"));
        assert!(all.contains("bash"));
    }

    #[test]
    fn sse_translate_tool_call_arguments_emit_input_json_delta() {
        let mut state = SseTranslationState::new("m".to_string());
        // Open the tool call block first.
        translate_openai_sse_chunk(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}"#,
            &mut state,
        );
        let events = translate_openai_sse_chunk(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"cmd\":\"ls\"}"}}]},"finish_reason":null}]}"#,
            &mut state,
        );
        let all = concat_events(&events);
        assert!(all.contains("input_json_delta"));
        assert!(all.contains("cmd"));
    }

    #[test]
    fn sse_translate_finish_reason_tool_calls_produces_tool_use() {
        let mut state = SseTranslationState::new("m".to_string());
        translate_openai_sse_chunk(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"bash","arguments":""}}]},"finish_reason":"tool_calls"}]}"#,
            &mut state,
        );
        assert_eq!(state.stop_reason, "tool_use");
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
        assert!(translate_openai_sse_chunk(": keep-alive", &mut state).is_empty());
    }

    fn concat_events(events: &[Bytes]) -> String {
        events
            .iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect()
    }
}
