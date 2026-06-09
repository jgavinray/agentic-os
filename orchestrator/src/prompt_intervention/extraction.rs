/// Deterministic prompt source extraction from captured request payloads.
///
/// This module extracts ordered text candidates from parsed request bodies and
/// raw HTTP bytes. It does not classify, store, or route extracted sources.
use serde_json::Value;

use crate::prompt_intervention_taxonomy::SourceKind;

/// Maximum characters copied from a single tool-result source.
pub const TOOL_RESULT_TEXT_MAX_CHARS: usize = 4_000;

/// A single extracted source candidate for prompt intervention detection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtractedPromptSource {
    /// Source kind used on the eventual intervention record.
    pub source_kind: SourceKind,
    /// Exact extracted text for this source candidate.
    pub text: String,
}

/// Extract prompt sources from a captured request body.
///
/// Structured extraction order is:
/// 1. latest user-role message
/// 2. assistant-role messages in request order
/// 3. tool-result messages or blocks in request order
/// 4. raw request body only when structured extraction yields no text
pub fn extract_prompt_sources(
    parsed_request_body: Option<&Value>,
    raw_request_body: &[u8],
) -> Vec<ExtractedPromptSource> {
    if let Some(messages) = parsed_request_body.and_then(messages_array) {
        let mut sources = Vec::new();
        if let Some(text) = latest_user_text(messages) {
            push_if_nonempty(&mut sources, SourceKind::UserMessage, text);
        }
        for message in messages {
            if role(message) == Some("assistant") {
                push_if_nonempty(
                    &mut sources,
                    SourceKind::AssistantMessage,
                    text_content_without_tool_results(message.get("content")),
                );
            }
        }
        for message in messages {
            if role(message) == Some("tool") {
                push_if_nonempty(
                    &mut sources,
                    SourceKind::ToolResult,
                    bounded_tool_text(message.get("content")),
                );
            }
            for text in tool_result_blocks(message.get("content")) {
                push_if_nonempty(&mut sources, SourceKind::ToolResult, text);
            }
        }
        if !sources.is_empty() {
            return sources;
        }
    }
    raw_prompt_fallback(raw_request_body)
}

fn messages_array(parsed: &Value) -> Option<&Vec<Value>> {
    parsed.get("messages").and_then(Value::as_array)
}

fn latest_user_text(messages: &[Value]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| role(message) == Some("user"))
        .map(|message| text_content_without_tool_results(message.get("content")))
}

fn role(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

fn text_content_without_tool_results(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter(|block| block_type(block) == Some("text"))
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn tool_result_blocks(content: Option<&Value>) -> Vec<String> {
    let Some(Value::Array(blocks)) = content else {
        return Vec::new();
    };
    blocks
        .iter()
        .filter(|block| block_type(block) == Some("tool_result"))
        .map(|block| bounded_tool_text(block.get("content")))
        .filter(|text| !text.is_empty())
        .collect()
}

fn block_type(block: &Value) -> Option<&str> {
    block.get("type").and_then(Value::as_str)
}

fn bounded_tool_text(content: Option<&Value>) -> String {
    let text = match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter(|block| block_type(block) == Some("text"))
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Value::Object(_)) => content
            .and_then(|value| value.get("text"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    };
    text.chars().take(TOOL_RESULT_TEXT_MAX_CHARS).collect()
}

fn push_if_nonempty(
    sources: &mut Vec<ExtractedPromptSource>,
    source_kind: SourceKind,
    text: String,
) {
    if !text.is_empty() {
        sources.push(ExtractedPromptSource { source_kind, text });
    }
}

fn raw_prompt_fallback(raw_request_body: &[u8]) -> Vec<ExtractedPromptSource> {
    let text = String::from_utf8_lossy(raw_request_body).to_string();
    if text.is_empty() {
        Vec::new()
    } else {
        vec![ExtractedPromptSource {
            source_kind: SourceKind::RawPrompt,
            text,
        }]
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn message(role: &str, content: Value) -> Value {
        json!({ "role": role, "content": content })
    }

    fn request(messages: Vec<Value>) -> Value {
        json!({ "messages": messages })
    }

    #[test]
    fn extracts_latest_user_not_older_user() {
        let body = request(vec![
            message("user", json!("older user")),
            message("assistant", json!("assistant text")),
            message("user", json!("latest user")),
        ]);
        let sources = extract_prompt_sources(Some(&body), b"");
        assert_eq!(sources[0].source_kind, SourceKind::UserMessage);
        assert_eq!(sources[0].text, "latest user");
        assert_eq!(sources[1].source_kind, SourceKind::AssistantMessage);
        assert_eq!(sources[1].text, "assistant text");
    }

    #[test]
    fn extracts_assistant_messages_in_request_order() {
        let body = request(vec![
            message("user", json!("user text")),
            message("assistant", json!("first assistant")),
            message("assistant", json!("second assistant")),
        ]);
        let sources = extract_prompt_sources(Some(&body), b"");
        assert_eq!(
            sources
                .iter()
                .filter(|source| source.source_kind == SourceKind::AssistantMessage)
                .map(|source| source.text.as_str())
                .collect::<Vec<_>>(),
            vec!["first assistant", "second assistant"]
        );
    }

    #[test]
    fn extracts_string_content_as_is() {
        let body = request(vec![message("user", json!("plain text"))]);
        let sources = extract_prompt_sources(Some(&body), b"");
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].text, "plain text");
    }

    #[test]
    fn extracts_array_text_blocks() {
        let body = request(vec![message(
            "user",
            json!([
                { "type": "text", "text": "first" },
                { "type": "text", "text": "second" }
            ]),
        )]);
        let sources = extract_prompt_sources(Some(&body), b"");
        assert_eq!(sources[0].text, "first\nsecond");
    }

    #[test]
    fn extracts_tool_result_blocks_separately_and_bounds_text() {
        let body = request(vec![message(
            "user",
            json!([
                { "type": "text", "text": "user text" },
                {
                    "type": "tool_result",
                    "content": [
                        { "type": "text", "text": "x".repeat(TOOL_RESULT_TEXT_MAX_CHARS + 10) }
                    ]
                }
            ]),
        )]);
        let sources = extract_prompt_sources(Some(&body), b"");
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0].source_kind, SourceKind::UserMessage);
        assert_eq!(sources[0].text, "user text");
        assert_eq!(sources[1].source_kind, SourceKind::ToolResult);
        assert_eq!(sources[1].text.chars().count(), TOOL_RESULT_TEXT_MAX_CHARS);
    }

    #[test]
    fn extracts_tool_role_message_as_tool_result() {
        let body = request(vec![message("tool", json!("tool output"))]);
        let sources = extract_prompt_sources(Some(&body), b"");
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].source_kind, SourceKind::ToolResult);
        assert_eq!(sources[0].text, "tool output");
    }

    #[test]
    fn ignores_non_text_image_blocks() {
        let body = request(vec![message(
            "user",
            json!([
                { "type": "image", "source": { "data": "base64" } },
                { "type": "text", "text": "visible text" }
            ]),
        )]);
        let sources = extract_prompt_sources(Some(&body), b"");
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].text, "visible text");
    }

    #[test]
    fn falls_back_to_raw_when_messages_missing_or_malformed() {
        let body = json!({ "not_messages": "invalid" });
        let sources = extract_prompt_sources(Some(&body), br#"{"raw":"body"}"#);
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].source_kind, SourceKind::RawPrompt);
        assert_eq!(sources[0].text, r#"{"raw":"body"}"#);

        let malformed = json!({ "messages": "not an array" });
        let sources = extract_prompt_sources(Some(&malformed), br#"{"raw":"again"}"#);
        assert_eq!(sources[0].source_kind, SourceKind::RawPrompt);
        assert_eq!(sources[0].text, r#"{"raw":"again"}"#);
    }

    #[test]
    fn falls_back_to_raw_when_structured_text_is_empty() {
        let body = request(vec![
            message("user", json!("")),
            message("assistant", json!([{ "type": "image", "source": {} }])),
        ]);
        let sources = extract_prompt_sources(Some(&body), br#"{"raw":"body"}"#);
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].source_kind, SourceKind::RawPrompt);
        assert_eq!(sources[0].text, r#"{"raw":"body"}"#);
    }

    #[test]
    fn raw_fallback_uses_utf8_lossy() {
        let raw = &[0xff, b'H', b'i'];
        let sources = extract_prompt_sources(None, raw);
        assert_eq!(sources[0].source_kind, SourceKind::RawPrompt);
        assert!(sources[0].text.contains("Hi"));
    }

    #[test]
    fn returns_no_empty_sources_when_raw_and_structured_are_empty() {
        let body = request(vec![message("user", json!(""))]);
        assert!(extract_prompt_sources(Some(&body), b"").is_empty());
    }
}
