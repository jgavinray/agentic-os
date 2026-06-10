use serde_json::Value;

const MAX_EXTRACTED_USER_CONTENT_CHARS: usize = 8_000;

/// Extract the latest real user text from an Anthropic request.
///
/// Claude Code can prepend large `<system-reminder>` text blocks inside the
/// user message before the actual prompt. Those reminders are context, not the
/// user's current task, and they can otherwise dominate request classification.
pub(crate) fn extract_user_content_from_anthropic(req: &Value) -> String {
    let Some(messages) = req.get("messages").and_then(|v| v.as_array()) else {
        return String::new();
    };

    messages
        .iter()
        .rev()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .filter_map(|m| m.get("content").map(extract_real_text_from_content))
        .find(|text| !text.trim().is_empty())
        .unwrap_or_default()
        .chars()
        .take(MAX_EXTRACTED_USER_CONTENT_CHARS)
        .collect()
}

fn extract_real_text_from_content(content: &Value) -> String {
    match content {
        Value::String(s) => strip_system_reminders(s).trim().to_string(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .map(strip_system_reminders)
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

pub(crate) fn strip_system_reminders(text: &str) -> String {
    let mut remaining = text;
    let mut out = String::new();

    while let Some(start) = remaining.find("<system-reminder>") {
        out.push_str(&remaining[..start]);
        let after_start = &remaining[start + "<system-reminder>".len()..];
        if let Some(end) = after_start.find("</system-reminder>") {
            remaining = &after_start[end + "</system-reminder>".len()..];
        } else {
            remaining = "";
        }
    }

    out.push_str(remaining);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_user_content_skips_claude_system_reminder_blocks() {
        let req = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<system-reminder>\n# claudeMd\nprovider and repo context\n</system-reminder>"
                    },
                    {
                        "type": "text",
                        "text": "Implement the retry policy fix in internal/retry/retry.go"
                    }
                ]
            }]
        });

        assert_eq!(
            extract_user_content_from_anthropic(&req),
            "Implement the retry policy fix in internal/retry/retry.go"
        );
    }

    #[test]
    fn claude_default_shape_classifies_current_task_not_repo_context() {
        let req = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<system-reminder>\n# claudeMd\nLLM provider configuration and prompt template context\n</system-reminder>"
                    },
                    {
                        "type": "text",
                        "text": "Implement the retry policy fix in internal/retry/retry.go"
                    }
                ]
            }]
        });

        let text = extract_user_content_from_anthropic(&req);
        let classification = crate::request_classification::classify_request_text(
            "omniscient",
            "session-a",
            &text,
            None,
            "user_message",
        );

        assert_eq!(classification.intent.as_str(), "implement");
    }

    #[test]
    fn extract_user_content_falls_back_to_prior_text_user_message_after_tool_result() {
        let req = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<system-reminder>large repo context</system-reminder>"
                        },
                        {
                            "type": "text",
                            "text": "Run the smoke test command."
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "Shell"}]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "content": "No such tool available: Shell",
                        "is_error": true
                    }]
                }
            ]
        });

        assert_eq!(
            extract_user_content_from_anthropic(&req),
            "Run the smoke test command."
        );
    }
}
