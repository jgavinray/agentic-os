use axum::http::StatusCode;
use serde_json::Value;

pub(crate) fn live_policy_openai_body(
    decision: &crate::request_classification::LivePolicyDecision,
    user_content: &str,
) -> (StatusCode, Value) {
    match decision.action {
        "web_required" => (
            StatusCode::PRECONDITION_REQUIRED,
            serde_json::json!({
                "error": {
                    "type": "request_policy",
                    "code": "web_required",
                    "message": "This request requires current external information. Use an enabled web/tool path before answering."
                }
            }),
        ),
        "deterministic_template" => (
            StatusCode::OK,
            openai_completion(live_policy_template_text(user_content)),
        ),
        "ask_clarification" => (
            StatusCode::OK,
            openai_completion("Please provide the specific target or context before I continue."),
        ),
        _ => (
            StatusCode::FORBIDDEN,
            serde_json::json!({
                "error": {
                    "type": "request_policy",
                    "code": decision.action,
                    "message": "This request was stopped by the deterministic request safety policy."
                }
            }),
        ),
    }
}

pub(crate) fn live_policy_anthropic_body(
    decision: &crate::request_classification::LivePolicyDecision,
    user_content: &str,
) -> (StatusCode, Value) {
    match decision.action {
        "web_required" => (
            StatusCode::PRECONDITION_REQUIRED,
            serde_json::json!({
                "type": "error",
                "error": {
                    "type": "request_policy",
                    "message": "This request requires current external information. Use an enabled web/tool path before answering."
                }
            }),
        ),
        "deterministic_template" => (
            StatusCode::OK,
            anthropic_text_message(live_policy_template_text(user_content)),
        ),
        "ask_clarification" => (
            StatusCode::OK,
            anthropic_text_message(
                "Please provide the specific target or context before I continue.",
            ),
        ),
        _ => (
            StatusCode::FORBIDDEN,
            serde_json::json!({
                "type": "error",
                "error": {
                    "type": "request_policy",
                    "message": "This request was stopped by the deterministic request safety policy."
                }
            }),
        ),
    }
}

fn live_policy_template_text(user_content: &str) -> &'static str {
    let lower = user_content.trim().to_ascii_lowercase();
    if lower == "ping" {
        "pong"
    } else {
        "ok"
    }
}

fn openai_completion(text: &str) -> Value {
    serde_json::json!({
        "id": "chatcmpl-request-policy",
        "object": "chat.completion",
        "created": 0,
        "model": "request-policy",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    })
}

fn anthropic_text_message(text: &str) -> Value {
    serde_json::json!({
        "id": "msg_request_policy",
        "type": "message",
        "role": "assistant",
        "model": "request-policy",
        "content": [{
            "type": "text",
            "text": text
        }],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0
        }
    })
}
