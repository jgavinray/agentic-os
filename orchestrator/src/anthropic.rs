use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::Value;

pub(crate) use crate::anthropic_user_content::extract_user_content_from_anthropic;

pub(crate) fn error_value(error_type: &'static str, message: impl Into<String>) -> Value {
    serde_json::json!({
        "type": "error",
        "error": {"type": error_type, "message": message.into()}
    })
}

pub(crate) fn error(
    status: StatusCode,
    error_type: &'static str,
    message: impl Into<String>,
) -> Response {
    (status, axum::Json(error_value(error_type, message))).into_response()
}

pub(crate) fn normalize_response_content_types(req: &mut Value) {
    if let Some(messages) = req.get_mut("messages").and_then(|v| v.as_array_mut()) {
        for msg in messages.iter_mut() {
            if let Some(content) = msg.get_mut("content").and_then(|v| v.as_array_mut()) {
                for block in content.iter_mut() {
                    if block.get("type").and_then(|t| t.as_str()) == Some("output_text") {
                        block["type"] = Value::String("text".to_string());
                    }
                }
            }
        }
    }
}

pub(crate) fn sanitize_litellm_request(req: &mut Value) {
    if let Some(obj) = req.as_object_mut() {
        // LiteLLM's Anthropic adapter forwards extended-thinking fields to
        // OpenAI-compatible local backends as unsupported Responses API params
        // such as `max_output_tokens`, which causes vLLM to return 500s.
        obj.remove("thinking");
        obj.remove("max_output_tokens");
        obj.remove("reasoning_effort");
        obj.remove("context_management");
    }
}
