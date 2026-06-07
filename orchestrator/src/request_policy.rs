use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::Value;

use crate::state::AppState;
use crate::telemetry;

pub(crate) fn maybe_openai_live_policy_response(
    state: &AppState,
    repo: &str,
    session_id: &str,
    user_content: &str,
) -> Option<Response> {
    let classification = crate::request_classification::classify_request_text(
        repo,
        session_id,
        user_content,
        None,
        "user_message",
    );
    live_policy_decision_response(state, &classification).map(|decision| {
        let (status, body) = live_policy_openai_body(&decision, user_content);
        (status, axum::Json(body)).into_response()
    })
}

pub(crate) fn maybe_anthropic_live_policy_response(
    state: &AppState,
    repo: &str,
    session_id: &str,
    user_content: &str,
) -> Option<Response> {
    let classification = crate::request_classification::classify_request_text(
        repo,
        session_id,
        user_content,
        None,
        "user_message",
    );
    live_policy_decision_response(state, &classification).map(|decision| {
        let (status, body) = live_policy_anthropic_body(&decision, user_content);
        (status, axum::Json(body)).into_response()
    })
}

fn live_policy_decision_response(
    state: &AppState,
    classification: &crate::request_classification::RequestClassification,
) -> Option<crate::request_classification::LivePolicyDecision> {
    if !state.request_live_policy_config.enabled {
        telemetry::record_request_live_policy_bypass("disabled");
        return None;
    }
    if state.request_live_policy_config.policy_version != "v1" {
        telemetry::record_request_live_policy_bypass("unsupported_policy_version");
        return None;
    }

    let decision = crate::request_classification::evaluate_live_policy(
        classification,
        &state.request_live_policy_config,
    );
    match &decision {
        Some(decision) => {
            telemetry::record_request_live_policy_action(decision.action, decision.reason);
        }
        None => telemetry::record_request_live_policy_bypass("shadow_only"),
    }
    decision
}

fn live_policy_openai_body(
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

fn live_policy_anthropic_body(
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
