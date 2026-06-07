use axum::response::{IntoResponse, Response};

use crate::request_policy_responses::{live_policy_anthropic_body, live_policy_openai_body};
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
