use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::auth::{authenticate, check_auth, check_rate_limit};
use crate::state::{AppState, HarnessGuardrailRequest, HarnessGuardrailResponse};
use crate::telemetry;

#[tracing::instrument(name = "handler.harness_guardrail", skip(state, headers, req))]
pub async fn harness_guardrail(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<HarnessGuardrailRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let event_type = req.event_type.as_deref().unwrap_or("harness_event");
    let summary = req.summary.as_deref().unwrap_or("");
    let metadata = req.metadata.unwrap_or_else(|| serde_json::json!({}));
    let decision = crate::harness_feedback::evaluate_runtime_guardrail(
        event_type,
        summary,
        req.evidence.as_deref(),
        &metadata,
    );
    telemetry::record_harness_guardrail_decision(decision.action, decision.reason);

    axum::Json(HarnessGuardrailResponse {
        action: decision.action.to_string(),
        reason: decision.reason.to_string(),
        should_stop: decision.should_stop,
        signals: decision
            .signals
            .iter()
            .map(|signal| (*signal).to_string())
            .collect(),
    })
    .into_response()
}

#[tracing::instrument(name = "handler.harness_outcome", skip(state, headers, req), fields(trajectory_id = %req.trajectory_id))]
pub async fn harness_outcome(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<crate::adversarial_harness::HarnessOutcomeRequest>,
) -> Response {
    let Some((caller_token, _namespace)) = authenticate(&state, &headers) else {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    };
    if let Some(response) = check_rate_limit(&state, &caller_token) {
        return response;
    }

    match crate::adversarial_harness::record_harness_outcome(&state.pool, &req).await {
        Ok(outcome_event_id) => axum::Json(crate::adversarial_harness::HarnessOutcomeResponse {
            captured: true,
            outcome_event_id,
        })
        .into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({
                "error": "invalid_harness_outcome",
                "detail": e.to_string()
            })),
        )
            .into_response(),
    }
}

#[tracing::instrument(name = "handler.litellm_callback_payload", skip(state, headers, req), fields(attempt_id = ?req.attempt_id))]
pub async fn litellm_callback_payload(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<crate::adversarial_harness::LiteLlmCallbackPayloadRequest>,
) -> Response {
    let Some((caller_token, _namespace)) = authenticate(&state, &headers) else {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    };
    if let Some(response) = check_rate_limit(&state, &caller_token) {
        return response;
    }

    match crate::adversarial_harness::record_litellm_callback_payload(&state.pool, &req).await {
        Ok((callback_payload_id, normalized_ledger)) => {
            axum::Json(crate::adversarial_harness::LiteLlmCallbackPayloadResponse {
                captured: true,
                callback_payload_id,
                normalized_ledger,
            })
            .into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({
                "error": "invalid_litellm_callback_payload",
                "detail": e.to_string()
            })),
        )
            .into_response(),
    }
}
