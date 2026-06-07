use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::handlers::{authenticate, check_rate_limit};
use crate::state::AppState;

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
