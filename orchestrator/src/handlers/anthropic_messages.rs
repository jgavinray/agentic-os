use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use serde_json::Value;
use std::sync::Arc;

use crate::anthropic;
use crate::auth::{authenticate, check_rate_limit};
use crate::handlers_anthropic_completion::handle_anthropic_non_streaming;
use crate::handlers_context::pack_context_into_anthropic_req;
use crate::handlers_litellm_attempt::prepare_litellm_attempt;
use crate::handlers_request::HandlerRequestScope;
use crate::handlers_request_preparation::prepare_anthropic_litellm_request;
use crate::handlers_streaming::handle_streaming_anthropic;
use crate::handlers_trajectory::{begin_and_persist_request, find_or_create_capture_session};
use crate::local_reasoning::local_reasoning_selection;
use crate::proxy_support::{baseline_arm_selection, litellm_route, merge_request_metadata};
use crate::request_policy::{
    classify_and_derive_request_policy, maybe_anthropic_live_policy_response,
};
use crate::state::AppState;
use crate::telemetry;

// ── Anthropic /v1/messages ──────────────────────────────────────

#[tracing::instrument(name = "handler.messages", skip(state, headers, body))]
pub async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let raw_client_body = body.to_vec();
    let mut capture =
        crate::client_capture::RawHttpCapture::new("messages", &headers, raw_client_body.clone());
    let payload: Value = match serde_json::from_slice(&raw_client_body) {
        Ok(payload) => payload,
        Err(e) => {
            let body = anthropic::error_value(
                "invalid_request_error",
                format!("invalid JSON request body: {e}"),
            );
            crate::client_capture::record_response_best_effort(
                state.capture_pool.as_ref(),
                capture,
                StatusCode::BAD_REQUEST,
                "application/json",
                crate::client_capture::to_json_bytes(&body),
            )
            .await;
            return anthropic::error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("invalid JSON request body: {e}"),
            );
        }
    };
    capture.parsed_request_body = Some(payload.clone());
    let Some((caller_token, namespace)) = authenticate(&state, &headers) else {
        let body = anthropic::error_value("authentication_error", "invalid or missing API key");
        crate::client_capture::record_response_best_effort(
            state.capture_pool.as_ref(),
            capture,
            StatusCode::UNAUTHORIZED,
            "application/json",
            crate::client_capture::to_json_bytes(&body),
        )
        .await;
        return anthropic::error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid or missing API key",
        );
    };
    if let Some(response) = check_rate_limit(&state, &caller_token) {
        return response;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let scope = HandlerRequestScope::from_headers(&headers, namespace, &state.default_task);
    let namespace = scope.namespace.clone();
    let repo = scope.repo.clone();
    let task = scope.task.clone();
    tracing::info!(repo = %repo, task = %task, endpoint = "messages", "routing request");
    scope.apply_to_capture(&mut capture);

    let user_content = anthropic::extract_user_content_from_anthropic(&payload);
    if let Some(response) =
        maybe_anthropic_live_policy_response(&state, &repo, &namespace, &user_content)
    {
        return response;
    }
    let baseline_arm = match baseline_arm_selection(&headers) {
        Ok(arm) => arm,
        Err(e) => {
            let body = anthropic::error_value(
                "invalid_request_error",
                format!("invalid baseline arm: {e}"),
            );
            return (StatusCode::BAD_REQUEST, axum::Json(body)).into_response();
        }
    };
    let reasoning_selection = local_reasoning_selection(&headers, &payload);
    let model = payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&state.default_model)
        .to_string();

    let route = litellm_route(&state, &namespace, Some(&model));
    // Stay in Anthropic format — no translation.
    let mut req =
        prepare_anthropic_litellm_request(payload, &route.routed_model, reasoning_selection);
    let session_id = find_or_create_capture_session(&state, &repo, &task).await;
    let (request_classification, request_policy) = classify_and_derive_request_policy(
        &repo,
        session_id.as_deref().unwrap_or("unknown"),
        &user_content,
        state.capture_pool.is_some(),
    );
    let tool_mediation_metadata = if state.tool_mediation_enabled {
        // Same policy-aware shaping as the OpenAI endpoint, but applied to the
        // Anthropic `tools` shape. The payload stays Anthropic-formatted all the
        // way through this handler.
        let outcome = crate::tool_mediation::shape_anthropic_request_with_policy(
            &mut req,
            &user_content,
            Some(&request_policy),
        );
        telemetry::record_tool_menu_outcome(&state.metrics, &outcome);
        Some(outcome.metadata())
    } else {
        None
    };
    let baseline_metadata = Some(serde_json::json!({
        "baseline_arm": baseline_arm.as_str(),
    }));
    let request_metadata = merge_request_metadata([tool_mediation_metadata, baseline_metadata]);
    let (trajectory, request_event_id) = begin_and_persist_request(
        &state,
        session_id.as_deref(),
        &repo,
        &user_content,
        request_metadata.clone(),
    )
    .await;
    let (context_pack_id, context_pack_hash) = pack_context_into_anthropic_req(
        &state,
        &mut req,
        session_id.as_deref(),
        &repo,
        &task,
        trajectory,
        request_event_id,
        Some(&request_classification),
        Some(&request_policy),
    )
    .await;
    let correlation_id = trajectory
        .map(|trajectory| trajectory.trajectory_id)
        .or_else(|| state.execution_feedback_enabled.then(uuid::Uuid::new_v4));
    let prepared_attempt = prepare_litellm_attempt(
        &state,
        &mut req,
        &mut capture,
        "messages",
        model.clone(),
        namespace.clone(),
        repo.clone(),
        task.clone(),
        request_event_id,
        trajectory,
        context_pack_id,
        &route,
        context_pack_hash.clone(),
        Some(baseline_arm.as_str().to_string()),
        reasoning_selection,
    )
    .await;

    if is_stream {
        return handle_streaming_anthropic(
            &state,
            req,
            user_content,
            repo,
            task,
            model,
            namespace,
            correlation_id,
            request_metadata,
            session_id,
            trajectory,
            request_event_id,
            context_pack_id,
            prepared_attempt.finalizer,
            prepared_attempt.vllm_cache_before,
            capture,
        )
        .await;
    }

    handle_anthropic_non_streaming(
        &state,
        req,
        repo,
        task,
        model,
        namespace,
        correlation_id,
        request_metadata,
        session_id,
        trajectory,
        request_event_id,
        context_pack_id,
        user_content,
        prepared_attempt.finalizer,
        prepared_attempt.vllm_cache_before,
        capture,
        route,
        context_pack_hash.clone(),
        Some(baseline_arm.as_str().to_string()),
    )
    .await
}
