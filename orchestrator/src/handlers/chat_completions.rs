use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use serde_json::Value;
use std::sync::Arc;

use crate::auth::{authenticate, check_rate_limit};
use crate::handlers_context::pack_context_into_req;
use crate::handlers_litellm_attempt::prepare_litellm_attempt;
use crate::handlers_openai_dispatch::handle_openai_non_streaming;
use crate::handlers_request::HandlerRequestScope;
use crate::handlers_request_preparation::prepare_openai_litellm_request;
use crate::handlers_streaming::handle_streaming;
use crate::handlers_trajectory::{begin_and_persist_request, find_or_create_capture_session};
use crate::local_reasoning::local_reasoning_selection;
use crate::proxy_support::{
    baseline_arm_selection, extract_user_content_openai, litellm_route, merge_request_metadata,
};
use crate::request_policy::{
    classify_and_derive_request_policy, maybe_openai_live_policy_response,
};
use crate::state::AppState;
use crate::telemetry;

#[tracing::instrument(name = "handler.chat_completions", skip(state, headers, body))]
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let raw_client_body = body.to_vec();
    let mut capture = crate::client_capture::RawHttpCapture::new(
        "chat/completions",
        &headers,
        raw_client_body.clone(),
    );
    let payload: Value = match serde_json::from_slice(&raw_client_body) {
        Ok(payload) => payload,
        Err(e) => {
            let body = serde_json::json!({
                "error": {
                    "type": "invalid_request_error",
                    "message": format!("invalid JSON request body: {e}")
                }
            });
            crate::client_capture::record_response_best_effort(
                state.capture_pool.as_ref(),
                capture,
                StatusCode::BAD_REQUEST,
                "application/json",
                crate::client_capture::to_json_bytes(&body),
            )
            .await;
            return (StatusCode::BAD_REQUEST, axum::Json(body)).into_response();
        }
    };
    capture.parsed_request_body = Some(payload.clone());
    let Some((caller_token, namespace)) = authenticate(&state, &headers) else {
        let body = serde_json::json!({"error": "unauthorized"});
        crate::client_capture::record_response_best_effort(
            state.capture_pool.as_ref(),
            capture,
            StatusCode::UNAUTHORIZED,
            "application/json",
            crate::client_capture::to_json_bytes(&body),
        )
        .await;
        return (StatusCode::UNAUTHORIZED, axum::Json(body)).into_response();
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
    tracing::info!(repo = %repo, task = %task, "routing request");
    scope.apply_to_capture(&mut capture);

    let requested_model = payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&state.default_model)
        .to_string();
    let user_content = extract_user_content_openai(&payload);
    if let Some(response) =
        maybe_openai_live_policy_response(&state, &repo, &namespace, &user_content)
    {
        return response;
    }
    let baseline_arm = match baseline_arm_selection(&headers) {
        Ok(arm) => arm,
        Err(e) => {
            let body = serde_json::json!({"error": "invalid_baseline_arm", "detail": e});
            return (StatusCode::BAD_REQUEST, axum::Json(body)).into_response();
        }
    };
    let reasoning_selection = local_reasoning_selection(&headers, &payload);
    let route = litellm_route(&state, &namespace, Some(&requested_model));
    let mut req =
        prepare_openai_litellm_request(&payload, &route.routed_model, reasoning_selection);
    let sampling_audit = crate::sampling::capture_and_maybe_override(
        &payload,
        &mut req,
        state.sampling_config,
        state.sampling_policy.as_ref(),
    );
    let sampling_metadata = sampling_audit.as_ref().map(|audit| {
        audit.metadata(
            "/v1/chat/completions",
            &requested_model,
            &route.routed_model,
        )
    });
    let session_id = find_or_create_capture_session(&state, &repo, &task).await;
    let (request_classification, request_policy) = classify_and_derive_request_policy(
        &repo,
        session_id.as_deref().unwrap_or("unknown"),
        &user_content,
        state.capture_pool.is_some(),
    );
    let tool_mediation_metadata = if state.tool_mediation_enabled {
        let outcome = crate::tool_mediation::shape_openai_request_with_policy(
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
    let request_metadata = merge_request_metadata([
        sampling_metadata,
        tool_mediation_metadata,
        baseline_metadata,
    ]);
    let (trajectory, request_event_id) = begin_and_persist_request(
        &state,
        session_id.as_deref(),
        &repo,
        &user_content,
        request_metadata.clone(),
    )
    .await;
    let (context_pack_id, context_pack_hash) = pack_context_into_req(
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
        "chat_completions",
        requested_model.clone(),
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
        return handle_streaming(
            &state,
            req,
            repo,
            task,
            requested_model,
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
        )
        .await;
    }

    handle_openai_non_streaming(
        &state,
        req,
        repo,
        task,
        requested_model,
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
    )
    .await
}
