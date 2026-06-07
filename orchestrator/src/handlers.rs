use axum::extract::State;
#[cfg(test)]
use axum::http::header;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::response::Response;
use bytes::Bytes;
use serde_json::Value;
use std::sync::Arc;

use crate::anthropic;
use crate::auth::{authenticate, check_rate_limit};
#[cfg(test)]
use crate::auth::{provided_api_token, rate_limited_response};
#[cfg(test)]
use crate::context_packing::apply_orchestration_context_limits;
#[cfg(test)]
use crate::context_packing::context_task_category;
use crate::db;
use crate::event_capture::{
    begin_trajectory_for_request, capture_tool_results_background,
    persist_exchange_with_correlation, persist_model_response_event, persist_request_event,
};
use crate::handlers_context::{pack_context_into_anthropic_req, pack_context_into_req};
use crate::handlers_request::HandlerRequestScope;
use crate::handlers_streaming::{handle_streaming, handle_streaming_anthropic};
#[cfg(test)]
use crate::local_reasoning::LocalReasoningPolicy;
#[cfg(test)]
use crate::local_reasoning::LocalReasoningSelection;
use crate::local_reasoning::{
    add_local_reasoning_metadata, apply_local_reasoning_defaults, inject_contract_anthropic,
    inject_contract_openai, local_reasoning_selection,
};
use crate::orchestration_policy;
use crate::proxy_support::{
    baseline_arm_selection, dispatch_non_streaming_raw, extract_user_content_openai, litellm_route,
    merge_request_metadata,
};
use crate::request_policy::{
    maybe_anthropic_live_policy_response, maybe_openai_live_policy_response,
};
use crate::sse::extract_assistant_from_anthropic_response;
#[cfg(test)]
use crate::sse::{
    extract_assistant_from_anthropic_sse, extract_assistant_from_sse,
    extract_token_usage_from_anthropic_sse, extract_token_usage_from_sse,
};
use crate::state::*;
#[cfg(test)]
use crate::system_context::{inject_system_context, inject_system_context_anthropic};
use crate::telemetry;
use crate::token_limits::{
    context_window_retry_max_tokens, enforce_min_max_tokens, set_max_tokens,
};

pub use crate::background::trajectory::run_trajectory_idle_sweep;
pub use crate::context_packing::context_pack;
pub use crate::routes::checkpoints::checkpoint;
pub use crate::routes::context::context_artifacts;
pub use crate::routes::harness::{harness_guardrail, harness_outcome, litellm_callback_payload};
pub use crate::routes::health::{health, health_live, health_ready, list_models};
pub use crate::routes::observability::{cache_stats, metrics, metrics_json};
pub use crate::routes::search::search;
pub use crate::routes::sessions::{append_event, start_session};
pub use crate::routes::tools::authorize_tool;
#[cfg(test)]
pub(crate) use crate::routes::tools::{
    derive_tool_authorization_policy, tool_authorization_classification_text,
};
pub use crate::routes::validations::validations;

// ── Chat completions ────────────────────────────────────────────

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
    let mut req = payload.clone();
    // Always route to the configured backend model regardless of what the client sent.
    let route = litellm_route(&state, &namespace);
    req["model"] = Value::String(route.routed_model.clone());
    apply_local_reasoning_defaults(&mut req, reasoning_selection);
    enforce_min_max_tokens(&mut req);
    inject_contract_openai(&mut req, reasoning_selection);
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
    let session_id = if state.trajectory_capture_enabled {
        match db::find_or_create_session(&state.pool, &repo, &task, "agent").await {
            Ok(session_id) => Some(session_id),
            Err(e) => {
                tracing::warn!("find_or_create_session failed before trajectory capture: {e}");
                None
            }
        }
    } else {
        None
    };
    let request_classification = crate::request_classification::classify_request_text(
        &repo,
        session_id.as_deref().unwrap_or("unknown"),
        &user_content,
        None,
        "user_message",
    );
    let request_policy = orchestration_policy::derive_orchestration_policy(
        &request_classification,
        &user_content,
        state.capture_pool.is_some(),
    );
    let tool_mediation_metadata = if state.tool_mediation_enabled {
        // Shape the OpenAI tool menu with the same policy model used by
        // `/tools/authorize`. This only removes tools from the client-provided
        // menu; proxy mode cannot invent tools the client did not offer.
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
    let trajectory = if let Some(session_id) = session_id.as_deref() {
        Some(begin_trajectory_for_request(&state, session_id).await)
    } else {
        None
    };
    let request_event_id =
        if let (Some(session_id), Some(trajectory)) = (session_id.as_deref(), trajectory) {
            persist_request_event(
                &state,
                session_id,
                &repo,
                &user_content,
                trajectory,
                request_metadata.clone(),
            )
            .await
        } else {
            None
        };
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
    let cache_policy = crate::litellm::exact_cache_decision("chat_completions", &req, false);
    let mut attempt = crate::litellm::new_attempt(
        request_event_id,
        trajectory.map(|trajectory| trajectory.trajectory_id),
        context_pack_id,
        namespace.clone(),
        repo.clone(),
        task.clone(),
        "chat_completions",
        requested_model.clone(),
        &route,
        cache_policy,
        context_pack_hash.clone(),
        Some(baseline_arm.as_str().to_string()),
    );
    add_local_reasoning_metadata(&mut attempt, reasoning_selection);
    capture.attempt_id = Some(attempt.attempt_id);
    crate::litellm::add_agentic_os_metadata(&mut req, &attempt);
    capture.forwarded_request_body = Some(crate::client_capture::to_json_bytes(&req));
    let vllm_cache_before = crate::vllm_metrics::cache_snapshot(&state).await;
    let finalizer = crate::litellm::LiteLlmCallFinalizer::begin(state.pool.clone(), attempt).await;

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
            finalizer,
            vllm_cache_before,
            capture,
        )
        .await;
    }

    // ── Non-streaming ───────────────────────────────────────────
    match dispatch_non_streaming_raw(&state, &req, &finalizer).await {
        Ok((val, latency_ms)) => {
            capture.response_status = Some(StatusCode::OK.as_u16() as i32);
            capture.response_headers =
                Some(serde_json::json!({"content-type": ["application/json"]}));
            capture.raw_response_body = Some(crate::client_capture::to_json_bytes(&val));
            crate::client_capture::record_best_effort(state.capture_pool.as_ref(), capture).await;
            let usage = TokenUsage::from_openai_value(&val);
            let provider_cache = crate::litellm::ProviderCacheCounters::from_value(&val);
            crate::vllm_metrics::record_cache_observation(
                &state,
                vllm_cache_before,
                session_id.as_deref(),
                &namespace,
                &repo,
                &task,
                finalizer.attempt(),
                &usage,
                provider_cache,
            )
            .await;
            telemetry::record_tokens(&state.metrics, &usage, &state.default_model);
            if !usage.is_empty() {
                let pool = state.pool.clone();
                let actual = state.default_model.clone();
                let rm = requested_model.clone();
                let ns = namespace.clone();
                let r = repo.clone();
                let u = usage.clone();
                tokio::spawn(async move {
                    if let Err(e) = db::record_token_usage(&pool, &rm, &actual, &ns, &r, &u).await {
                        tracing::warn!("failed to record token usage: {e}");
                    }
                });
            }

            let assistant_content: String = val["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .chars()
                .take(500)
                .collect();
            // Non-streaming responses may contain provider/tool envelopes. We
            // inspect them after normal exchange persistence has an event parent.
            let tool_results = crate::execution_feedback::tool_results_from_value(&val);

            if let (Some(session_id), Some(trajectory)) = (session_id.as_deref(), trajectory) {
                let (input_tokens, output_tokens) =
                    crate::trajectory::optional_token_counts_from_value(&val);
                let metadata = crate::trajectory::model_response_metadata(
                    &state.default_model,
                    "litellm",
                    input_tokens,
                    output_tokens,
                    Some(latency_ms as i64),
                    crate::trajectory::model_finish_reason(&val),
                    request_metadata.clone(),
                    context_pack_id,
                    Some(trajectory),
                );
                let assistant_event_id = persist_model_response_event(
                    &state,
                    session_id,
                    &repo,
                    &assistant_content,
                    metadata,
                    trajectory,
                    request_event_id,
                )
                .await;
                capture_tool_results_background(
                    state.as_ref().clone(),
                    session_id.to_string(),
                    repo.clone(),
                    task.clone(),
                    trajectory.trajectory_id,
                    assistant_event_id,
                    Some(trajectory),
                    tool_results,
                );
            } else {
                match db::find_or_create_session(&state.pool, &repo, &task, "agent").await {
                    Ok(session_id) => {
                        let assistant_event_id = persist_exchange_with_correlation(
                            &state,
                            &session_id,
                            &repo,
                            &user_content,
                            &assistant_content,
                            correlation_id,
                            request_metadata.clone(),
                        )
                        .await;
                        if let Some(correlation_id) = correlation_id {
                            capture_tool_results_background(
                                state.as_ref().clone(),
                                session_id,
                                repo.clone(),
                                task.clone(),
                                correlation_id,
                                assistant_event_id,
                                None,
                                tool_results,
                            );
                        }
                    }
                    Err(e) => tracing::warn!("find_or_create_session failed: {e}"),
                }
            }
            axum::Json(val).into_response()
        }
        Err(resp) => resp,
    }
}

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

    // Stay in Anthropic format — no translation.
    let mut req = payload;
    let route = litellm_route(&state, &namespace);
    req["model"] = Value::String(route.routed_model.clone());
    apply_local_reasoning_defaults(&mut req, reasoning_selection);
    enforce_min_max_tokens(&mut req);
    anthropic::normalize_response_content_types(&mut req);
    anthropic::sanitize_litellm_request(&mut req);
    inject_contract_anthropic(&mut req, reasoning_selection);
    let session_id = if state.trajectory_capture_enabled {
        match db::find_or_create_session(&state.pool, &repo, &task, "agent").await {
            Ok(session_id) => Some(session_id),
            Err(e) => {
                tracing::warn!("find_or_create_session failed before trajectory capture: {e}");
                None
            }
        }
    } else {
        None
    };
    let request_classification = crate::request_classification::classify_request_text(
        &repo,
        session_id.as_deref().unwrap_or("unknown"),
        &user_content,
        None,
        "user_message",
    );
    let request_policy = orchestration_policy::derive_orchestration_policy(
        &request_classification,
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
    let trajectory = if let Some(session_id) = session_id.as_deref() {
        Some(begin_trajectory_for_request(&state, session_id).await)
    } else {
        None
    };
    let request_event_id =
        if let (Some(session_id), Some(trajectory)) = (session_id.as_deref(), trajectory) {
            persist_request_event(
                &state,
                session_id,
                &repo,
                &user_content,
                trajectory,
                request_metadata.clone(),
            )
            .await
        } else {
            None
        };
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
    let cache_policy = crate::litellm::exact_cache_decision("messages", &req, false);
    let mut attempt = crate::litellm::new_attempt(
        request_event_id,
        trajectory.map(|trajectory| trajectory.trajectory_id),
        context_pack_id,
        namespace.clone(),
        repo.clone(),
        task.clone(),
        "messages",
        model.clone(),
        &route,
        cache_policy,
        context_pack_hash.clone(),
        Some(baseline_arm.as_str().to_string()),
    );
    add_local_reasoning_metadata(&mut attempt, reasoning_selection);
    capture.attempt_id = Some(attempt.attempt_id);
    crate::litellm::add_agentic_os_metadata(&mut req, &attempt);
    capture.forwarded_request_body = Some(crate::client_capture::to_json_bytes(&req));
    let vllm_cache_before = crate::vllm_metrics::cache_snapshot(&state).await;
    let mut finalizer =
        crate::litellm::LiteLlmCallFinalizer::begin(state.pool.clone(), attempt).await;

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
            finalizer,
            vllm_cache_before,
            capture,
        )
        .await;
    }

    // ── Non-streaming: passthrough to LiteLLM /messages ────────
    let url = format!("{}/messages", state.litellm_url);
    let started = std::time::Instant::now();
    let upstream_resp = match state
        .http
        .post(&url)
        .bearer_auth(&state.litellm_key)
        .json(&req)
        .send()
        .await
    {
        Ok(r) => {
            let status = r.status();
            telemetry::record_upstream_litellm(
                "messages",
                started.elapsed(),
                &status.as_u16().to_string(),
            );
            if !status.is_success() {
                telemetry::record_upstream_litellm_error(
                    "messages",
                    telemetry::upstream_error_kind(status),
                );
            }
            r
        }
        Err(e) => {
            telemetry::record_upstream_litellm("messages", started.elapsed(), "error");
            telemetry::record_upstream_litellm_error("messages", telemetry::reqwest_error_kind(&e));
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::NetworkError,
                    Some(telemetry::reqwest_error_kind(&e)),
                    Some(&e.to_string()),
                    crate::litellm::ProviderCacheCounters::default(),
                )
                .await;
            return anthropic::error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                "upstream LiteLLM request failed",
            );
        }
    };

    let mut status = upstream_resp.status();
    let mut val: Value = match upstream_resp.json().await {
        Ok(v) => v,
        Err(_) => {
            telemetry::record_upstream_litellm_error("messages", "parse");
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::ParseError,
                    Some("parse"),
                    Some("invalid upstream response"),
                    crate::litellm::ProviderCacheCounters::default(),
                )
                .await;
            return anthropic::error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                "invalid upstream response",
            );
        }
    };

    if !status.is_success() {
        let error_body = val.to_string();
        if let Some(retry_max_tokens) = context_window_retry_max_tokens(&error_body) {
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::HttpError,
                    Some(telemetry::upstream_error_kind(status)),
                    Some("upstream returned non-success status"),
                    crate::litellm::ProviderCacheCounters::from_value(&val),
                )
                .await;
            tracing::warn!(
                retry_max_tokens,
                upstream_status = status.as_u16(),
                "retrying messages request with reduced max_tokens after context window error"
            );
            set_max_tokens(&mut req, retry_max_tokens);
            let retry_attempt = crate::litellm::new_attempt(
                request_event_id,
                trajectory.map(|trajectory| trajectory.trajectory_id),
                context_pack_id,
                namespace.clone(),
                repo.clone(),
                task.clone(),
                "messages",
                model.clone(),
                &route,
                crate::litellm::exact_cache_decision("messages", &req, false),
                context_pack_hash.clone(),
                Some(baseline_arm.as_str().to_string()),
            );
            crate::litellm::add_agentic_os_metadata(&mut req, &retry_attempt);
            finalizer =
                crate::litellm::LiteLlmCallFinalizer::begin(state.pool.clone(), retry_attempt)
                    .await;
            let retry_started = std::time::Instant::now();
            let retry_resp = match state
                .http
                .post(&url)
                .bearer_auth(&state.litellm_key)
                .json(&req)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    telemetry::record_upstream_litellm(
                        "messages",
                        retry_started.elapsed(),
                        "error",
                    );
                    telemetry::record_upstream_litellm_error(
                        "messages",
                        telemetry::reqwest_error_kind(&e),
                    );
                    finalizer
                        .finalize(
                            crate::litellm::TerminalStatus::NetworkError,
                            Some(telemetry::reqwest_error_kind(&e)),
                            Some(&e.to_string()),
                            crate::litellm::ProviderCacheCounters::default(),
                        )
                        .await;
                    return anthropic::error(
                        StatusCode::BAD_GATEWAY,
                        "api_error",
                        "upstream LiteLLM retry failed",
                    );
                }
            };
            status = retry_resp.status();
            telemetry::record_upstream_litellm(
                "messages",
                retry_started.elapsed(),
                &status.as_u16().to_string(),
            );
            val = match retry_resp.json().await {
                Ok(v) => v,
                Err(_) => {
                    telemetry::record_upstream_litellm_error("messages", "parse");
                    finalizer
                        .finalize(
                            crate::litellm::TerminalStatus::ParseError,
                            Some("parse"),
                            Some("invalid upstream retry response"),
                            crate::litellm::ProviderCacheCounters::default(),
                        )
                        .await;
                    return anthropic::error(
                        StatusCode::BAD_GATEWAY,
                        "api_error",
                        "invalid upstream retry response",
                    );
                }
            };
        }
    }

    if !status.is_success() {
        finalizer
            .finalize(
                crate::litellm::TerminalStatus::HttpError,
                Some(telemetry::upstream_error_kind(status)),
                Some("upstream returned non-success status"),
                crate::litellm::ProviderCacheCounters::from_value(&val),
            )
            .await;
        return (status, axum::Json(val)).into_response();
    }
    finalizer
        .finalize(
            crate::litellm::TerminalStatus::Success,
            None,
            None,
            crate::litellm::ProviderCacheCounters::from_value(&val),
        )
        .await;

    let usage = TokenUsage::from_openai_value(&val);
    let provider_cache = crate::litellm::ProviderCacheCounters::from_value(&val);
    let vllm_delta = crate::vllm_metrics::record_cache_observation(
        &state,
        vllm_cache_before,
        session_id.as_deref(),
        &namespace,
        &repo,
        &task,
        finalizer.attempt(),
        &usage,
        provider_cache,
    )
    .await;
    let provider_cache =
        crate::vllm_metrics::merge_provider_cache_from_delta(provider_cache, vllm_delta);
    crate::vllm_metrics::inject_anthropic_cache_usage(&mut val, provider_cache);
    capture.response_status = Some(StatusCode::OK.as_u16() as i32);
    capture.response_headers = Some(serde_json::json!({"content-type": ["application/json"]}));
    capture.raw_response_body = Some(crate::client_capture::to_json_bytes(&val));
    crate::client_capture::record_best_effort(state.capture_pool.as_ref(), capture).await;
    telemetry::record_tokens(&state.metrics, &usage, &state.default_model);
    if !usage.is_empty() {
        let pool = state.pool.clone();
        let actual = state.default_model.clone();
        let rm = model.clone();
        let ns = namespace.clone();
        let r = repo.clone();
        let u = usage.clone();
        tokio::spawn(async move {
            if let Err(e) = db::record_token_usage(&pool, &rm, &actual, &ns, &r, &u).await {
                tracing::warn!("failed to record token usage: {e}");
            }
        });
    }

    let assistant_content = extract_assistant_from_anthropic_response(&val);
    // LiteLLM may surface Anthropic tool results inside the response content.
    // Feed them through the same deterministic parser used for OpenAI results.
    let tool_results = crate::execution_feedback::tool_results_from_value(&val);
    if let (Some(session_id), Some(trajectory)) = (session_id.as_deref(), trajectory) {
        let (input_tokens, output_tokens) =
            crate::trajectory::optional_token_counts_from_value(&val);
        let metadata = crate::trajectory::model_response_metadata(
            &state.default_model,
            "litellm",
            input_tokens,
            output_tokens,
            Some(started.elapsed().as_millis() as i64),
            crate::trajectory::model_finish_reason(&val),
            request_metadata.clone(),
            context_pack_id,
            Some(trajectory),
        );
        let assistant_event_id = persist_model_response_event(
            &state,
            session_id,
            &repo,
            &assistant_content,
            metadata,
            trajectory,
            request_event_id,
        )
        .await;
        capture_tool_results_background(
            state.as_ref().clone(),
            session_id.to_string(),
            repo.clone(),
            task.clone(),
            trajectory.trajectory_id,
            assistant_event_id,
            Some(trajectory),
            tool_results,
        );
    } else {
        match db::find_or_create_session(&state.pool, &repo, &task, "agent").await {
            Ok(sid) => {
                let assistant_event_id = persist_exchange_with_correlation(
                    &state,
                    &sid,
                    &repo,
                    &user_content,
                    &assistant_content,
                    correlation_id,
                    None,
                )
                .await;
                if let Some(correlation_id) = correlation_id {
                    capture_tool_results_background(
                        state.as_ref().clone(),
                        sid,
                        repo.clone(),
                        task.clone(),
                        correlation_id,
                        assistant_event_id,
                        None,
                        tool_results,
                    );
                }
            }
            Err(e) => tracing::warn!("messages: find_or_create_session failed: {e}"),
        }
    }

    axum::Json(val).into_response()
}

#[cfg(test)]
#[path = "handlers_policy_tests.rs"]
mod handlers_policy_tests;

#[cfg(test)]
#[path = "handlers_tests.rs"]
mod handlers_tests;
