use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::response::Response;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;
use std::sync::Arc;

use crate::anthropic;
use crate::auth::{authenticate, check_rate_limit};
#[cfg(test)]
use crate::auth::{provided_api_token, rate_limited_response};
#[cfg(test)]
use crate::context_packing::apply_orchestration_context_limits;
use crate::context_packing::{cached_context_for_request, context_task_category};
use crate::db;
use crate::event_capture::{
    begin_trajectory_for_request, capture_tool_results_background, maybe_write_context_pack_event,
    persist_exchange_with_correlation, persist_model_response_event, persist_request_event,
};
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
use crate::sse::{
    extract_assistant_from_anthropic_response, extract_assistant_from_anthropic_sse,
    extract_assistant_from_sse, extract_token_usage_from_anthropic_sse,
    extract_token_usage_from_sse, optional_token_usage_from_sse,
};
use crate::state::*;
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

/// Pack orchestrator context into an Anthropic-format request's system field.
async fn pack_context_into_anthropic_req(
    state: &AppState,
    req: &mut Value,
    session_id: Option<&str>,
    repo: &str,
    task: &str,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    parent_event_id: Option<uuid::Uuid>,
    classification: Option<&crate::request_classification::RequestClassification>,
    policy: Option<&crate::orchestration_policy::OrchestrationPolicy>,
) -> (Option<uuid::Uuid>, Option<String>) {
    let task_category = context_task_category(task, classification);
    let task_config = crate::state::TaskContextConfig::for_category(task_category);
    let cached = cached_context_for_request(
        state,
        repo,
        task,
        session_id,
        trajectory,
        &task_config,
        classification,
        policy,
    );
    let context_pack_id = maybe_write_context_pack_event(
        state,
        session_id,
        repo,
        task,
        trajectory,
        parent_event_id,
        &cached.stats,
        task_config.char_budget / 4,
    );
    let context_pack_hash = Some(crate::litellm::context_pack_hash(&cached.context));
    inject_system_context_anthropic(req, &cached.context);
    (context_pack_id, context_pack_hash)
}

/// Pack orchestrator context into an OpenAI-shaped request.
/// Sets a default model if absent, fetches memory events, builds context string,
/// and injects it as a system message.
async fn pack_context_into_req(
    state: &AppState,
    req: &mut Value,
    session_id: Option<&str>,
    repo: &str,
    task: &str,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    parent_event_id: Option<uuid::Uuid>,
    classification: Option<&crate::request_classification::RequestClassification>,
    policy: Option<&crate::orchestration_policy::OrchestrationPolicy>,
) -> (Option<uuid::Uuid>, Option<String>) {
    if req.get("model").is_none() {
        req["model"] = Value::String(state.default_model.clone());
    }
    let task_category = context_task_category(task, classification);
    let task_config = crate::state::TaskContextConfig::for_category(task_category);
    let cached = cached_context_for_request(
        state,
        repo,
        task,
        session_id,
        trajectory,
        &task_config,
        classification,
        policy,
    );
    let context_pack_id = maybe_write_context_pack_event(
        state,
        session_id,
        repo,
        task,
        trajectory,
        parent_event_id,
        &cached.stats,
        task_config.char_budget / 4,
    );
    let context_pack_hash = Some(crate::litellm::context_pack_hash(&cached.context));
    inject_system_context(req, &cached.context);
    (context_pack_id, context_pack_hash)
}

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

    // Explicit headers take precedence; fall back to token-bound namespace so
    // standard clients (OpenCode, curl) get memory without custom headers.
    let repo = headers
        .get("x-agent-repo")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| namespace.clone());
    let task = headers
        .get("x-agent-task")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| state.default_task.clone());
    tracing::info!(repo = %repo, task = %task, "routing request");
    capture.namespace = Some(namespace.clone());
    capture.repo = Some(repo.clone());
    capture.task = Some(task.clone());

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

#[allow(clippy::too_many_arguments)]
async fn handle_streaming(
    state: &AppState,
    req: Value,
    repo: String,
    task: String,
    requested_model: String,
    namespace: String,
    correlation_id: Option<uuid::Uuid>,
    request_metadata: Option<Value>,
    session_id: Option<String>,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    request_event_id: Option<uuid::Uuid>,
    context_pack_id: Option<uuid::Uuid>,
    user_content: String,
    mut finalizer: crate::litellm::LiteLlmCallFinalizer,
    vllm_cache_before: Option<(String, crate::vllm_metrics::VllmCacheSnapshot)>,
    capture: crate::client_capture::RawHttpCapture,
) -> Response {
    let url = format!("{}/chat/completions", state.litellm_url);
    let mut capture = capture;

    let started = finalizer.attempt_mut().started_at;
    let upstream = match state
        .http_stream
        .post(&url)
        .bearer_auth(&state.litellm_key)
        .json(&req)
        .send()
        .await
    {
        Ok(r) => {
            let status = r.status();
            telemetry::record_upstream_litellm(
                "chat_completions",
                started.elapsed(),
                &status.as_u16().to_string(),
            );
            if !status.is_success() {
                telemetry::record_upstream_litellm_error(
                    "chat_completions",
                    telemetry::upstream_error_kind(status),
                );
            }
            r
        }
        Err(e) => {
            telemetry::record_upstream_litellm("chat_completions", started.elapsed(), "error");
            telemetry::record_upstream_litellm_error(
                "chat_completions",
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
            return (
                StatusCode::BAD_GATEWAY,
                axum::Json(
                    serde_json::json!({"error": "litellm_unreachable", "detail": e.to_string()}),
                ),
            )
                .into_response();
        }
    };

    if !upstream.status().is_success() {
        let status = upstream.status();
        telemetry::record_upstream_litellm_error(
            "chat_completions",
            telemetry::upstream_error_kind(status),
        );
        let body = upstream.text().await.unwrap_or_default();
        finalizer
            .finalize(
                crate::litellm::TerminalStatus::HttpError,
                Some(telemetry::upstream_error_kind(status)),
                Some("upstream returned non-success status"),
                crate::litellm::ProviderCacheCounters::default(),
            )
            .await;
        return (status, [(header::CONTENT_TYPE, "application/json")], body).into_response();
    }

    let bytes_stream = upstream.bytes_stream();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Vec<u8>>();
    let accumulated = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let acc_clone = accumulated.clone();
    let mut stream_finalizer = finalizer;
    let observation_attempt = stream_finalizer.attempt().clone();

    let tapped = async_stream::stream! {
        tokio::pin!(bytes_stream);
        let mut stream_metrics = telemetry::StreamTracker::new("chat_completions", started);
        let mut tx_opt = Some(done_tx);
        let mut first_seen = false;
        let mut first_token_at = None;
        while let Some(chunk) = bytes_stream.next().await {
            match chunk {
                Ok(b) => {
                    if !first_seen && crate::litellm::sse_chunk_has_non_empty_data(&b) {
                        first_seen = true;
                        first_token_at = Some(std::time::Instant::now());
                        stream_finalizer.attempt_mut().first_token_at = first_token_at;
                        stream_metrics.first_token();
                    }
                    if let Ok(mut g) = acc_clone.lock() { g.extend_from_slice(&b); }
                    yield Ok::<Bytes, std::io::Error>(b);
                }
                Err(e) => {
                    stream_metrics.fail("upstream_error");
                    stream_finalizer
                        .finalize(
                            crate::litellm::TerminalStatus::StreamError,
                            Some("upstream_error"),
                            Some(&e.to_string()),
                            crate::litellm::ProviderCacheCounters::default(),
                        )
                        .await;
                    yield Err(std::io::Error::other(e));
                    return;
                }
            }
        }
        stream_metrics.finish();
        let data = acc_clone.lock().map(|g| g.clone()).unwrap_or_default();
        let raw = String::from_utf8_lossy(&data);
        let counters = crate::litellm::provider_counters_from_sse(&raw);
        let mut done_finalizer = stream_finalizer.clone();
        done_finalizer.attempt_mut().first_token_at = first_token_at;
        done_finalizer.attempt_mut().completed_at = Some(std::time::Instant::now());
        done_finalizer
            .finalize(crate::litellm::TerminalStatus::Success, None, None, counters)
            .await;
        if let Some(tx) = tx_opt.take() {
            let _ = tx.send(data);
        }
    };

    let state_bg = state.clone();
    tokio::spawn(async move {
        if let Ok(raw_bytes) = done_rx.await {
            capture.response_status = Some(StatusCode::OK.as_u16() as i32);
            capture.response_headers =
                Some(serde_json::json!({"content-type": ["text/event-stream"]}));
            capture.raw_response_body = Some(raw_bytes.clone());
            crate::client_capture::record_best_effort(state_bg.capture_pool.as_ref(), capture)
                .await;
            let raw = String::from_utf8_lossy(&raw_bytes);
            let usage = extract_token_usage_from_sse(&raw);
            let provider_cache = crate::litellm::provider_counters_from_sse(&raw);
            crate::vllm_metrics::record_cache_observation(
                &state_bg,
                vllm_cache_before,
                session_id.as_deref(),
                &namespace,
                &repo,
                &task,
                &observation_attempt,
                &usage,
                provider_cache,
            )
            .await;
            telemetry::record_tokens(&state_bg.metrics, &usage, &state_bg.default_model);
            if !usage.is_empty() {
                if let Err(e) = db::record_token_usage(
                    &state_bg.pool,
                    &requested_model,
                    &state_bg.default_model,
                    &namespace,
                    &repo,
                    &usage,
                )
                .await
                {
                    tracing::warn!("failed to record token usage: {e}");
                }
            }
            let assistant_content = extract_assistant_from_sse(&raw);
            // Stream capture reuses the accumulated bytes already needed for
            // token accounting, so no extra read is introduced.
            let tool_results = crate::execution_feedback::tool_results_from_sse(&raw);
            if let (Some(session_id), Some(trajectory)) = (session_id.as_deref(), trajectory) {
                let (input_tokens, output_tokens) = optional_token_usage_from_sse(&raw);
                let metadata = crate::trajectory::model_response_metadata(
                    &state_bg.default_model,
                    "litellm",
                    input_tokens,
                    output_tokens,
                    Some(started.elapsed().as_millis() as i64),
                    None,
                    request_metadata.clone(),
                    context_pack_id,
                    Some(trajectory),
                );
                let assistant_event_id = persist_model_response_event(
                    &state_bg,
                    session_id,
                    &repo,
                    &assistant_content,
                    metadata,
                    trajectory,
                    request_event_id,
                )
                .await;
                capture_tool_results_background(
                    state_bg.clone(),
                    session_id.to_string(),
                    repo.clone(),
                    task.clone(),
                    trajectory.trajectory_id,
                    assistant_event_id,
                    Some(trajectory),
                    tool_results,
                );
            } else {
                match db::find_or_create_session(&state_bg.pool, &repo, &task, "agent").await {
                    Ok(sid) => {
                        let assistant_event_id = persist_exchange_with_correlation(
                            &state_bg,
                            &sid,
                            &repo,
                            &user_content,
                            &assistant_content,
                            correlation_id,
                            request_metadata,
                        )
                        .await;
                        if let Some(correlation_id) = correlation_id {
                            capture_tool_results_background(
                                state_bg.clone(),
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
                    Err(e) => tracing::warn!("stream: find_or_create_session failed: {e}"),
                }
            }
        }
    });

    (
        [
            (axum::http::header::CONTENT_TYPE, "text/event-stream"),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
            (axum::http::header::CONNECTION, "keep-alive"),
        ],
        axum::body::Body::from_stream(tapped),
    )
        .into_response()
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

    let repo = headers
        .get("x-agent-repo")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| namespace.clone());
    let task = headers
        .get("x-agent-task")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| state.default_task.clone());
    tracing::info!(repo = %repo, task = %task, endpoint = "messages", "routing request");
    capture.namespace = Some(namespace.clone());
    capture.repo = Some(repo.clone());
    capture.task = Some(task.clone());

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

/// Proxy an Anthropic streaming request to LiteLLM /messages, passing bytes through unchanged.
#[allow(clippy::too_many_arguments)]
async fn handle_streaming_anthropic(
    state: &AppState,
    req: Value,
    user_content: String,
    repo: String,
    task: String,
    model: String,
    namespace: String,
    correlation_id: Option<uuid::Uuid>,
    request_metadata: Option<Value>,
    session_id: Option<String>,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    request_event_id: Option<uuid::Uuid>,
    context_pack_id: Option<uuid::Uuid>,
    mut finalizer: crate::litellm::LiteLlmCallFinalizer,
    vllm_cache_before: Option<(String, crate::vllm_metrics::VllmCacheSnapshot)>,
    capture: crate::client_capture::RawHttpCapture,
) -> Response {
    let url = format!("{}/messages", state.litellm_url);
    let mut req = req;
    let mut capture = capture;

    let started = finalizer.attempt_mut().started_at;
    let mut upstream = match state
        .http_stream
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
                format!("upstream unreachable: {e}"),
            );
        }
    };

    let upstream_status = upstream.status();
    if !upstream_status.is_success() {
        let body = upstream.text().await.unwrap_or_else(|e| {
            serde_json::json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": format!("failed to read upstream error body: {e}")
                }
            })
            .to_string()
        });
        if let Some(retry_max_tokens) = context_window_retry_max_tokens(&body) {
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::HttpError,
                    Some(telemetry::upstream_error_kind(upstream_status)),
                    Some("upstream returned non-success status"),
                    crate::litellm::ProviderCacheCounters::default(),
                )
                .await;
            tracing::warn!(
                retry_max_tokens,
                upstream_status = upstream_status.as_u16(),
                "retrying streaming messages request with reduced max_tokens after context window error"
            );
            set_max_tokens(&mut req, retry_max_tokens);
            let prior_attempt = finalizer.attempt_mut().clone();
            let route = crate::litellm::RouteSelection {
                routed_model: prior_attempt.routed_model.clone(),
                selected_route: prior_attempt
                    .selected_route
                    .clone()
                    .unwrap_or_else(|| "default".to_string()),
                selection_reason: prior_attempt
                    .selection_reason
                    .clone()
                    .unwrap_or_else(|| "default_model".to_string()),
                policy_version: prior_attempt
                    .policy_version
                    .clone()
                    .unwrap_or_else(|| "default-routing-v1".to_string()),
            };
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
                prior_attempt.context_pack_hash.clone(),
                prior_attempt.baseline_arm.clone(),
            );
            crate::litellm::add_agentic_os_metadata(&mut req, &retry_attempt);
            finalizer =
                crate::litellm::LiteLlmCallFinalizer::begin(state.pool.clone(), retry_attempt)
                    .await;
            let retry_started = std::time::Instant::now();
            upstream = match state
                .http_stream
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
                        retry_started.elapsed(),
                        &status.as_u16().to_string(),
                    );
                    r
                }
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
                        format!("upstream retry unreachable: {e}"),
                    );
                }
            };
            if upstream.status().is_success() {
                // Continue below with the successful retry stream.
            } else {
                let retry_status = upstream.status();
                let retry_body = upstream.text().await.unwrap_or_else(|e| {
                    serde_json::json!({
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": format!("failed to read upstream retry error body: {e}")
                        }
                    })
                    .to_string()
                });
                finalizer
                    .finalize(
                        crate::litellm::TerminalStatus::HttpError,
                        Some(telemetry::upstream_error_kind(retry_status)),
                        Some("upstream returned non-success status"),
                        crate::litellm::ProviderCacheCounters::default(),
                    )
                    .await;
                return (
                    retry_status,
                    [(header::CONTENT_TYPE, "application/json")],
                    retry_body,
                )
                    .into_response();
            }
        } else {
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::HttpError,
                    Some(telemetry::upstream_error_kind(upstream_status)),
                    Some("upstream returned non-success status"),
                    crate::litellm::ProviderCacheCounters::default(),
                )
                .await;
            return (
                upstream_status,
                [(header::CONTENT_TYPE, "application/json")],
                body,
            )
                .into_response();
        }
    }

    let bytes_stream = upstream.bytes_stream();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Vec<u8>>();
    let accumulated = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let acc_clone = accumulated.clone();
    let mut stream_finalizer = finalizer;
    let observation_attempt = stream_finalizer.attempt().clone();
    let cache_state = state.clone();
    let cache_session_id = session_id.clone();
    let cache_namespace = namespace.clone();
    let cache_repo = repo.clone();
    let cache_task = task.clone();

    // Proxy bytes verbatim — LiteLLM already returns Anthropic SSE format.
    let tapped = async_stream::stream! {
        tokio::pin!(bytes_stream);
        let mut stream_metrics = telemetry::StreamTracker::new("messages", started);
        let mut tx_opt = Some(done_tx);
        let mut first_seen = false;
        let mut first_token_at = None;
        while let Some(chunk) = bytes_stream.next().await {
            match chunk {
                Ok(b) => {
                    if !first_seen && crate::litellm::sse_chunk_has_non_empty_data(&b) {
                        first_seen = true;
                        first_token_at = Some(std::time::Instant::now());
                        stream_finalizer.attempt_mut().first_token_at = first_token_at;
                        stream_metrics.first_token();
                    }
                    if let Ok(mut g) = acc_clone.lock() { g.extend_from_slice(&b); }
                    yield Ok::<Bytes, std::io::Error>(b);
                }
                Err(e) => {
                    stream_metrics.fail("upstream_error");
                    stream_finalizer
                        .finalize(
                            crate::litellm::TerminalStatus::StreamError,
                            Some("upstream_error"),
                            Some(&e.to_string()),
                            crate::litellm::ProviderCacheCounters::default(),
                        )
                        .await;
                    yield Err(std::io::Error::other(e));
                    return;
                }
            }
        }
        stream_metrics.finish();
        let mut data = acc_clone.lock().map(|g| g.clone()).unwrap_or_default();
        let raw = String::from_utf8_lossy(&data);
        let usage = extract_token_usage_from_anthropic_sse(&raw);
        let provider_cache = crate::litellm::provider_counters_from_sse(&raw);
        let vllm_delta = crate::vllm_metrics::record_cache_observation(
            &cache_state,
            vllm_cache_before,
            cache_session_id.as_deref(),
            &cache_namespace,
            &cache_repo,
            &cache_task,
            &observation_attempt,
            &usage,
            provider_cache,
        )
        .await;
        let counters = crate::vllm_metrics::merge_provider_cache_from_delta(provider_cache, vllm_delta);
        let cache_usage_event = crate::vllm_metrics::anthropic_cache_usage_sse_event(counters);
        data.extend_from_slice(cache_usage_event.as_bytes());
        let mut done_finalizer = stream_finalizer.clone();
        done_finalizer.attempt_mut().first_token_at = first_token_at;
        done_finalizer.attempt_mut().completed_at = Some(std::time::Instant::now());
        done_finalizer
            .finalize(crate::litellm::TerminalStatus::Success, None, None, counters)
            .await;
        yield Ok::<Bytes, std::io::Error>(Bytes::from(cache_usage_event));
        if let Some(tx) = tx_opt.take() {
            let _ = tx.send(data);
        }
    };

    let state_bg = state.clone();
    tokio::spawn(async move {
        if let Ok(raw_bytes) = done_rx.await {
            capture.response_status = Some(StatusCode::OK.as_u16() as i32);
            capture.response_headers =
                Some(serde_json::json!({"content-type": ["text/event-stream"]}));
            capture.raw_response_body = Some(raw_bytes.clone());
            crate::client_capture::record_best_effort(state_bg.capture_pool.as_ref(), capture)
                .await;
            let raw = String::from_utf8_lossy(&raw_bytes);
            let usage = extract_token_usage_from_anthropic_sse(&raw);
            telemetry::record_tokens(&state_bg.metrics, &usage, &state_bg.default_model);
            if !usage.is_empty() {
                if let Err(e) = db::record_token_usage(
                    &state_bg.pool,
                    &model,
                    &state_bg.default_model,
                    &namespace,
                    &repo,
                    &usage,
                )
                .await
                {
                    tracing::warn!("failed to record token usage: {e}");
                }
            }
            let assistant_content = extract_assistant_from_anthropic_sse(&raw);
            // Anthropic streaming tool_result blocks are parsed from the final
            // accumulated SSE transcript after the upstream stream completes.
            let tool_results = crate::execution_feedback::tool_results_from_sse(&raw);
            if let (Some(session_id), Some(trajectory)) = (session_id.as_deref(), trajectory) {
                let (input_tokens, output_tokens) = optional_token_usage_from_sse(&raw);
                let metadata = crate::trajectory::model_response_metadata(
                    &state_bg.default_model,
                    "litellm",
                    input_tokens,
                    output_tokens,
                    Some(started.elapsed().as_millis() as i64),
                    None,
                    request_metadata.clone(),
                    context_pack_id,
                    Some(trajectory),
                );
                let assistant_event_id = persist_model_response_event(
                    &state_bg,
                    session_id,
                    &repo,
                    &assistant_content,
                    metadata,
                    trajectory,
                    request_event_id,
                )
                .await;
                capture_tool_results_background(
                    state_bg.clone(),
                    session_id.to_string(),
                    repo.clone(),
                    task.clone(),
                    trajectory.trajectory_id,
                    assistant_event_id,
                    Some(trajectory),
                    tool_results,
                );
            } else {
                match db::find_or_create_session(&state_bg.pool, &repo, &task, "agent").await {
                    Ok(sid) => {
                        let assistant_event_id = persist_exchange_with_correlation(
                            &state_bg,
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
                                state_bg.clone(),
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
                    Err(e) => tracing::warn!("messages stream: find_or_create_session failed: {e}"),
                }
            }
        }
    });

    (
        [
            (axum::http::header::CONTENT_TYPE, "text/event-stream"),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
            (axum::http::header::CONNECTION, "keep-alive"),
        ],
        axum::body::Body::from_stream(tapped),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;

    #[test]
    fn derive_tool_authorization_policy_uses_request_fields() {
        let req = crate::tool_mediation::ToolAuthorizeRequest {
            session_id: Some("session-a".to_string()),
            repo: Some("repo-a".to_string()),
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("explain the logs".to_string()),
            tool_name: "Read".to_string(),
            arguments: json!({}),
            available_tools: vec![],
        };

        let (classification, policy) = derive_tool_authorization_policy(&req, "namespace-a", false);

        assert_eq!(classification.repo, "repo-a");
        assert_eq!(classification.session_id, "session-a");
        assert!(policy
            .scope_policy
            .contains(&orchestration_policy::ScopePolicy::NoScp));
    }

    #[test]
    fn derive_tool_authorization_policy_falls_back_to_namespace_and_tool_name() {
        let req = crate::tool_mediation::ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: None,
            tool_name: "Bash".to_string(),
            arguments: json!({}),
            available_tools: vec![],
        };

        let (classification, policy) = derive_tool_authorization_policy(&req, "namespace-a", false);

        assert_eq!(classification.repo, "namespace-a");
        assert_eq!(classification.session_id, "unknown");
        assert!(policy
            .scope_policy
            .contains(&orchestration_policy::ScopePolicy::NoScp));
    }

    #[test]
    fn live_classification_overrides_generic_task_for_context_category() {
        let classification = crate::request_classification::classify_request_text(
            "agentic-os",
            "session-a",
            "Read README.md",
            None,
            "user_message",
        );

        assert_eq!(
            context_task_category("default task", Some(&classification)),
            crate::state::TaskCategory::Narrow
        );
    }

    #[test]
    fn orchestration_policy_narrows_context_memory_levels() {
        let classification = crate::request_classification::classify_request_text(
            "agentic-os",
            "session-a",
            "Read README.md",
            None,
            "user_message",
        );
        let orchestration_policy = orchestration_policy::derive_orchestration_policy(
            &classification,
            "Read README.md",
            false,
        );
        let mut context_policy =
            crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Moderate);

        apply_orchestration_context_limits(&mut context_policy, &orchestration_policy);

        assert_eq!(context_policy.l2_limit, 0);
        assert_eq!(context_policy.l3_limit, 0);
        assert_eq!(context_policy.failure_limit, 0);
        assert!(!orchestration_policy
            .context_sources
            .contains(&orchestration_policy::ContextSource::CompiledSummaries));
        assert!(!orchestration_policy
            .context_sources
            .contains(&orchestration_policy::ContextSource::TotalRecall));
    }

    // ── Tool authorization classification text ────────────────────────

    #[test]
    fn derive_tool_authorization_policy_sees_destructive_command_arguments() {
        let req = crate::tool_mediation::ToolAuthorizeRequest {
            session_id: Some("session-a".to_string()),
            repo: Some("repo-a".to_string()),
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("run a command".to_string()),
            tool_name: "Bash".to_string(),
            arguments: json!({"command":"rm -rf /tmp/example"}),
            available_tools: vec![],
        };

        let (classification, policy) = derive_tool_authorization_policy(&req, "namespace-a", false);

        assert!(classification
            .risk
            .contains(&crate::request_classification::RequestRisk::DestructiveCommand));
        assert!(policy
            .blocked_tools
            .contains(&orchestration_policy::ToolCapability::ShellMutation));
    }

    #[test]
    fn derive_tool_authorization_policy_allows_search_command_arguments_without_destructive_risk() {
        let req = crate::tool_mediation::ToolAuthorizeRequest {
            session_id: Some("session-a".to_string()),
            repo: Some("repo-a".to_string()),
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: None,
            tool_name: "Bash".to_string(),
            arguments: json!({"command":"rg pattern src"}),
            available_tools: vec![],
        };

        let (classification, policy) = derive_tool_authorization_policy(&req, "namespace-a", false);

        assert!(!classification
            .risk
            .contains(&crate::request_classification::RequestRisk::DestructiveCommand));
        assert!(policy
            .scope_policy
            .contains(&orchestration_policy::ScopePolicy::NoScp));
    }

    #[test]
    fn tool_authorization_classification_text_ignores_unknown_argument_keys() {
        let req = crate::tool_mediation::ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: None,
            tool_name: "Bash".to_string(),
            arguments: json!({"secret":"sk-test", "command":"echo ok"}),
            available_tools: vec![],
        };

        let text = tool_authorization_classification_text(&req);

        assert!(!text.contains("sk-test"));
        assert!(text.contains("command:echo ok"));
    }

    // ── Auth headers ──────────────────────────────────────────────────

    #[test]
    fn provided_api_token_accepts_bearer_authorization() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            "Bearer sk-test".parse().expect("valid header"),
        );

        assert_eq!(provided_api_token(&headers), "sk-test");
    }

    #[test]
    fn provided_api_token_accepts_anthropic_x_api_key() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "sk-test".parse().expect("valid header"));

        assert_eq!(provided_api_token(&headers), "sk-test");
    }

    // ── inject_system_context ──────────────────────────────────────────

    #[test]
    fn inject_inserts_system_message_when_none_exists() {
        let mut payload = json!({
            "messages": [{"role": "user", "content": "hello"}]
        });
        inject_system_context(&mut payload, "prior context");
        let msgs = payload["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "prior context");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn inject_appends_to_existing_system_message_not_duplicates_it() {
        let mut payload = json!({
            "messages": [
                {"role": "system", "content": "base prompt"},
                {"role": "user", "content": "hello"}
            ]
        });
        inject_system_context(&mut payload, "prior context");
        let msgs = payload["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2, "must not insert a second system message");
        let content = msgs[0]["content"].as_str().unwrap();
        assert!(content.contains("base prompt"));
        assert!(content.contains("prior context"));
    }

    #[test]
    fn inject_keeps_system_message_at_index_zero() {
        let mut payload = json!({
            "messages": [
                {"role": "system", "content": "base prompt"},
                {"role": "user", "content": "hello"}
            ]
        });
        inject_system_context(&mut payload, "context");
        let msgs = payload["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
    }

    #[test]
    fn inject_is_noop_when_messages_missing() {
        let mut payload = json!({"model": "gpt-4"});
        inject_system_context(&mut payload, "context");
        assert!(payload.get("messages").is_none());
    }

    // ── extract_assistant_from_sse ─────────────────────────────────────

    #[test]
    fn extract_sse_assembles_content_across_chunks() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\
                   data: [DONE]\n";
        assert_eq!(extract_assistant_from_sse(sse), "Hello world");
    }

    #[test]
    fn extract_sse_stops_at_done() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"before\"}}]}\n\
                   data: [DONE]\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"after\"}}]}\n";
        assert_eq!(extract_assistant_from_sse(sse), "before");
    }

    #[test]
    fn extract_sse_returns_empty_for_blank_input() {
        assert_eq!(extract_assistant_from_sse(""), "");
    }

    #[test]
    fn extract_sse_ignores_malformed_json_data_lines() {
        let sse = "data: not-valid-json\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\
                   data: [DONE]\n";
        assert_eq!(extract_assistant_from_sse(sse), "ok");
    }

    #[test]
    fn extract_sse_ignores_non_data_lines() {
        let sse = ": keep-alive\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\
                   data: [DONE]\n";
        assert_eq!(extract_assistant_from_sse(sse), "hi");
    }

    #[test]
    fn extract_sse_usage_records_processed_cached_and_generated_tokens() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\
                   data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":40,\"prompt_tokens_details\":{\"cached_tokens\":80}}}\n\
                   data: [DONE]\n";
        let usage = extract_token_usage_from_sse(sse);
        assert_eq!(usage.processed_tokens, 120);
        assert_eq!(usage.cached_tokens, 80);
        assert_eq!(usage.generated_tokens, 40);
    }

    #[test]
    fn local_reasoning_policy_prefers_header_over_client_fields() {
        let mut headers = HeaderMap::new();
        headers.insert("x-agent-reasoning-policy", "high".parse().unwrap());
        let payload = serde_json::json!({
            "reasoning_effort": "low",
            "thinking": {"type": "enabled", "budget_tokens": 1024}
        });

        let selection = local_reasoning_selection(&headers, &payload);

        assert_eq!(selection.policy, LocalReasoningPolicy::High);
        assert_eq!(selection.source, "x-agent-reasoning-policy");
    }

    #[test]
    fn local_reasoning_policy_maps_anthropic_thinking_budget() {
        let headers = HeaderMap::new();

        let low = local_reasoning_selection(
            &headers,
            &serde_json::json!({"thinking": {"budget_tokens": 1024}}),
        );
        let medium = local_reasoning_selection(
            &headers,
            &serde_json::json!({"thinking": {"budget_tokens": 4096}}),
        );
        let high = local_reasoning_selection(
            &headers,
            &serde_json::json!({"thinking": {"budget_tokens": 12000}}),
        );

        assert_eq!(low.policy, LocalReasoningPolicy::Low);
        assert_eq!(medium.policy, LocalReasoningPolicy::Medium);
        assert_eq!(high.policy, LocalReasoningPolicy::High);
        assert_eq!(high.source, "client_thinking_budget");
    }

    #[test]
    fn local_reasoning_defaults_preserve_explicit_sampling() {
        let mut req = serde_json::json!({
            "max_tokens": 333,
            "temperature": 0.7
        });

        apply_local_reasoning_defaults(
            &mut req,
            LocalReasoningSelection {
                policy: LocalReasoningPolicy::Low,
                source: "test",
            },
        );

        assert_eq!(req["max_tokens"], serde_json::json!(333));
        assert_eq!(req["temperature"], serde_json::json!(0.7));
    }

    #[test]
    fn local_reasoning_defaults_shape_missing_sampling() {
        let mut req = serde_json::json!({});

        apply_local_reasoning_defaults(
            &mut req,
            LocalReasoningSelection {
                policy: LocalReasoningPolicy::Low,
                source: "test",
            },
        );

        assert_eq!(req["max_tokens"], serde_json::json!(2048));
        assert_eq!(req["temperature"], serde_json::json!(0.1));
    }

    #[test]
    fn extract_sse_handles_missing_content_field_gracefully() {
        let sse = "data: {\"choices\":[{\"delta\":{}}]}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"real\"}}]}\n\
                   data: [DONE]\n";
        assert_eq!(extract_assistant_from_sse(sse), "real");
    }

    #[test]
    fn retry_backoff_durations_are_correct() {
        let delays: Vec<u64> = (0u32..2).map(|a| 200 * 2u64.pow(a)).collect();
        assert_eq!(delays, vec![200, 400]);
    }

    #[test]
    fn rate_limited_response_sets_429_and_retry_after() {
        let response = rate_limited_response("secret-token", 3);
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response
                .headers()
                .get(header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok()),
            Some("3")
        );
    }

    // ── API_KEYS parsing: semicolon-delimited token,namespace entries ──

    #[test]
    fn parse_simple_single_entry() {
        let input = "agent-os,project-alpha";
        let mut parts = input.splitn(2, ',');
        let token = parts.next().unwrap_or(input).trim().to_string();
        let namespace = parts.next().unwrap_or(&token).trim().to_string();
        assert_eq!(token, "agent-os");
        assert_eq!(namespace, "project-alpha");
    }

    #[test]
    fn parse_multiple_semicolon_entries() {
        let input = "agent-os,project-alpha;agent-os,project-beta;sk-work,work";
        let entries: Vec<(String, String)> = input
            .split(';')
            .map(|s| {
                let s = s.trim();
                let mut parts = s.splitn(2, ',');
                let token = parts.next().unwrap_or(s).trim().to_string();
                let namespace = parts.next().unwrap_or(&token).trim().to_string();
                (token, namespace)
            })
            .filter(|(t, _)| !t.is_empty())
            .collect();

        assert_eq!(entries.len(), 3);
        assert_eq!(
            entries[0],
            ("agent-os".to_string(), "project-alpha".to_string())
        );
        assert_eq!(
            entries[1],
            ("agent-os".to_string(), "project-beta".to_string())
        );
        assert_eq!(entries[2], ("sk-work".to_string(), "work".to_string()));
    }

    #[test]
    fn parse_empty_token_filtered() {
        let input = "token1,ns1;;token2,ns2";
        let entries: Vec<(String, String)> = input
            .split(';')
            .map(|s| {
                let s = s.trim();
                let mut parts = s.splitn(2, ',');
                let token = parts.next().unwrap_or(s).trim().to_string();
                let namespace = parts.next().unwrap_or(&token).trim().to_string();
                (token, namespace)
            })
            .filter(|(t, _)| !t.is_empty())
            .collect();

        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn parse_fallback_to_token_when_no_namespace() {
        let input = "my-token";
        let mut parts = input.splitn(2, ',');
        let token = parts.next().unwrap_or(input).trim().to_string();
        let namespace = parts.next().unwrap_or(&token).trim().to_string();
        assert_eq!(token, "my-token");
        assert_eq!(namespace, "my-token");
    }

    #[test]
    fn parse_default_value() {
        let input = "agent-os,agentic-os";
        let entries: Vec<(String, String)> = input
            .split(';')
            .map(|s| {
                let s = s.trim();
                let mut parts = s.splitn(2, ',');
                let token = parts.next().unwrap_or(s).trim().to_string();
                let namespace = parts.next().unwrap_or(&token).trim().to_string();
                (token, namespace)
            })
            .filter(|(t, _)| !t.is_empty())
            .collect();

        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0],
            ("agent-os".to_string(), "agentic-os".to_string())
        );
    }

    // ── Model substitution ────────────────────────────────────────

    #[test]
    fn model_substituted_with_default_in_chat_completions() {
        let mut req = json!({
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let default_model = "qwen36-35b-heretic";
        req["model"] = Value::String(default_model.to_string());
        assert_eq!(req["model"].as_str().unwrap(), default_model);
    }

    #[test]
    fn model_substituted_with_default_for_any_client_model_name() {
        let client_models = [
            "gpt-4o",
            "claude-sonnet-4-6",
            "claude-opus-4-7",
            "gpt-4-turbo",
        ];
        let default_model = "qwen36-35b-heretic";
        for client_model in client_models {
            let mut req = json!({
                "model": client_model,
                "messages": [{"role": "user", "content": "hi"}]
            });
            req["model"] = Value::String(default_model.to_string());
            assert_eq!(
                req["model"].as_str().unwrap(),
                default_model,
                "client model '{client_model}' was not replaced"
            );
        }
    }

    #[test]
    fn anthropic_sanitize_removes_local_backend_incompatible_params() {
        let mut req = json!({
            "model": "qwen36-27b",
            "max_tokens": 1024,
            "max_output_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "reasoning_effort": "high",
            "context_management": {"edits": []},
            "messages": [{"role": "user", "content": "hi"}]
        });

        anthropic::sanitize_litellm_request(&mut req);

        assert!(req.get("thinking").is_none());
        assert!(req.get("max_output_tokens").is_none());
        assert!(req.get("reasoning_effort").is_none());
        assert!(req.get("context_management").is_none());
        assert_eq!(req["max_tokens"], json!(1024));
        assert_eq!(req["messages"][0]["content"], "hi");
    }

    #[test]
    fn max_tokens_clamp_respects_explicit_small_values() {
        let mut req = json!({
            "model": "qwen36-27b",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}]
        });

        enforce_min_max_tokens(&mut req);

        assert_eq!(req["max_tokens"], json!(1));
    }

    #[test]
    fn max_tokens_clamp_defaults_missing_values() {
        let mut req = json!({
            "model": "qwen36-27b",
            "messages": [{"role": "user", "content": "hi"}]
        });

        enforce_min_max_tokens(&mut req);

        assert_eq!(req["max_tokens"], json!(DEFAULT_MAX_TOKENS));
    }

    #[test]
    fn max_tokens_clamp_caps_oversized_values() {
        let mut req = json!({
            "model": "qwen36-27b",
            "max_tokens": MAX_MAX_TOKENS + 1,
            "messages": [{"role": "user", "content": "hi"}]
        });

        enforce_min_max_tokens(&mut req);

        assert_eq!(req["max_tokens"], json!(MAX_MAX_TOKENS));
    }

    #[test]
    fn metrics_include_context_pack_counts() {
        let metrics = telemetry::MetricsRegistry::new();
        telemetry::record_context_pack(
            &metrics,
            &ContextPackStats {
                context_chars: 400,
                context_tokens_estimate: 100,
                l0_items_injected: 1,
                l1_items_injected: 2,
                l2_items_injected: 3,
                l3_items_injected: 4,
                failed_attempts_injected: 1,
                remediations_injected: 1,
                retrieval_semantic_hits: 5,
                retrieval_fts_hits: 6,
                retrieval_deduped_hits: 7,
                cache_hit: false,
                ..Default::default()
            },
        );
        telemetry::record_tokens(
            &metrics,
            &TokenUsage {
                processed_tokens: 120,
                cached_tokens: 80,
                generated_tokens: 40,
            },
            "test-model",
        );
        telemetry::record_promotion(&metrics, false, false);
        telemetry::record_promotion(&metrics, true, true);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.context_pack_requests, 1);
        assert_eq!(snapshot.context_cache_misses, 1);
        assert_eq!(snapshot.l3_items_injected, 4);
        assert_eq!(snapshot.retrieval_deduped_hits, 7);
        assert_eq!(snapshot.processed_tokens, 120);
        assert_eq!(snapshot.cached_tokens, 80);
        assert_eq!(snapshot.generated_tokens, 40);
        assert_eq!(snapshot.promotion_attempts, 2);
        assert_eq!(snapshot.promotion_accepted, 1);
        assert_eq!(snapshot.promotion_rejected, 1);
        assert_eq!(snapshot.memory_source_coverage, 0.5);
    }

    #[test]
    fn context_pack_sections_are_stable() {
        let mut evidence = db::ContextEvidence::default();
        evidence.l3_project = vec![test_event("summary", "L3 architecture truth", 3)];
        evidence.l2_repo = vec![test_event("summary", "L2 repo pattern", 2)];
        evidence.l1_matching = vec![test_event("summary", "L1 task summary", 1)];
        evidence.l0_recent = vec![test_event("edit", "L0 raw evidence", 0)];
        evidence.failures = vec![test_event("failed_attempt", "do not repeat this", 0)];
        let policy = ContextPolicy::for_category(TaskCategory::Architecture);

        let (context, _stats) = db::build_layered_context(
            "repo",
            "architecture task",
            &evidence,
            &[],
            &[],
            &policy,
            16_000,
            4000,
        );

        let l3 = context.find("== Durable Project Memory ==").unwrap();
        let l2 = context.find("== Repo Patterns and Decisions ==").unwrap();
        let l1 = context.find("== Relevant Session Summaries ==").unwrap();
        let l0 = context.find("== Recent Evidence ==").unwrap();
        let failures = context
            .find("== Failed Attempts and Remediations ==")
            .unwrap();

        assert!(l3 < l2);
        assert!(l2 < l1);
        assert!(l1 < l0);
        assert!(l0 < failures);
    }

    #[test]
    fn failed_attempts_are_injected_for_debug_tasks() {
        let mut evidence = db::ContextEvidence::default();
        evidence.failures = vec![db::AgentEvent {
            evidence: Some("the old retry loop timed out".to_string()),
            metadata: json!({"outcome": "bounded retry fixed the lag"}),
            ..test_event("failed_attempt", "unbounded retries caused lag", 0)
        }];
        let policy = ContextPolicy::for_category(TaskCategory::Narrow);

        let (context, stats) = db::build_layered_context(
            "repo",
            "debug lag",
            &evidence,
            &[],
            &[],
            &policy,
            8000,
            4000,
        );

        assert!(context.contains("unbounded retries caused lag"));
        assert!(context.contains("Evidence: the old retry loop timed out"));
        assert!(context.contains("Outcome: bounded retry fixed the lag"));
        assert_eq!(stats.failed_attempts_injected, 1);
    }

    #[test]
    fn cache_hit_rate_updates() {
        let metrics = telemetry::MetricsRegistry::new();
        telemetry::record_context_pack(
            &metrics,
            &ContextPackStats {
                cache_hit: false,
                ..Default::default()
            },
        );
        telemetry::record_context_pack(
            &metrics,
            &ContextPackStats {
                cache_hit: true,
                ..Default::default()
            },
        );

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.context_pack_requests, 2);
        assert_eq!(snapshot.context_cache_misses, 1);
        assert_eq!(snapshot.context_cache_hits, 1);
    }

    #[test]
    fn failure_history_and_validation_capture_are_feature_flagged() {
        let src = include_str!("handlers.rs");
        let context_src = include_str!("context_packing.rs");
        assert!(context_src.contains("state.execution_feedback_enabled"));
        let ctx_start = context_src
            .find("async fn get_or_build_cached_context")
            .expect("get_or_build_cached_context not found");
        let ctx_body: String = context_src[ctx_start..].chars().take(4500).collect();
        assert!(ctx_body.contains("state.execution_feedback_enabled"));

        let validation_start = src
            .find("pub async fn validations")
            .expect("validations handler not found");
        let validation_body: String = src[validation_start..].chars().take(1200).collect();
        assert!(validation_body.contains("!state.execution_feedback_enabled"));
    }

    #[test]
    fn trajectory_capture_is_feature_flagged() {
        let src = include_str!("handlers.rs");
        assert!(src.contains("state.trajectory_capture_enabled"));
        let append_start = src
            .find("pub async fn append_event")
            .expect("append_event handler not found");
        let append_body: String = src[append_start..].chars().take(1400).collect();
        assert!(append_body.contains("req.trajectory_id = None"));
        assert!(append_body.contains("req.attempt_index = None"));
        assert!(append_body.contains("req.event_role = None"));

        let sweep_start = src
            .find("pub async fn run_trajectory_idle_sweep")
            .expect("trajectory idle sweep not found");
        let sweep_body: String = src[sweep_start..].chars().take(800).collect();
        assert!(sweep_body.contains("!state.trajectory_capture_enabled"));
    }

    /// Verify the model request path does not await a full context rebuild.
    #[test]
    fn pack_context_into_req_uses_async_cache_refresh() {
        let src = include_str!("handlers.rs");
        let pctr_start = src
            .find("async fn pack_context_into_req")
            .expect("pack_context_into_req not found");
        let body = &src[pctr_start..pctr_start + 1500];
        assert!(
            body.contains("cached_context_for_request"),
            "pack_context_into_req should use cached/minimal context immediately"
        );
        assert!(
            !body.contains("get_or_build_cached_context("),
            "pack_context_into_req should not await full context construction"
        );
        assert!(
            include_str!("context_packing.rs").contains("fn spawn_context_cache_refresh")
                && include_str!("context_packing.rs").contains("get_or_build_cached_context_inner")
                && include_str!("background/mod.rs").contains("tokio::spawn(async move"),
            "context cache refresh should run in the background"
        );
    }

    // ── inject_system_context_anthropic ───────────────────────────────

    #[test]
    fn anthropic_inject_sets_system_when_absent() {
        let mut payload = json!({"messages": [{"role": "user", "content": "hi"}]});
        inject_system_context_anthropic(&mut payload, "ctx");
        assert_eq!(
            payload["system"],
            json!([{
                "type": "text",
                "text": "ctx"
            }])
        );
    }

    #[test]
    fn anthropic_inject_appends_to_string_system() {
        let mut payload = json!({"system": "base", "messages": []});
        inject_system_context_anthropic(&mut payload, "ctx");
        let sys = payload["system"].as_array().unwrap();
        assert_eq!(sys[0], json!({"type": "text", "text": "base"}));
        assert_eq!(
            sys[1],
            json!({
                "type": "text",
                "text": "ctx"
            })
        );
    }

    #[test]
    fn anthropic_inject_flattens_array_system_and_appends() {
        let mut payload = json!({
            "system": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}],
            "messages": []
        });
        inject_system_context_anthropic(&mut payload, "ctx");
        let sys = payload["system"].as_array().unwrap();
        assert_eq!(sys[0], json!({"type": "text", "text": "part1"}));
        assert_eq!(sys[1], json!({"type": "text", "text": "part2"}));
        assert_eq!(
            sys[2],
            json!({
                "type": "text",
                "text": "ctx"
            })
        );
    }

    #[test]
    fn anthropic_inject_ignores_non_text_system_type() {
        let mut payload = json!({"system": 42, "messages": []});
        inject_system_context_anthropic(&mut payload, "ctx");
        assert_eq!(
            payload["system"],
            json!([{
                "type": "text",
                "text": "ctx"
            }])
        );
    }

    #[test]
    fn anthropic_inject_caches_stable_prefix_not_dynamic_tail() {
        let mut payload = json!({"messages": []});
        let context =
            "== Stable Context Artifacts ==\n[repo:service_topology:active]\nstable\n\nRepository: repo\nTask: task\n";
        inject_system_context_anthropic(&mut payload, context);
        let sys = payload["system"].as_array().unwrap();
        assert_eq!(sys.len(), 2);
        assert_eq!(
            sys[0]["cache_control"],
            json!({"type": "ephemeral"}),
            "stable compiler prefix should be provider-cacheable"
        );
        assert!(sys[0]["text"]
            .as_str()
            .unwrap()
            .contains("Stable Context Artifacts"));
        assert_eq!(
            sys[1].get("cache_control"),
            None,
            "dynamic repository/task tail must not become a provider cache breakpoint"
        );
        assert!(sys[1]["text"]
            .as_str()
            .unwrap()
            .starts_with("Repository: repo"));
    }

    // ── extract_assistant_from_anthropic_response ─────────────────────

    #[test]
    fn anthropic_response_extracts_text_block() {
        let resp = json!({"content": [{"type": "text", "text": "hello"}]});
        assert_eq!(extract_assistant_from_anthropic_response(&resp), "hello");
    }

    #[test]
    fn anthropic_response_skips_tool_use_blocks() {
        let resp = json!({"content": [
            {"type": "tool_use", "id": "c1", "name": "bash", "input": {}},
            {"type": "text", "text": "done"}
        ]});
        assert_eq!(extract_assistant_from_anthropic_response(&resp), "done");
    }

    #[test]
    fn anthropic_response_concatenates_multiple_text_blocks() {
        let resp = json!({"content": [
            {"type": "text", "text": "foo"},
            {"type": "text", "text": "bar"}
        ]});
        assert_eq!(extract_assistant_from_anthropic_response(&resp), "foobar");
    }

    #[test]
    fn anthropic_response_returns_empty_when_no_content() {
        assert_eq!(extract_assistant_from_anthropic_response(&json!({})), "");
    }

    #[test]
    fn anthropic_response_truncates_at_500_chars() {
        let long = "x".repeat(600);
        let resp = json!({"content": [{"type": "text", "text": long}]});
        assert_eq!(extract_assistant_from_anthropic_response(&resp).len(), 500);
    }

    // ── extract_token_usage_from_anthropic_sse ────────────────────────

    #[test]
    fn anthropic_sse_usage_reads_message_start_and_delta() {
        let raw = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":100,\"cache_read_input_tokens\":40}}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":25}}\n\n",
        );
        let usage = extract_token_usage_from_anthropic_sse(raw);
        assert_eq!(usage.processed_tokens, 100);
        assert_eq!(usage.cached_tokens, 40);
        assert_eq!(usage.generated_tokens, 25);
    }

    #[test]
    fn anthropic_sse_usage_is_zero_for_empty_stream() {
        let usage = extract_token_usage_from_anthropic_sse("");
        assert_eq!(usage.processed_tokens, 0);
        assert_eq!(usage.generated_tokens, 0);
    }

    // ── extract_assistant_from_anthropic_sse ──────────────────────────

    #[test]
    fn anthropic_sse_collects_text_deltas() {
        let raw = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
        );
        assert_eq!(extract_assistant_from_anthropic_sse(raw), "hello");
    }

    #[test]
    fn anthropic_sse_skips_non_text_deltas() {
        let raw = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n",
        );
        assert_eq!(extract_assistant_from_anthropic_sse(raw), "");
    }

    #[test]
    fn anthropic_sse_truncates_at_500_chars() {
        let chunk_text = "x".repeat(300);
        let make_line = |t: &str| {
            format!(
                "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{t}\"}}}}\n\n"
            )
        };
        let raw = make_line(&chunk_text) + &make_line(&chunk_text);
        assert_eq!(extract_assistant_from_anthropic_sse(&raw).len(), 500);
    }

    fn test_event(event_type: &str, summary: &str, summary_level: i32) -> db::AgentEvent {
        db::AgentEvent {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: "session".to_string(),
            repo: "repo".to_string(),
            actor: "actor".to_string(),
            event_type: event_type.to_string(),
            summary: summary.to_string(),
            evidence: None,
            metadata: json!({}),
            correlation_id: None,
            parent_event_id: None,
            trajectory_id: None,
            attempt_index: None,
            event_role: None,
            created_at: Utc::now(),
            summary_level,
        }
    }
}
