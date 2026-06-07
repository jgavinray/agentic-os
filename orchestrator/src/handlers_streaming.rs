use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;

use crate::anthropic;
use crate::handlers_stream_persistence::{
    persist_stream_completion, StreamCompletionPersistence, StreamResponseFormat,
};
use crate::sse::{extract_token_usage_from_anthropic_sse, extract_token_usage_from_sse};
use crate::state::AppState;
use crate::telemetry;
use crate::token_limits::{context_window_retry_max_tokens, set_max_tokens};

#[allow(clippy::too_many_arguments)]
pub(crate) async fn handle_streaming(
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
            persist_stream_completion(StreamCompletionPersistence {
                state: state_bg,
                raw_bytes,
                capture,
                response_format: StreamResponseFormat::ChatCompletions,
                requested_model,
                namespace,
                repo,
                task,
                user_content,
                correlation_id,
                request_metadata,
                session_id,
                trajectory,
                request_event_id,
                context_pack_id,
                started,
            })
            .await;
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

#[allow(clippy::too_many_arguments)]
pub(crate) async fn handle_streaming_anthropic(
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
            persist_stream_completion(StreamCompletionPersistence {
                state: state_bg,
                raw_bytes,
                capture,
                response_format: StreamResponseFormat::AnthropicMessages,
                requested_model: model,
                namespace,
                repo,
                task,
                user_content,
                correlation_id,
                request_metadata,
                session_id,
                trajectory,
                request_event_id,
                context_pack_id,
                started,
            })
            .await;
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
