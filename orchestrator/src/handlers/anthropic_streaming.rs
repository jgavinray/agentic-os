use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;

use crate::handlers_anthropic_upstream::dispatch_anthropic_streaming_with_retry;
use crate::handlers_stream_persistence::{
    persist_stream_completion, StreamCompletionPersistence, StreamResponseFormat,
};
use crate::sse::extract_token_usage_from_anthropic_sse;
use crate::state::AppState;
use crate::telemetry;

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
    let started = finalizer.attempt_mut().started_at;
    let dispatch = match dispatch_anthropic_streaming_with_retry(
        state,
        req,
        finalizer,
        request_event_id,
        trajectory,
        context_pack_id,
        &namespace,
        &repo,
        &task,
        &model,
    )
    .await
    {
        Ok(dispatch) => dispatch,
        Err(response) => return response,
    };
    let upstream = dispatch.response;
    let finalizer = dispatch.finalizer;

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
