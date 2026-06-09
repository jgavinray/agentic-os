use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;

use crate::handlers_openai_upstream::dispatch_openai_streaming;
use crate::handlers_stream_persistence::{
    persist_stream_completion, StreamCompletionPersistence, StreamResponseFormat,
};
use crate::sse::extract_token_usage_from_sse;
use crate::state::AppState;
use crate::telemetry;

pub(crate) use crate::handlers_anthropic_streaming::handle_streaming_anthropic;

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
    finalizer: crate::litellm::LiteLlmCallFinalizer,
    vllm_cache_before: Option<(String, crate::vllm_metrics::VllmCacheSnapshot)>,
    capture: crate::client_capture::RawHttpCapture,
) -> Response {
    let dispatch = match dispatch_openai_streaming(state, req, finalizer).await {
        Ok(dispatch) => dispatch,
        Err(response) => return response,
    };
    let upstream = dispatch.response;
    let finalizer = dispatch.finalizer;
    let started = dispatch.started;

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
