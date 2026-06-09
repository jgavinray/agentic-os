use axum::response::{IntoResponse, Response};
use serde_json::Value;
use std::sync::Arc;
use uuid::Uuid;

use crate::client_capture::RawHttpCapture;
use crate::handlers_capture::record_json_success_capture;
use crate::handlers_openai_persistence::{
    persist_openai_exchange, persist_openai_trajectory_response,
};
use crate::handlers_usage::record_success_usage;
use crate::proxy_support::dispatch_non_streaming_raw;
use crate::state::{AppState, TokenUsage};
use crate::trajectory::TrajectoryContext;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn handle_openai_non_streaming(
    state: &Arc<AppState>,
    req: Value,
    repo: String,
    task: String,
    requested_model: String,
    namespace: String,
    correlation_id: Option<Uuid>,
    request_metadata: Option<Value>,
    session_id: Option<String>,
    trajectory: Option<TrajectoryContext>,
    request_event_id: Option<Uuid>,
    context_pack_id: Option<Uuid>,
    user_content: String,
    finalizer: crate::litellm::LiteLlmCallFinalizer,
    vllm_cache_before: Option<(String, crate::vllm_metrics::VllmCacheSnapshot)>,
    capture: RawHttpCapture,
) -> Response {
    match dispatch_non_streaming_raw(state, &req, &finalizer).await {
        Ok((val, latency_ms)) => {
            record_json_success_capture(state, capture, &val).await;
            let usage = TokenUsage::from_openai_value(&val);
            let provider_cache = crate::litellm::ProviderCacheCounters::from_value(&val);
            crate::vllm_metrics::record_cache_observation(
                state,
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
            record_success_usage(state, &usage, &requested_model, &namespace, &repo);

            let assistant_content: String = val["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .chars()
                .take(500)
                .collect();
            let tool_results = crate::execution_feedback::tool_results_from_value(&val);

            if let (Some(session_id), Some(trajectory)) = (session_id.as_deref(), trajectory) {
                persist_openai_trajectory_response(
                    state,
                    session_id,
                    &repo,
                    &task,
                    &assistant_content,
                    &val,
                    latency_ms,
                    request_metadata.clone(),
                    context_pack_id,
                    trajectory,
                    request_event_id,
                    tool_results,
                )
                .await;
            } else {
                persist_openai_exchange(
                    state,
                    &repo,
                    &task,
                    &user_content,
                    &assistant_content,
                    correlation_id,
                    request_metadata.clone(),
                    tool_results,
                )
                .await;
            }
            axum::Json(val).into_response()
        }
        Err(resp) => resp,
    }
}
