use axum::response::{IntoResponse, Response};
use serde_json::Value;
use std::sync::Arc;
use uuid::Uuid;

use crate::client_capture::RawHttpCapture;
use crate::handlers_anthropic_dispatch::dispatch_anthropic_messages_with_retry;
use crate::handlers_anthropic_persistence::{
    persist_anthropic_exchange, persist_anthropic_trajectory_response,
};
use crate::handlers_capture::record_json_success_capture;
use crate::handlers_usage::record_success_usage;
use crate::litellm::{LiteLlmCallFinalizer, RouteSelection};
use crate::sse::extract_assistant_from_anthropic_response;
use crate::state::{AppState, TokenUsage};
use crate::trajectory::TrajectoryContext;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn handle_anthropic_non_streaming(
    state: &Arc<AppState>,
    mut req: Value,
    repo: String,
    task: String,
    model: String,
    namespace: String,
    correlation_id: Option<Uuid>,
    request_metadata: Option<Value>,
    session_id: Option<String>,
    trajectory: Option<TrajectoryContext>,
    request_event_id: Option<Uuid>,
    context_pack_id: Option<Uuid>,
    user_content: String,
    finalizer: LiteLlmCallFinalizer,
    vllm_cache_before: Option<(String, crate::vllm_metrics::VllmCacheSnapshot)>,
    capture: RawHttpCapture,
    route: RouteSelection,
    context_pack_hash: Option<String>,
    baseline_arm: Option<String>,
) -> Response {
    let dispatch = match dispatch_anthropic_messages_with_retry(
        state,
        &mut req,
        finalizer,
        request_event_id,
        trajectory,
        context_pack_id,
        &namespace,
        &repo,
        &task,
        &model,
        &route,
        context_pack_hash,
        baseline_arm,
    )
    .await
    {
        Ok(dispatch) => dispatch,
        Err(response) => return response,
    };
    let mut val = dispatch.value;
    let finalizer = dispatch.finalizer;

    let usage = TokenUsage::from_openai_value(&val);
    let provider_cache = crate::litellm::ProviderCacheCounters::from_value(&val);
    let vllm_delta = crate::vllm_metrics::record_cache_observation(
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
    let provider_cache =
        crate::vllm_metrics::merge_provider_cache_from_delta(provider_cache, vllm_delta);
    crate::vllm_metrics::inject_anthropic_cache_usage(&mut val, provider_cache);
    record_json_success_capture(state, capture, &val).await;
    record_success_usage(state, &usage, &model, &namespace, &repo);

    let assistant_content = extract_assistant_from_anthropic_response(&val);
    let tool_results = crate::execution_feedback::tool_results_from_value(&val);
    if let (Some(session_id), Some(trajectory)) = (session_id.as_deref(), trajectory) {
        persist_anthropic_trajectory_response(
            state,
            session_id,
            &repo,
            &task,
            &assistant_content,
            &val,
            dispatch.latency_ms,
            request_metadata,
            context_pack_id,
            trajectory,
            request_event_id,
            tool_results,
        )
        .await;
    } else {
        persist_anthropic_exchange(
            state,
            &repo,
            &task,
            &user_content,
            &assistant_content,
            correlation_id,
            tool_results,
        )
        .await;
    }

    axum::Json(val).into_response()
}
