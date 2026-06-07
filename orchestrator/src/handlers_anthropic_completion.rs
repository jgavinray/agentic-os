use axum::response::{IntoResponse, Response};
use serde_json::Value;
use std::sync::Arc;
use uuid::Uuid;

use crate::client_capture::RawHttpCapture;
use crate::event_capture::{
    capture_tool_results_background, persist_exchange_with_correlation,
    persist_model_response_event,
};
use crate::handlers_anthropic_dispatch::dispatch_anthropic_messages_with_retry;
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

#[allow(clippy::too_many_arguments)]
async fn persist_anthropic_trajectory_response(
    state: &Arc<AppState>,
    session_id: &str,
    repo: &str,
    task: &str,
    assistant_content: &str,
    val: &Value,
    latency_ms: i64,
    request_metadata: Option<Value>,
    context_pack_id: Option<Uuid>,
    trajectory: TrajectoryContext,
    request_event_id: Option<Uuid>,
    tool_results: Vec<crate::execution_feedback::CapturedToolResult>,
) {
    let (input_tokens, output_tokens) = crate::trajectory::optional_token_counts_from_value(val);
    let metadata = crate::trajectory::model_response_metadata(
        &state.default_model,
        "litellm",
        input_tokens,
        output_tokens,
        Some(latency_ms),
        crate::trajectory::model_finish_reason(val),
        request_metadata,
        context_pack_id,
        Some(trajectory),
    );
    let assistant_event_id = persist_model_response_event(
        state,
        session_id,
        repo,
        assistant_content,
        metadata,
        trajectory,
        request_event_id,
    )
    .await;
    capture_tool_results_background(
        state.as_ref().clone(),
        session_id.to_string(),
        repo.to_string(),
        task.to_string(),
        trajectory.trajectory_id,
        assistant_event_id,
        Some(trajectory),
        tool_results,
    );
}

async fn persist_anthropic_exchange(
    state: &Arc<AppState>,
    repo: &str,
    task: &str,
    user_content: &str,
    assistant_content: &str,
    correlation_id: Option<Uuid>,
    tool_results: Vec<crate::execution_feedback::CapturedToolResult>,
) {
    match crate::db::find_or_create_session(&state.pool, repo, task, "agent").await {
        Ok(session_id) => {
            let assistant_event_id = persist_exchange_with_correlation(
                state,
                &session_id,
                repo,
                user_content,
                assistant_content,
                correlation_id,
                None,
            )
            .await;
            if let Some(correlation_id) = correlation_id {
                capture_tool_results_background(
                    state.as_ref().clone(),
                    session_id,
                    repo.to_string(),
                    task.to_string(),
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
