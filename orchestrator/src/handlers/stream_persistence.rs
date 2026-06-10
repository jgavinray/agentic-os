use crate::db;
use crate::event_capture::{
    capture_tool_results_background, persist_exchange_with_correlation,
    persist_model_response_event,
};
use crate::execution_feedback::CapturedToolResult;
pub(crate) use crate::handlers_stream_persistence_types::{
    StreamCompletionPersistence, StreamResponseFormat,
};
use crate::sse::{
    extract_assistant_from_anthropic_sse, extract_assistant_from_sse,
    extract_token_usage_from_anthropic_sse, extract_token_usage_from_sse,
    optional_token_usage_from_sse,
};
use crate::telemetry;

pub(crate) async fn persist_stream_completion(completion: StreamCompletionPersistence) {
    let mut capture = completion.capture.clone();
    capture.response_status = Some(axum::http::StatusCode::OK.as_u16() as i32);
    capture.response_headers = Some(serde_json::json!({"content-type": ["text/event-stream"]}));
    capture.raw_response_body = Some(completion.raw_bytes.clone());
    crate::client_capture::record_best_effort(completion.state.capture_pool.as_ref(), capture)
        .await;

    let raw = String::from_utf8_lossy(&completion.raw_bytes);
    let usage = match completion.response_format {
        StreamResponseFormat::ChatCompletions => extract_token_usage_from_sse(&raw),
        StreamResponseFormat::AnthropicMessages => extract_token_usage_from_anthropic_sse(&raw),
    };
    telemetry::record_tokens(
        &completion.state.metrics,
        &usage,
        &completion.state.default_model,
    );
    if !usage.is_empty() {
        if let Err(e) = db::record_token_usage(
            &completion.state.pool,
            &completion.requested_model,
            &completion.state.default_model,
            &completion.namespace,
            &completion.repo,
            &usage,
        )
        .await
        {
            tracing::warn!("failed to record token usage: {e}");
        }
    }

    let assistant_content = match completion.response_format {
        StreamResponseFormat::ChatCompletions => extract_assistant_from_sse(&raw),
        StreamResponseFormat::AnthropicMessages => extract_assistant_from_anthropic_sse(&raw),
    };
    let tool_results = crate::execution_feedback::tool_results_from_sse(&raw);
    if let (Some(session_id), Some(trajectory)) =
        (completion.session_id.as_deref(), completion.trajectory)
    {
        persist_stream_trajectory_response(
            &completion,
            session_id,
            trajectory,
            &raw,
            &assistant_content,
            tool_results,
        )
        .await;
    } else {
        persist_stream_exchange(&completion, &assistant_content, tool_results).await;
    }
}

async fn persist_stream_trajectory_response(
    completion: &StreamCompletionPersistence,
    session_id: &str,
    trajectory: crate::trajectory::TrajectoryContext,
    raw: &str,
    assistant_content: &str,
    tool_results: Vec<CapturedToolResult>,
) {
    let (input_tokens, output_tokens) = optional_token_usage_from_sse(raw);
    let metadata = crate::trajectory::model_response_metadata(
        &completion.state.default_model,
        "litellm",
        input_tokens,
        output_tokens,
        Some(completion.started.elapsed().as_millis() as i64),
        crate::sse::optional_stop_reason_from_sse(raw),
        completion.request_metadata.clone(),
        completion.context_pack_id,
        Some(trajectory),
    );
    let assistant_event_id = persist_model_response_event(
        &completion.state,
        session_id,
        &completion.repo,
        assistant_content,
        metadata,
        trajectory,
        completion.request_event_id,
    )
    .await;
    capture_tool_results_background(
        completion.state.clone(),
        session_id.to_string(),
        completion.repo.clone(),
        completion.task.clone(),
        trajectory.trajectory_id,
        assistant_event_id,
        Some(trajectory),
        tool_results,
    );
}

async fn persist_stream_exchange(
    completion: &StreamCompletionPersistence,
    assistant_content: &str,
    tool_results: Vec<CapturedToolResult>,
) {
    match db::find_or_create_session(
        &completion.state.pool,
        &completion.repo,
        &completion.task,
        "agent",
    )
    .await
    {
        Ok(sid) => {
            let request_metadata = match completion.response_format {
                StreamResponseFormat::ChatCompletions => completion.request_metadata.clone(),
                StreamResponseFormat::AnthropicMessages => None,
            };
            let assistant_event_id = persist_exchange_with_correlation(
                &completion.state,
                &sid,
                &completion.repo,
                &completion.user_content,
                assistant_content,
                completion.correlation_id,
                request_metadata,
            )
            .await;
            if let Some(correlation_id) = completion.correlation_id {
                capture_tool_results_background(
                    completion.state.clone(),
                    sid,
                    completion.repo.clone(),
                    completion.task.clone(),
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
