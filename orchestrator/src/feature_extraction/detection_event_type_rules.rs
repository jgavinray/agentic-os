use crate::feature_extraction_metadata::{
    context_pack_empty, context_pack_truncated, event_input_tokens_from_metadata,
    event_latency_ms_from_metadata, string_path, trajectory_abandoned_before_model,
    trajectory_single_model_abandoned_no_tools, HIGH_INPUT_TOKEN_THRESHOLD,
    SLOW_UPSTREAM_MODEL_MS_THRESHOLD,
};
use crate::feature_extraction_types::DetectionTag;
use serde_json::Value;

pub(crate) fn append_context_pack_tags(
    tags: &mut Vec<DetectionTag>,
    metadata: &Value,
    source_override: Option<&str>,
) {
    if context_pack_empty(metadata) {
        tags.push(DetectionTag::new(
            "context_pack_empty",
            source_override.unwrap_or("context_builder"),
        ));
    }
    if context_pack_truncated(metadata) {
        tags.push(DetectionTag::new(
            "context_pack_truncated",
            source_override.unwrap_or("context_builder"),
        ));
    }
}

pub(crate) fn append_assistant_message_tags(
    tags: &mut Vec<DetectionTag>,
    summary: &str,
    metadata: &Value,
    source_override: Option<&str>,
) {
    if event_input_tokens_from_metadata(metadata)
        .is_some_and(|tokens| tokens >= HIGH_INPUT_TOKEN_THRESHOLD)
    {
        tags.push(DetectionTag::new(
            "high_input_tokens",
            source_override.unwrap_or("model_response_parser"),
        ));
    }
    if event_latency_ms_from_metadata(metadata)
        .is_some_and(|latency_ms| latency_ms >= SLOW_UPSTREAM_MODEL_MS_THRESHOLD)
    {
        tags.push(DetectionTag::new(
            "slow_upstream_model",
            source_override.unwrap_or("model_response_parser"),
        ));
    }
    if string_path(metadata, &["finish_reason"])
        .or_else(|| string_path(metadata, &["payload", "finish_reason"]))
        == Some("tool_use")
        && summary.trim().is_empty()
    {
        tags.push(DetectionTag::new(
            "empty_tool_use_message",
            source_override.unwrap_or("model_response_parser"),
        ));
    }
}

pub(crate) fn append_trajectory_result_tags(
    tags: &mut Vec<DetectionTag>,
    metadata: &Value,
    source_override: Option<&str>,
) {
    if event_input_tokens_from_metadata(metadata)
        .is_some_and(|tokens| tokens >= HIGH_INPUT_TOKEN_THRESHOLD)
    {
        tags.push(DetectionTag::new(
            "high_input_tokens",
            source_override.unwrap_or("trajectory_analyzer"),
        ));
    }
    if event_latency_ms_from_metadata(metadata)
        .is_some_and(|latency_ms| latency_ms >= SLOW_UPSTREAM_MODEL_MS_THRESHOLD)
    {
        tags.push(DetectionTag::new(
            "slow_upstream_model",
            source_override.unwrap_or("trajectory_analyzer"),
        ));
    }
    if trajectory_abandoned_before_model(metadata) {
        tags.push(DetectionTag::new(
            "abandoned_before_model",
            source_override.unwrap_or("trajectory_analyzer"),
        ));
    }
    if trajectory_single_model_abandoned_no_tools(metadata) {
        tags.push(DetectionTag::new(
            "single_model_abandoned_no_tools",
            source_override.unwrap_or("trajectory_analyzer"),
        ));
    }
}
