use crate::feature_detection_bootstrap::bootstrap_text_tags;
use crate::feature_detection_producer_signals::explicit_producer_signal_tags;
use crate::feature_detection_tag_merge::dedupe_tags_within_producer;
use crate::feature_extraction_metadata::{
    bool_path, context_pack_empty, context_pack_truncated, event_input_tokens_from_metadata,
    event_latency_ms_from_metadata, string_path, summarizer_shares_litellm_upstream,
    trajectory_abandoned_before_model, trajectory_single_model_abandoned_no_tools,
    HIGH_INPUT_TOKEN_THRESHOLD, SLOW_UPSTREAM_MODEL_MS_THRESHOLD,
};
use crate::feature_extraction_types::DetectionTag;
use serde_json::Value;

pub(crate) fn deterministic_tags_for_event(
    event_type: &str,
    summary: &str,
    evidence: Option<&str>,
    metadata: &Value,
    allow_text_bootstrap: bool,
    source_override: Option<&str>,
) -> Vec<DetectionTag> {
    let mut tags = Vec::new();

    tags.extend(explicit_producer_signal_tags(metadata, source_override));

    if bool_path(metadata, &["hook", "tool_loop"]) {
        if let Some(tool) = string_path(metadata, &["hook", "tool"]) {
            tags.push(DetectionTag::tool_loop(
                source_override.unwrap_or("hook_parser"),
                tool,
            ));
        }
    }
    if let Some(tool) = string_path(metadata, &["hook", "tool_loop_tool"]) {
        tags.push(DetectionTag::tool_loop(
            source_override.unwrap_or("hook_parser"),
            tool,
        ));
    }

    if bool_path(metadata, &["tool_loop_detected"])
        || bool_path(metadata, &["loop_detected"])
        || string_path(metadata, &["failure_class"]) == Some("tool_loop")
    {
        let tool = string_path(metadata, &["tool"])
            .or_else(|| string_path(metadata, &["tool_name"]))
            .or_else(|| string_path(metadata, &["payload", "tool_name"]))
            .unwrap_or("unknown");
        tags.push(DetectionTag::tool_loop(
            source_override.unwrap_or("tool_loop_detector"),
            tool,
        ));
    }

    if let Some(class) = string_path(metadata, &["failure_class"])
        .or_else(|| string_path(metadata, &["payload", "failure_class"]))
        .or_else(|| string_path(metadata, &["validation_failure"]))
    {
        match class {
            "missing_auth" | "wrong_endpoint" | "summarization_failure" | "migration_failure" => {
                tags.push(DetectionTag::new(
                    class,
                    source_override.unwrap_or("validation_parser"),
                ));
            }
            "user_interruption" if event_type == "failed_attempt" => {
                tags.push(DetectionTag::new(
                    "user_interruption",
                    source_override.unwrap_or("failed_attempt_classifier"),
                ));
            }
            _ => {}
        }
    }

    if event_type == "failed_attempt" {
        if bool_path(metadata, &["user_interruption"]) {
            tags.push(DetectionTag::new(
                "user_interruption",
                source_override.unwrap_or("failed_attempt_classifier"),
            ));
        }
        if let Some(tool) = string_path(metadata, &["repeated_tool"]) {
            tags.push(DetectionTag::tool_loop(
                source_override.unwrap_or("failed_attempt_classifier"),
                tool,
            ));
        }
    }

    if event_type == "summary"
        && (bool_path(metadata, &["summarization_failure"])
            || metadata.get("success").and_then(Value::as_bool) == Some(false))
    {
        tags.push(DetectionTag::new(
            "summarization_failure",
            source_override.unwrap_or("summarizer"),
        ));
    }

    if event_type == "context_pack" {
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

    if event_type == "assistant_message" {
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

    if event_type == "trajectory_result" {
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

    if summarizer_shares_litellm_upstream(metadata) {
        tags.push(DetectionTag::new(
            "summarizer_shared_upstream",
            source_override.unwrap_or("config_validator"),
        ));
    }

    if bool_path(metadata, &["correction_acknowledged"])
        || bool_path(metadata, &["payload", "correction_acknowledged"])
    {
        tags.push(DetectionTag::new(
            "correction_acknowledged",
            source_override.unwrap_or("remediation_parser"),
        ));
    }

    if allow_text_bootstrap {
        tags.extend(bootstrap_text_tags(
            summary,
            evidence,
            metadata,
            source_override,
        ));
    }

    dedupe_tags_within_producer(tags)
}
