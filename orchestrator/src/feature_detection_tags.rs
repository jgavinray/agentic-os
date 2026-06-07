//! Deterministic event detection tags for feature extraction.
//!
//! This module owns the producer-signal parsing, bootstrap text matching, and
//! metadata merging used before feature records are aggregated.

use crate::db::AgentEvent;
use crate::feature_extraction_metadata::{
    bool_path, context_pack_empty, context_pack_truncated, event_input_tokens_from_metadata,
    event_latency_ms_from_metadata, string_path, summarizer_shares_litellm_upstream,
    trajectory_abandoned_before_model, trajectory_single_model_abandoned_no_tools,
    HIGH_INPUT_TOKEN_THRESHOLD, SLOW_UPSTREAM_MODEL_MS_THRESHOLD,
};
use crate::feature_extraction_types::{DetectionTag, DETECTION_TAG_SOURCES, DETECTION_TAG_TYPES};
use serde_json::{json, Value};
use std::collections::BTreeSet;

pub fn annotate_event_metadata(
    event_type: &str,
    summary: &str,
    evidence: Option<&str>,
    metadata: Value,
) -> Value {
    merge_detection_tags(
        metadata.clone(),
        live_detection_tags_for_event(event_type, summary, evidence, &metadata),
    )
}

pub fn bootstrap_annotate_event_metadata(event: &AgentEvent) -> Value {
    merge_detection_tags(
        event.metadata.clone(),
        bootstrap_detection_tags_for_event(event),
    )
}

pub fn live_detection_tags_for_event(
    event_type: &str,
    summary: &str,
    evidence: Option<&str>,
    metadata: &Value,
) -> Vec<DetectionTag> {
    deterministic_tags_for_event(event_type, summary, evidence, metadata, false, None)
}

pub fn bootstrap_detection_tags_for_event(event: &AgentEvent) -> Vec<DetectionTag> {
    deterministic_tags_for_event(
        &event.event_type,
        &event.summary,
        event.evidence.as_deref(),
        &event.metadata,
        true,
        Some("bootstrap_migration"),
    )
}

fn deterministic_tags_for_event(
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

fn explicit_producer_signal_tags(
    metadata: &Value,
    source_override: Option<&str>,
) -> Vec<DetectionTag> {
    let mut tags = Vec::new();
    let Some(signals) = metadata.get("producer_signals").and_then(Value::as_object) else {
        return tags;
    };
    for (source, value) in signals {
        let source = source_override.unwrap_or(source.as_str());
        match value {
            Value::Array(values) => {
                for value in values {
                    if let Some(tag) = tag_from_signal_value(source, value) {
                        tags.push(tag);
                    }
                }
            }
            Value::Object(_) => {
                if let Some(tag) = tag_from_signal_value(source, value) {
                    tags.push(tag);
                }
            }
            Value::String(tag_type) => tags.push(DetectionTag::new(tag_type, source)),
            _ => {}
        }
    }
    tags
}

fn tag_from_signal_value(source: &str, value: &Value) -> Option<DetectionTag> {
    let obj = value.as_object()?;
    let tag_type = obj.get("type").and_then(Value::as_str)?;
    if tag_type == "tool_loop" {
        let tool = obj.get("tool").and_then(Value::as_str).unwrap_or("unknown");
        Some(DetectionTag::tool_loop(source, tool))
    } else {
        Some(DetectionTag::new(tag_type, source))
    }
}

fn bootstrap_text_tags(
    summary: &str,
    evidence: Option<&str>,
    metadata: &Value,
    source_override: Option<&str>,
) -> Vec<DetectionTag> {
    let source = source_override.unwrap_or("bootstrap_migration");
    let haystack = format!(
        "{}\n{}\n{}",
        summary,
        evidence.unwrap_or_default(),
        metadata
    )
    .to_ascii_lowercase();
    let mut tags = Vec::new();

    if haystack.contains("repeated read")
        || haystack.contains("read loop")
        || haystack.contains("read tool loop")
        || haystack.contains("tool 'read' called")
        || haystack.contains("tool \"read\" called")
        || haystack.contains("loop warning: tool 'read'")
    {
        tags.push(DetectionTag::tool_loop(source, "Read"));
    } else if haystack.contains("repeated bash")
        || haystack.contains("bash loop")
        || haystack.contains("bash tool loop")
        || haystack.contains("tool 'bash' called")
        || haystack.contains("tool \"bash\" called")
        || haystack.contains("loop warning: tool 'bash'")
    {
        tags.push(DetectionTag::tool_loop(source, "Bash"));
    } else if haystack.contains("tool loop") || haystack.contains("repeat identical tool") {
        tags.push(DetectionTag::tool_loop(source, "unknown"));
    }

    if haystack.contains("user interruption")
        || haystack.contains("user interrupted")
        || haystack.contains("user correction")
        || haystack.contains("request interrupted by user")
        || haystack.contains("interrupted by user for tool use")
    {
        tags.push(DetectionTag::new("user_interruption", source));
    }
    if haystack.contains("missing authorization")
        || haystack.contains("missing auth")
        || haystack.contains("authorization header")
        || haystack.contains("unauthorized")
        || haystack.contains("without authorization header")
        || haystack.contains("without the authorization header")
        || (haystack.contains("without") && haystack.contains("bearer"))
    {
        tags.push(DetectionTag::new("missing_auth", source));
    }
    if (haystack.contains("wrong endpoint") || haystack.contains("incorrect endpoint"))
        || (haystack.contains("localhost") && haystack.contains("correct endpoint"))
        || (haystack.contains("localhost") && haystack.contains("should be using"))
        || haystack.contains("trying localhost")
        || haystack.contains("not localhost")
    {
        tags.push(DetectionTag::new("wrong_endpoint", source));
    }
    if haystack.contains("summarization failure")
        || haystack.contains("summary failure")
        || haystack.contains("empty summary")
        || (haystack.contains("summarization") && haystack.contains("empty response"))
    {
        tags.push(DetectionTag::new("summarization_failure", source));
    }
    if haystack.contains("migration failure")
        || haystack.contains("baseline migration")
        || haystack.contains("extension creation")
    {
        tags.push(DetectionTag::new("migration_failure", source));
    }
    if haystack.contains("acknowledge")
        && (haystack.contains("correction") || haystack.contains("interruption"))
    {
        tags.push(DetectionTag::new("correction_acknowledged", source));
    }

    tags
}

fn merge_detection_tags(mut metadata: Value, new_tags: Vec<DetectionTag>) -> Value {
    if new_tags.is_empty() && metadata.get("detection_tags").is_none() {
        return metadata;
    }
    if !metadata.is_object() {
        metadata = json!({});
    }

    let mut tags = metadata
        .get("detection_tags")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let mut seen_by_source_type = BTreeSet::new();
    tags.retain(|value| {
        let Some(source) = value.get("source").and_then(Value::as_str) else {
            return true;
        };
        let Some(tag_type) = value.get("type").and_then(Value::as_str) else {
            return true;
        };
        seen_by_source_type.insert((source.to_string(), tag_type.to_string()))
    });

    for tag in new_tags {
        if !DETECTION_TAG_TYPES.contains(&tag.tag_type.as_str()) {
            continue;
        }
        if !DETECTION_TAG_SOURCES.contains(&tag.source.as_str()) {
            continue;
        }
        let key = (tag.source.clone(), tag.tag_type.clone());
        if !seen_by_source_type.insert(key) {
            tracing::debug!(
                source = %tag.source,
                tag_type = %tag.tag_type,
                "dropping duplicate detection tag from same producer for event"
            );
            continue;
        }
        if let Ok(value) = serde_json::to_value(tag) {
            tracing::debug!(
                target: "feature_extraction",
                tag_type = %value.get("type").and_then(|v| v.as_str()).unwrap_or("unknown"),
                source = %value.get("source").and_then(|v| v.as_str()).unwrap_or("unknown"),
                tool = %bounded_tool_label(value.get("tool").and_then(|v| v.as_str())),
                "detection tag emitted"
            );
            tags.push(value);
        }
    }

    if let Some(obj) = metadata.as_object_mut() {
        obj.insert("detection_tags".to_string(), Value::Array(tags));
    }
    metadata
}

fn bounded_tool_label(value: Option<&str>) -> &'static str {
    match value {
        Some("Read") => "Read",
        Some("Bash") => "Bash",
        Some("unknown") | None => "unknown",
        Some(_) => "other",
    }
}

fn dedupe_tags_within_producer(tags: Vec<DetectionTag>) -> Vec<DetectionTag> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for tag in tags {
        let key = (tag.source.clone(), tag.tag_type.clone());
        if seen.insert(key) {
            out.push(tag);
        } else {
            tracing::debug!(
                source = %tag.source,
                tag_type = %tag.tag_type,
                "dropping duplicate detection tag from same producer"
            );
        }
    }
    out
}
