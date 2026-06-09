use serde_json::Value;

use crate::db::AgentEvent;
pub(crate) use crate::feature_metadata_paths::{
    array_path_has_values, bool_path, bool_path_value, i64_path, string_path,
};
pub(crate) use crate::feature_metadata_summarizer::{
    summarizer_has_dedicated_upstream, summarizer_shares_litellm_upstream,
};
pub(crate) use crate::feature_metadata_trajectory::{
    trajectory_abandoned_before_model, trajectory_single_model_abandoned_no_tools,
};

pub const CONTEXT_PACK_EMPTY_TOKEN_THRESHOLD: i64 = 50;
pub const HIGH_INPUT_TOKEN_THRESHOLD: i64 = 100_000;
pub const SLOW_UPSTREAM_MODEL_MS_THRESHOLD: i64 = 60_000;

pub(crate) fn known_fact<'a>(event: &'a AgentEvent, key: &str) -> Option<&'a str> {
    event
        .metadata
        .get("known_facts")
        .and_then(|facts| facts.get(key))
        .and_then(Value::as_str)
        .or_else(|| {
            event
                .metadata
                .get("payload")
                .and_then(|payload| payload.get("known_facts"))
                .and_then(|facts| facts.get(key))
                .and_then(Value::as_str)
        })
}

pub(crate) fn event_success(event: &AgentEvent) -> bool {
    event.metadata.get("success").and_then(Value::as_bool) == Some(true)
        || event
            .metadata
            .get("payload")
            .and_then(|payload| payload.get("success"))
            .and_then(Value::as_bool)
            == Some(true)
}

pub(crate) fn event_tool_name(event: &AgentEvent) -> Option<&str> {
    event
        .metadata
        .get("payload")
        .and_then(|payload| payload.get("tool_name"))
        .and_then(Value::as_str)
        .or_else(|| event.metadata.get("tool_name").and_then(Value::as_str))
        .or_else(|| event.metadata.get("tool").and_then(Value::as_str))
}

pub(crate) fn event_endpoint(event: &AgentEvent) -> Option<&str> {
    event
        .metadata
        .get("endpoint")
        .and_then(Value::as_str)
        .or_else(|| {
            event
                .metadata
                .get("payload")
                .and_then(|payload| payload.get("endpoint"))
                .and_then(Value::as_str)
        })
}

pub(crate) fn is_successful_authenticated_request(event: &AgentEvent) -> bool {
    event_success(event)
        && (bool_path(&event.metadata, &["authenticated"])
            || bool_path(&event.metadata, &["payload", "authenticated"])
            || bool_path(&event.metadata, &["auth", "authenticated"]))
}

pub(crate) fn is_successful_summarization(event: &AgentEvent) -> bool {
    event_success(event) && (event.event_type == "summary" || event.event_type == "summarization")
}

pub(crate) fn is_successful_migration(event: &AgentEvent) -> bool {
    event_success(event)
        && (event.event_type == "migration_result"
            || event.event_type == "migration"
            || bool_path(&event.metadata, &["payload", "migration_success"]))
}

pub(crate) fn context_pack_empty(metadata: &Value) -> bool {
    let Some(tokens) = context_pack_token_count(metadata) else {
        return false;
    };
    tokens <= CONTEXT_PACK_EMPTY_TOKEN_THRESHOLD
        && !array_path_has_values(metadata, &["retrieved_event_ids"])
        && !array_path_has_values(metadata, &["payload", "retrieved_event_ids"])
        && !array_path_has_values(metadata, &["memory_levels_used"])
        && !array_path_has_values(metadata, &["payload", "memory_levels_used"])
}

pub(crate) fn context_pack_truncated(metadata: &Value) -> bool {
    context_pack_truncated_value(metadata).unwrap_or(false)
}

pub(crate) fn context_pack_token_count(metadata: &Value) -> Option<i64> {
    i64_path(metadata, &["total_context_tokens"])
        .or_else(|| i64_path(metadata, &["payload", "total_context_tokens"]))
}

pub(crate) fn context_pack_truncated_value(metadata: &Value) -> Option<bool> {
    bool_path_value(metadata, &["truncated"])
        .or_else(|| bool_path_value(metadata, &["payload", "truncated"]))
}

pub(crate) fn event_input_tokens_from_metadata(metadata: &Value) -> Option<i64> {
    i64_path(metadata, &["input_tokens"])
        .or_else(|| i64_path(metadata, &["payload", "input_tokens"]))
        .or_else(|| i64_path(metadata, &["total_input_tokens"]))
        .or_else(|| i64_path(metadata, &["payload", "total_input_tokens"]))
        .or_else(|| i64_path(metadata, &["usage", "input_tokens"]))
        .or_else(|| i64_path(metadata, &["payload", "usage", "input_tokens"]))
        .or_else(|| i64_path(metadata, &["usage", "prompt_tokens"]))
        .or_else(|| i64_path(metadata, &["payload", "usage", "prompt_tokens"]))
}

pub(crate) fn event_latency_ms_from_metadata(metadata: &Value) -> Option<i64> {
    i64_path(metadata, &["latency_ms"])
        .or_else(|| i64_path(metadata, &["payload", "latency_ms"]))
        .or_else(|| i64_path(metadata, &["total_latency_ms"]))
        .or_else(|| i64_path(metadata, &["payload", "total_latency_ms"]))
        .or_else(|| {
            i64_path(metadata, &["latency_seconds"])
                .or_else(|| i64_path(metadata, &["payload", "latency_seconds"]))
                .map(|seconds| seconds.saturating_mul(1000))
        })
}
