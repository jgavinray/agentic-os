use std::collections::BTreeSet;

use serde_json::Value;

use crate::db::AgentEvent;
use crate::feature_extraction_metadata::known_fact;
use crate::feature_extraction_types::{
    DetectionTag, FeatureRecord, DETECTION_TAG_SOURCES, DETECTION_TAG_TYPES, TAG_SCHEMA_VERSION,
};

#[derive(Default)]
pub(crate) struct ParsedTags {
    pub(crate) tags: Vec<DetectionTag>,
    pub(crate) unknown_schema_versions: u64,
}

pub(crate) fn recognized_tags(event: &AgentEvent) -> ParsedTags {
    let mut parsed = ParsedTags::default();
    let Some(values) = event
        .metadata
        .get("detection_tags")
        .and_then(Value::as_array)
    else {
        return parsed;
    };
    for value in values {
        let schema_version = value
            .get("tag_schema_version")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if schema_version != TAG_SCHEMA_VERSION as u64 {
            parsed.unknown_schema_versions += 1;
            continue;
        }
        let Ok(tag) = serde_json::from_value::<DetectionTag>(value.clone()) else {
            continue;
        };
        if !DETECTION_TAG_TYPES.contains(&tag.tag_type.as_str()) {
            continue;
        }
        if !DETECTION_TAG_SOURCES.contains(&tag.source.as_str()) {
            continue;
        }
        if tag.tag_type == "tool_loop" && tag.tool.as_deref().unwrap_or("").is_empty() {
            continue;
        }
        parsed.tags.push(tag);
    }
    parsed
}

pub(crate) fn extraction_condition_key(tag: &DetectionTag) -> String {
    if tag.tag_type == "tool_loop" {
        format!("tool_loop:{}", tag.tool.as_deref().unwrap_or("unknown"))
    } else {
        tag.tag_type.clone()
    }
}

pub(crate) fn apply_tag_to_record(
    event: &AgentEvent,
    tag: &DetectionTag,
    record: &mut FeatureRecord,
    failure_classes: &mut BTreeSet<String>,
) {
    match tag.tag_type.as_str() {
        "tool_loop" => {
            record.tool_loop_count += 1;
            record.loop_detected = true;
            failure_classes.insert("tool_loop".to_string());
            match tag.tool.as_deref() {
                Some("Read") => record.repeated_read_loop_count += 1,
                Some("Bash") => record.repeated_bash_loop_count += 1,
                _ => {}
            }
        }
        "user_interruption" => {
            record.user_interruption_count += 1;
            record.user_interrupted = true;
            failure_classes.insert("user_interruption".to_string());
        }
        "missing_auth" => {
            record.missing_auth_count += 1;
            failure_classes.insert("missing_auth".to_string());
            if let Some(value) = known_fact(event, "auth_header") {
                record.known_auth_header = Some(value.to_string());
            }
        }
        "wrong_endpoint" => {
            record.wrong_endpoint_count += 1;
            failure_classes.insert("wrong_endpoint".to_string());
            if let Some(value) = known_fact(event, "endpoint") {
                record.known_endpoint = Some(value.to_string());
            }
        }
        "summarization_failure" => {
            record.summarization_failure_count += 1;
            failure_classes.insert("summarization_failure".to_string());
        }
        "migration_failure" => {
            record.migration_failure_count += 1;
            failure_classes.insert("migration_failure".to_string());
            if let Some(value) = known_fact(event, "migration_fix") {
                record.known_migration_fix = Some(value.to_string());
            }
        }
        "context_pack_empty" => {
            record.context_pack_empty_count += 1;
            failure_classes.insert("context_pack_empty".to_string());
        }
        "context_pack_truncated" => {
            record.context_pack_truncated_count += 1;
            failure_classes.insert("context_pack_truncated".to_string());
        }
        "high_input_tokens" => {
            record.high_input_token_count += 1;
            failure_classes.insert("high_input_tokens".to_string());
        }
        "slow_upstream_model" => {
            record.slow_upstream_model_count += 1;
            failure_classes.insert("slow_upstream_model".to_string());
        }
        "empty_tool_use_message" => {
            record.empty_tool_use_message_count += 1;
            failure_classes.insert("empty_tool_use_message".to_string());
        }
        "abandoned_before_model" => {
            record.abandoned_before_model_count += 1;
            failure_classes.insert("abandoned_before_model".to_string());
        }
        "single_model_abandoned_no_tools" => {
            record.single_model_abandoned_no_tools_count += 1;
            failure_classes.insert("single_model_abandoned_no_tools".to_string());
        }
        "summarizer_shared_upstream" => {
            record.summarizer_shared_upstream_count += 1;
            failure_classes.insert("summarizer_shared_upstream".to_string());
        }
        "correction_acknowledged" => {}
        _ => {}
    }
}
