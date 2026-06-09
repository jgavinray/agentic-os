//! Metadata merge helpers for deterministic feature detection tags.

use crate::feature_extraction_types::{DetectionTag, DETECTION_TAG_SOURCES, DETECTION_TAG_TYPES};
use serde_json::{json, Value};
use std::collections::BTreeSet;

pub(crate) fn merge_detection_tags(mut metadata: Value, new_tags: Vec<DetectionTag>) -> Value {
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

pub(crate) fn dedupe_tags_within_producer(tags: Vec<DetectionTag>) -> Vec<DetectionTag> {
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

fn bounded_tool_label(value: Option<&str>) -> &'static str {
    match value {
        Some("Read") => "Read",
        Some("Bash") => "Bash",
        Some("unknown") | None => "unknown",
        Some(_) => "other",
    }
}
