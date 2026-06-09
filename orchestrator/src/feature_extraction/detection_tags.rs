//! Deterministic event detection tags for feature extraction.
//!
//! This module owns the producer-signal parsing, bootstrap text matching, and
//! metadata merging used before feature records are aggregated.

use crate::db::AgentEvent;
use crate::feature_detection_event_rules::deterministic_tags_for_event;
use crate::feature_detection_tag_merge::merge_detection_tags;
use crate::feature_extraction_types::DetectionTag;
use serde_json::Value;

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
