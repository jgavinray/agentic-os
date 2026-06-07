//! Deterministic trajectory feature extraction and operational constraints.
//!
//! The source of truth remains `agent_events`. This module consumes structured
//! detection tags from event metadata, aggregates compact feature rows, and
//! renders bounded operational constraints without LLM calls or probabilistic
//! classification. Free-form prose matching is intentionally limited to the
//! bootstrap tagging path for historical events that predate structured tags.

use crate::db::AgentEvent;

pub use crate::feature_constraints::enforce_constraint_token_budget;
pub use crate::feature_detection_tags::{
    annotate_event_metadata, bootstrap_annotate_event_metadata, bootstrap_detection_tags_for_event,
    live_detection_tags_for_event,
};
use crate::feature_extraction_grouping::group_events_by_feature_window;
pub use crate::feature_extraction_persistence::persist_feature_record;
use crate::feature_extraction_records::extract_record_for_group;
pub use crate::feature_extraction_store::{
    bootstrap_tag_events, extract_and_persist_scope, load_events_for_scope,
    operational_constraints_for_context, run_backfill, run_inline_extraction_best_effort,
};
pub use crate::feature_extraction_types::{
    constraint_freshness_window_sec_from_env, feature_extraction_enabled_from_env,
    feature_window_sec_from_env, max_operational_constraints_from_env,
    operational_constraints_token_budget_from_env, BackfillOptions, BackfillReport, DetectionTag,
    ExtractionConfig, ExtractionReport, ExtractionScope, FeatureRecord, OperationalConstraint,
    SuppressedConstraint, DEFAULT_CONSTRAINT_FRESHNESS_WINDOW_SEC, DEFAULT_FEATURE_WINDOW_SEC,
    DEFAULT_MAX_OPERATIONAL_CONSTRAINTS, DEFAULT_OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET,
    DETECTION_TAG_SOURCES, DETECTION_TAG_TYPES, FEATURE_FAILURE_CLASSES,
    OPERATIONAL_CONSTRAINT_TYPES, OPERATIONAL_SUPPRESSION_REASONS, TAG_SCHEMA_VERSION,
};

pub use crate::feature_extraction_metadata::{
    CONTEXT_PACK_EMPTY_TOKEN_THRESHOLD, HIGH_INPUT_TOKEN_THRESHOLD,
    SLOW_UPSTREAM_MODEL_MS_THRESHOLD,
};

pub fn extract_records(events: &[AgentEvent], config: &ExtractionConfig) -> ExtractionReport {
    let mut report = ExtractionReport::default();
    for (key, grouped_events) in group_events_by_feature_window(events, config.feature_window_sec) {
        let (record, unknown) = extract_record_for_group(&key, &grouped_events, config);
        report.unknown_tag_schema_versions += unknown;
        report.records.push(record);
    }
    report
}

#[cfg(test)]
#[path = "feature_extraction_bootstrap_tests.rs"]
mod bootstrap_tests;

#[cfg(test)]
#[path = "feature_extraction_tests.rs"]
mod tests;
