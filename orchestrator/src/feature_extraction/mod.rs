//! Deterministic trajectory feature extraction and operational constraints.
//!
//! The source of truth remains `agent_events`. This module consumes structured
//! detection tags from event metadata, aggregates compact feature rows, and
//! renders bounded operational constraints without LLM calls or probabilistic
//! classification. Free-form prose matching is intentionally limited to the
//! bootstrap tagging path for historical events that predate structured tags.

use crate::db::AgentEvent;

pub mod config;
pub mod constraint_budget;
pub mod constraint_failures;
pub mod constraint_suppression;
pub mod constraint_templates;
pub mod constraints;
pub mod detection_bootstrap;
pub mod detection_event_rules;
pub mod detection_event_type_rules;
pub mod detection_producer_signals;
pub mod detection_tag_merge;
pub mod detection_tags;
pub mod env;
pub mod grouping;
pub mod inline;
pub mod metadata;
pub mod metadata_paths;
pub mod metadata_summarizer;
pub mod metadata_trajectory;
pub mod persistence;
pub mod persistence_sql;
pub mod queries;
pub mod record_init;
pub mod records;
pub mod recovery;
pub mod store;
pub mod tags;
pub mod types;

pub use constraints::enforce_constraint_token_budget;
pub use detection_tags::{
    annotate_event_metadata, bootstrap_annotate_event_metadata, bootstrap_detection_tags_for_event,
    live_detection_tags_for_event,
};
use grouping::group_events_by_feature_window;
pub use inline::run_inline_extraction_best_effort;
pub use persistence::persist_feature_record;
use records::extract_record_for_group;
pub use store::{
    bootstrap_tag_events, extract_and_persist_scope, load_events_for_scope,
    operational_constraints_for_context, run_backfill,
};
pub use types::{
    constraint_freshness_window_sec_from_env, feature_extraction_enabled_from_env,
    feature_window_sec_from_env, max_operational_constraints_from_env,
    operational_constraints_token_budget_from_env, BackfillOptions, BackfillReport, DetectionTag,
    ExtractionConfig, ExtractionReport, ExtractionScope, FeatureRecord, OperationalConstraint,
    SuppressedConstraint, DEFAULT_CONSTRAINT_FRESHNESS_WINDOW_SEC, DEFAULT_FEATURE_WINDOW_SEC,
    DEFAULT_MAX_OPERATIONAL_CONSTRAINTS, DEFAULT_OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET,
    DETECTION_TAG_SOURCES, DETECTION_TAG_TYPES, FEATURE_FAILURE_CLASSES,
    OPERATIONAL_CONSTRAINT_TYPES, OPERATIONAL_SUPPRESSION_REASONS, TAG_SCHEMA_VERSION,
};

pub use metadata::{
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
mod tests {
    pub(crate) use super::*;
    pub(crate) mod support;
    pub(crate) use support as test_support;

    mod bootstrap;
    mod constraint;
    mod detection;
}
