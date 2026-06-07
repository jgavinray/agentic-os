//! Deterministic request classification taxonomy.
//!
//! This module defines the bounded labels and table-facing structs for the
//! pre-LLM request classification layer. Feature extraction, backfill, and live
//! routing are later phases.

use chrono::Utc;
use serde_json::Value;

use crate::request_classification_composition::analyze_composition;
use crate::request_classification_feature_json::features_to_json;
use crate::request_classification_features::extract_features;
use crate::request_classification_input::{event_text, has_request_text, metadata_key_text};
pub use crate::request_classification_labels::{
    bounded_complexity, bounded_domain, bounded_intent, bounded_live_policy_action,
    bounded_live_policy_bypass, bounded_live_policy_reason, bounded_risk, bounded_route,
};
pub use crate::request_classification_report::request_classification_report;
use crate::request_classification_rules::{
    classify_artifact, classify_complexity, classify_domain, classify_intent, classify_risk,
    detected_domains, recommend_route, response_contract,
};
pub use crate::request_classification_runtime::{
    evaluate_live_policy, record_classification_metrics,
};
pub use crate::request_classification_store::{
    classify_and_persist_event, persist_classification, run_backfill,
    update_classification_if_changed,
};
pub use crate::request_classification_types::{
    enum_inventory, live_policy_config_from_env, request_classification_startup_backfill_enabled,
    request_classification_startup_batch_size, BackfillOptions, BackfillReport, LabelCount,
    LivePolicyConfig, LivePolicyDecision, PersistOutcome, RecommendedRoute, ReportOptions,
    RequestArtifactType, RequestClassification, RequestClassificationReport, RequestComplexity,
    RequestDomain, RequestIntent, RequestRisk, ResponseContract, SessionRouteCount,
    CLASSIFICATION_SCHEMA_VERSION, CLASSIFIER_SOURCE_DETERMINISTIC_RULES, FEATURE_KEYS,
    ROUTING_POLICY_VERSION,
};

/// Classify a loaded event into deterministic request-level features and labels.
///
/// This is a pure Phase 2 entry point: it does not query storage, call models,
/// touch the filesystem, emit metrics, or change live routing behavior.
pub fn classify_request_event(event: &crate::db::AgentEvent) -> RequestClassification {
    let text = event_text(event);
    let lower = text.to_ascii_lowercase();
    let metadata_keys = metadata_key_text(&event.metadata);
    let metadata_keys_lower = metadata_keys.to_ascii_lowercase();
    let features = extract_features(&text, &lower, &metadata_keys_lower);
    let detected_domains = detected_domains(&features, &lower);
    let composite = analyze_composition(&text, &lower, &event.event_type);

    let mut row = RequestClassification::deterministic(
        event.id.clone(),
        event.repo.clone(),
        event.session_id.clone(),
        event.created_at,
    );
    row.trajectory_id = event.trajectory_id;
    row.features = features_to_json(&features, &detected_domains, &composite);
    row.intent = classify_intent(&features, &lower, &event.event_type);
    row.domain = classify_domain(&features, &lower, &detected_domains);
    row.secondary_domains = detected_domains
        .iter()
        .copied()
        .filter(|domain| *domain != row.domain)
        .collect();
    row.artifact_type = classify_artifact(&features, &lower);
    row.risk = classify_risk(&features, &lower, row.domain);
    row.complexity = classify_complexity(&features, row.intent, &row.risk, detected_domains.len());
    row.recommended_route = recommend_route(row.intent, row.complexity, &row.risk, &features);
    row.response_contract = response_contract(row.intent, row.artifact_type, row.recommended_route);
    row
}

pub fn classify_request_text(
    repo: &str,
    session_id: &str,
    summary: &str,
    evidence: Option<&str>,
    event_type: &str,
) -> RequestClassification {
    let event = crate::db::AgentEvent {
        id: "live-request".to_string(),
        session_id: session_id.to_string(),
        repo: repo.to_string(),
        actor: "user".to_string(),
        event_type: event_type.to_string(),
        summary: summary.to_string(),
        evidence: evidence.map(str::to_string),
        metadata: Value::Object(Default::default()),
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: Some("request".to_string()),
        created_at: Utc::now(),
        summary_level: 0,
    };
    classify_request_event(&event)
}

pub fn is_classifiable_request_event(event: &crate::db::AgentEvent) -> bool {
    if !has_request_text(event) {
        return false;
    }

    event.event_type == "user_message"
        || event.event_role.as_deref() == Some("request")
        || (event.event_type == "context_pack"
            && event.event_role.as_deref() == Some("context_pack")
            && event.metadata.get("request").is_some())
}

#[cfg(test)]
#[path = "request_classification_tests.rs"]
mod tests;
