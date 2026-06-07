//! Deterministic request classification taxonomy.
//!
//! This module defines the bounded labels and table-facing structs for the
//! pre-LLM request classification layer. Feature extraction, backfill, and live
//! routing are later phases.

use chrono::Utc;
use serde_json::Value;

use crate::request_classification_composition::{
    decomposition_fragments, has_subtask_action_signal,
};
use crate::request_classification_feature_json::features_to_json;
use crate::request_classification_features::extract_features;
use crate::request_classification_input::{event_text, has_request_text, metadata_key_text};
pub use crate::request_classification_labels::{
    bounded_complexity, bounded_domain, bounded_intent, bounded_live_policy_action,
    bounded_live_policy_bypass, bounded_live_policy_reason, bounded_risk, bounded_route,
};
use crate::request_classification_rules::{
    classify_artifact, classify_complexity, classify_domain, classify_intent, classify_risk,
    detected_domains, recommend_route, response_contract,
};
pub use crate::request_classification_runtime::{
    evaluate_live_policy, record_classification_metrics,
};
pub use crate::request_classification_store::{
    classify_and_persist_event, persist_classification, request_classification_report,
    run_backfill, update_classification_if_changed,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompositeAnalysis {
    pub(crate) is_composite: bool,
    pub(crate) decomposition_candidate: bool,
    pub(crate) reason: &'static str,
    pub(crate) sub_intents: Vec<RequestIntent>,
}

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

fn analyze_composition(text: &str, lower: &str, event_type: &str) -> CompositeAnalysis {
    if text.trim().is_empty() {
        return CompositeAnalysis {
            is_composite: false,
            decomposition_candidate: false,
            reason: "none",
            sub_intents: Vec::new(),
        };
    }

    let (fragments, reason) = decomposition_fragments(text, lower);
    let mut sub_intents = Vec::new();
    for fragment in fragments.iter().take(5) {
        let fragment = fragment.trim();
        if fragment.len() < 3 {
            continue;
        }
        let fragment_lower = fragment.to_ascii_lowercase();
        if !has_subtask_action_signal(&fragment_lower) {
            continue;
        }
        let features = extract_features(fragment, &fragment_lower, "");
        sub_intents.push(classify_intent(&features, &fragment_lower, event_type));
    }

    let decomposition_candidate = sub_intents.len() >= 2;
    CompositeAnalysis {
        is_composite: decomposition_candidate,
        decomposition_candidate,
        reason: if decomposition_candidate {
            reason
        } else {
            "none"
        },
        sub_intents: if decomposition_candidate {
            sub_intents
        } else {
            Vec::new()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::AgentEvent;
    use serde_json::json;
    use std::collections::HashSet;

    #[test]
    fn feature_extraction_is_deterministic_for_identical_input() {
        let event = event(
            "e-1",
            "Please explain the Docker compose error",
            Some("ERROR failed to connect to http://localhost:8088"),
        );

        let first = classify_request_event(&event);
        let second = classify_request_event(&event);

        assert_eq!(first, second);
        assert_eq!(first.domain, RequestDomain::Docker);
        assert_eq!(first.intent, RequestIntent::Debug);
    }

    #[test]
    fn feature_keys_are_closed_and_stable() {
        let row = classify_request_event(&event(
            "e-keys",
            "Summarize this Kubernetes log",
            Some("[ERROR] pod failed"),
        ));
        let object = row.features.as_object().expect("features must be object");
        let actual: HashSet<&str> = object.keys().map(String::as_str).collect();
        let expected: HashSet<&str> = FEATURE_KEYS.iter().copied().collect();

        assert_eq!(actual, expected);
    }

    #[test]
    fn composite_requests_emit_bounded_sub_intents() {
        let row = classify_request_event(&event(
            "e-composite",
            "Search the repo for context injection; implement the fix in src/main.rs; run cargo test; summarize the result",
            None,
        ));

        assert_eq!(row.features["is_composite"], true);
        assert_eq!(row.features["decomposition_candidate"], true);
        assert_eq!(row.features["decomposition_reason"], "sequence_separator");
        assert_eq!(row.features["sub_intent_count"], 4);
        assert_eq!(
            row.features["sub_intents"],
            json!(["search", "implement", "operate_tool", "summarize"])
        );
    }

    #[test]
    fn implementation_language_maps_to_implement_intent() {
        let row = classify_request_event(&event(
            "e-implement",
            "Implement the classifier change in src/request_classification.rs",
            None,
        ));

        assert_eq!(row.intent, RequestIntent::Implement);
        assert_eq!(row.response_contract, ResponseContract::ValidationRequired);
    }

    #[test]
    fn single_intent_with_conjunction_is_not_decomposed() {
        let row = classify_request_event(&event(
            "e-not-composite",
            "Explain Docker and Kubernetes networking",
            None,
        ));

        assert_eq!(row.features["is_composite"], false);
        assert_eq!(row.features["decomposition_candidate"], false);
        assert_eq!(row.features["decomposition_reason"], "none");
        assert_eq!(row.features["sub_intent_count"], 0);
        assert_eq!(row.features["sub_intents"], json!([]));
    }

    #[test]
    fn secret_candidates_set_secret_present() {
        let row = classify_request_event(&event(
            "e-secret",
            "The request included Authorization: Bearer sk-secret-value",
            None,
        ));

        assert!(row.risk.contains(&RequestRisk::SecretPresent));
        assert_eq!(row.recommended_route, RecommendedRoute::RefuseOrGuardrail);
        assert_eq!(row.features["has_secret_candidate"], true);
    }

    #[test]
    fn destructive_commands_set_destructive_command() {
        let row = classify_request_event(&event(
            "e-destructive",
            "Run rm -rf /tmp/agentic-os-cache to clean everything",
            None,
        ));

        assert!(row.risk.contains(&RequestRisk::DestructiveCommand));
        assert_eq!(row.complexity, RequestComplexity::L5HighRisk);
        assert_eq!(row.recommended_route, RecommendedRoute::RefuseOrGuardrail);
    }

    #[test]
    fn latest_current_phrasing_sets_external_current_info_required() {
        let row = classify_request_event(&event(
            "e-latest",
            "What is the latest vLLM release today?",
            None,
        ));

        assert!(row.risk.contains(&RequestRisk::ExternalCurrentInfoRequired));
        assert_eq!(row.domain, RequestDomain::LlmInference);
        assert_eq!(row.recommended_route, RecommendedRoute::WebRequired);
    }

    #[test]
    fn high_stakes_domains_set_high_stakes_risk() {
        for (id, summary, domain) in [
            (
                "e-medical",
                "Can you diagnose this medical symptom?",
                RequestDomain::Medical,
            ),
            (
                "e-legal",
                "Is this legal contract enforceable?",
                RequestDomain::Legal,
            ),
            (
                "e-finance",
                "Should I make this tax and investment move?",
                RequestDomain::Finance,
            ),
        ] {
            let row = classify_request_event(&event(id, summary, None));
            assert_eq!(row.domain, domain);
            assert!(row.risk.contains(&RequestRisk::HighStakes));
            assert_eq!(row.recommended_route, RecommendedRoute::RefuseOrGuardrail);
        }
    }

    #[test]
    fn infrastructure_terms_map_to_expected_domains() {
        let row = classify_request_event(&event(
            "e-domains",
            "kubectl deployment, Docker container, vLLM model, DNS proxy, and JWT auth",
            None,
        ));

        assert_eq!(row.domain, RequestDomain::Kubernetes);
        assert!(row.secondary_domains.contains(&RequestDomain::Docker));
        assert!(row.secondary_domains.contains(&RequestDomain::LlmInference));
        assert!(row.secondary_domains.contains(&RequestDomain::Networking));
        assert!(row.secondary_domains.contains(&RequestDomain::Security));
        assert!(row.secondary_domains.contains(&RequestDomain::Shell));
        assert_eq!(
            row.features["detected_domain_terms"],
            json!([
                "kubernetes",
                "docker",
                "llm_inference",
                "networking",
                "security",
                "shell"
            ])
        );
    }

    #[test]
    fn unknown_or_empty_events_produce_bounded_safe_defaults() {
        let row = classify_request_event(&event("e-empty", "", None));

        assert_eq!(row.intent, RequestIntent::Unknown);
        assert_eq!(row.domain, RequestDomain::Unknown);
        assert_eq!(row.artifact_type, RequestArtifactType::Unknown);
        assert_eq!(row.risk, vec![RequestRisk::Unknown]);
        assert_eq!(row.complexity, RequestComplexity::Unknown);
        assert_eq!(row.recommended_route, RecommendedRoute::Unknown);
        assert_eq!(row.response_contract, ResponseContract::Unknown);
        assert_eq!(row.features["char_count"], 0);
        assert_eq!(row.features["estimated_tokens"], 0);
    }

    #[test]
    fn non_empty_generic_requests_use_safe_fallback_labels() {
        let row = classify_request_event(&event("e-generic", "Can you help with this?", None));

        assert_eq!(row.intent, RequestIntent::Explain);
        assert_eq!(row.domain, RequestDomain::Generic);
        assert_eq!(row.artifact_type, RequestArtifactType::PlainText);
        assert_eq!(row.risk, vec![RequestRisk::None]);
        assert_ne!(row.complexity, RequestComplexity::Unknown);
        assert_ne!(row.recommended_route, RecommendedRoute::Unknown);
        assert_ne!(row.response_contract, ResponseContract::Unknown);
    }

    #[test]
    fn features_do_not_copy_raw_text() {
        let raw_secret = "sk-raw-secret-value";
        let row = classify_request_event(&event(
            "e-privacy",
            &format!("Please classify this Authorization: Bearer {raw_secret}"),
            Some("Raw evidence body should not be copied into features"),
        ));
        let serialized = row.features.to_string();

        assert!(!serialized.contains(raw_secret));
        assert!(!serialized.contains("Raw evidence body"));
        assert!(feature_string_values_are_bounded(&row.features));
    }

    #[test]
    fn classifiable_request_event_selection_is_bounded() {
        let user_event = event("e-user", "hello", None);
        assert!(is_classifiable_request_event(&user_event));

        let empty_user_event = event("e-empty-user", "", None);
        assert!(!is_classifiable_request_event(&empty_user_event));

        let whitespace_user_event = event("e-whitespace-user", " \n\t ", None);
        assert!(!is_classifiable_request_event(&whitespace_user_event));

        let evidence_only_event = event("e-evidence-only", "", Some("hello from evidence"));
        assert!(is_classifiable_request_event(&evidence_only_event));

        let mut request_role = event("e-role", "hello", None);
        request_role.event_type = "checkpoint".to_string();
        request_role.event_role = Some("request".to_string());
        assert!(is_classifiable_request_event(&request_role));

        let mut maintenance = event("e-maint", "summary maintenance", None);
        maintenance.event_type = "summary".to_string();
        maintenance.event_role = None;
        assert!(!is_classifiable_request_event(&maintenance));
    }

    #[test]
    fn backfill_sql_does_not_treat_separator_newline_as_request_text() {
        let source = include_str!("request_classification.rs");

        assert!(source.contains("btrim(coalesce(e.summary, '') || coalesce(e.evidence, '')"));
        assert!(!source.contains("btrim(coalesce(e.summary, '') || E'\\n'"));
    }

    fn event(id: &str, summary: &str, evidence: Option<&str>) -> AgentEvent {
        AgentEvent {
            id: id.to_string(),
            session_id: "session-1".to_string(),
            repo: "agent-os".to_string(),
            actor: "user".to_string(),
            event_type: "user_message".to_string(),
            summary: summary.to_string(),
            evidence: evidence.map(str::to_string),
            metadata: json!({
                "payload": {
                    "kind": "test"
                }
            }),
            correlation_id: None,
            parent_event_id: None,
            trajectory_id: None,
            attempt_index: None,
            event_role: Some("request".to_string()),
            created_at: chrono::DateTime::parse_from_rfc3339("2026-05-23T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            summary_level: 0,
        }
    }

    fn feature_string_values_are_bounded(value: &Value) -> bool {
        match value {
            Value::String(value) => enum_inventory()
                .iter()
                .any(|(_name, variants)| variants.contains(&value.as_str())),
            Value::Array(values) => values.iter().all(feature_string_values_are_bounded),
            Value::Object(values) => values.values().all(feature_string_values_are_bounded),
            _ => true,
        }
    }
}
