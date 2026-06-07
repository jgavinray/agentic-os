use super::*;
use crate::request_classification::{
    RequestArtifactType, RequestClassification, RequestIntent, RequestRisk,
};
use chrono::Utc;

fn classification(
    intent: RequestIntent,
    risk: Vec<RequestRisk>,
    artifact_type: RequestArtifactType,
) -> RequestClassification {
    RequestClassification {
        event_id: "test-event".to_string(),
        repo: "test-repo".to_string(),
        session_id: "test-session".to_string(),
        trajectory_id: None,
        event_created_at: Utc::now(),
        classified_at: Utc::now(),
        classification_schema_version: 1,
        routing_policy_version: "deterministic-v1".to_string(),
        classifier_source: "deterministic_rules".to_string(),
        intent,
        domain: crate::request_classification::RequestDomain::Generic,
        secondary_domains: vec![],
        artifact_type,
        risk,
        complexity: crate::request_classification::RequestComplexity::L1Simple,
        recommended_route: crate::request_classification::RecommendedRoute::SmallLocalModel,
        response_contract: crate::request_classification::ResponseContract::DirectAnswer,
        features: serde_json::json!({}),
    }
}

#[test]
fn test_serde_round_trip() {
    let c = classification(
        RequestIntent::Explain,
        vec![RequestRisk::None],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(&c, "explain this", false);

    let serialized = serde_json::to_string(&policy).expect("serialize");
    let deserialized: OrchestrationPolicy = serde_json::from_str(&serialized).expect("deserialize");
    assert_eq!(policy, deserialized);
}

#[test]
fn test_vector_label_serialization_snake_case() {
    let c = classification(
        RequestIntent::Debug,
        vec![RequestRisk::HighStakes, RequestRisk::DestructiveCommand],
        RequestArtifactType::Code,
    );
    let policy = derive_orchestration_policy(&c, "fix the bug", false);

    let ctx_json: Vec<&str> = policy.context_sources.iter().map(|s| s.as_str()).collect();
    assert!(ctx_json
        .iter()
        .all(|s| s.chars().all(|c| c.is_lowercase() || c == '_')));

    let allowed_json: Vec<&str> = policy.allowed_tools.iter().map(|t| t.as_str()).collect();
    assert!(allowed_json
        .iter()
        .all(|s| s.chars().all(|c| c.is_lowercase() || c == '_')));

    let risk_json: Vec<&str> = policy.risk_policy.iter().map(|r| r.as_str()).collect();
    assert!(risk_json
        .iter()
        .all(|s| s.chars().all(|c| c.is_lowercase() || c == '_')));

    let scope_json: Vec<&str> = policy.scope_policy.iter().map(|s| s.as_str()).collect();
    assert!(scope_json
        .iter()
        .all(|s| s.chars().all(|c| c.is_lowercase() || c == '_')));

    assert!(allowed_json.contains(&"file_read"));
    assert!(allowed_json.contains(&"repo_read"));
    assert!(risk_json.contains(&"high_stakes_guardrail"));
    assert!(risk_json.contains(&"destructive_requires_confirmation"));
    assert!(scope_json.contains(&"no_scp"));
}

#[test]
fn test_scalar_as_str_labels_snake_case() {
    let c = classification(
        RequestIntent::Debug,
        vec![RequestRisk::None],
        RequestArtifactType::Code,
    );
    let policy = derive_orchestration_policy(&c, "fix the bug", false);

    assert_eq!(policy.edit_policy.as_str(), "single_file_edit");
    assert_eq!(policy.validation_policy.as_str(), "targeted_tests");
    assert_eq!(policy.git_policy.as_str(), "no_git_changes");
    assert_eq!(policy.runtime_policy.as_str(), "no_restart");
    assert_eq!(policy.prompt_refinement_policy.as_str(), "none");
}

#[test]
fn test_migration_has_expected_structure() {
    let migration = include_str!("../migrations/V16__orchestration_policies.sql");

    assert!(
        migration.contains("agent_orchestration_policies"),
        "migration must reference agent_orchestration_policies"
    );
    assert!(
        migration.contains("policy_schema_version"),
        "must have policy_schema_version"
    );
    assert!(
        migration.contains("prompt_refinement_policy"),
        "must have prompt_refinement_policy"
    );
    assert!(
        migration.contains("context_sources"),
        "must have context_sources"
    );
    assert!(
        migration.contains("allowed_tools"),
        "must have allowed_tools"
    );
    assert!(
        migration.contains("required_tools"),
        "must have required_tools"
    );
    assert!(
        migration.contains("blocked_tools"),
        "must have blocked_tools"
    );
    assert!(migration.contains("risk_policy"), "must have risk_policy");
    assert!(migration.contains("scope_policy"), "must have scope_policy");
    assert!(
        migration.contains("JSONB"),
        "must use JSONB for array columns"
    );
    assert!(
        migration.contains("session_id"),
        "must have session_id index"
    );
    assert!(migration.contains("repo"), "must have repo index");
    assert!(migration.contains("event_id"), "must have event_id index");
    assert!(migration.contains("intent"), "must have intent index");
    assert!(
        migration.contains("prompt_refinement_policy"),
        "must have prompt_refinement index"
    );
}

#[test]
fn test_compact_policy_metadata() {
    let c = classification(
        RequestIntent::Debug,
        vec![RequestRisk::None],
        RequestArtifactType::Code,
    );
    let policy = derive_orchestration_policy(&c, "fix the bug", false);

    let meta = compact_policy_metadata(&c, &policy);

    assert!(meta["intent"].is_string());
    assert!(meta["recommended_route"].is_string());
    assert!(meta["edit_policy"].is_string());
    assert!(meta["validation_policy"].is_string());
    assert!(meta["git_policy"].is_string());
    assert!(meta["runtime_policy"].is_string());
    assert!(meta["prompt_refinement_policy"].is_string());

    assert!(meta["context_sources"].is_array());
    assert!(meta["allowed_tools"].is_array());
    assert!(meta["required_tools"].is_array());
    assert!(meta["blocked_tools"].is_array());
    assert!(meta["scope_policy"].is_array());
    assert!(meta["risk_policy"].is_array());

    assert_eq!(meta["policy_schema_version"], 1);
    assert_eq!(meta["source"], "deterministic_rules");

    let allowed: Vec<&str> = meta["allowed_tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(allowed.contains(&"repo_read"));

    let scope: Vec<&str> = meta["scope_policy"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(scope.contains(&"no_scp"));
}
