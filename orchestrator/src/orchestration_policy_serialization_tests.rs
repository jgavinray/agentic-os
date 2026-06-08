use super::*;

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
