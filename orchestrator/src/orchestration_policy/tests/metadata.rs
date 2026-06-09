use super::*;

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
