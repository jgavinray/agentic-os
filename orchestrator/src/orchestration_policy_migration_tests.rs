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
