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

// 1. Explain allows web_search, read_only, no_git_changes
#[test]
fn test_explain_allows_web_search_read_only_no_git() {
    let c = classification(
        RequestIntent::Explain,
        vec![RequestRisk::None],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(&c, "explain this", false);

    assert!(policy.allowed_tools.contains(&ToolCapability::WebSearch));
    assert_eq!(policy.edit_policy, EditPolicy::ReadOnly);
    assert_eq!(policy.git_policy, GitPolicy::NoGitChanges);
}

// 2. Explain + ExternalCurrentInfoRequired requires web_search
#[test]
fn test_explain_external_info_requires_web_search() {
    let c = classification(
        RequestIntent::Explain,
        vec![RequestRisk::ExternalCurrentInfoRequired],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(&c, "explain this", false);

    assert!(policy.required_tools.contains(&ToolCapability::WebSearch));
}

// 3. Debug Code => single_file_edit, file_edit allowed, targeted_tests
#[test]
fn test_debug_code_single_file_edit() {
    let c = classification(
        RequestIntent::Debug,
        vec![RequestRisk::None],
        RequestArtifactType::Code,
    );
    let policy = derive_orchestration_policy(&c, "fix the bug in main.rs", false);

    assert_eq!(policy.edit_policy, EditPolicy::SingleFileEdit);
    assert!(policy.allowed_tools.contains(&ToolCapability::FileEdit));
    assert_eq!(policy.validation_policy, ValidationPolicy::TargetedTests);
}

// 4. ModifyConfig Code => build
#[test]
fn test_modify_config_code_build() {
    let c = classification(
        RequestIntent::ModifyConfig,
        vec![RequestRisk::None],
        RequestArtifactType::Code,
    );
    let policy = derive_orchestration_policy(&c, "edit the config", false);

    assert_eq!(policy.validation_policy, ValidationPolicy::Build);
}

// 5. ModifyConfig Yaml => targeted_tests
#[test]
fn test_modify_config_yaml_targeted_tests() {
    let c = classification(
        RequestIntent::ModifyConfig,
        vec![RequestRisk::None],
        RequestArtifactType::Yaml,
    );
    let policy = derive_orchestration_policy(&c, "edit the yaml config", false);

    assert_eq!(policy.validation_policy, ValidationPolicy::TargetedTests);
}

// 5b. ModifyConfig Json => targeted_tests
#[test]
fn test_modify_config_json_targeted_tests() {
    let c = classification(
        RequestIntent::ModifyConfig,
        vec![RequestRisk::None],
        RequestArtifactType::Json,
    );
    let policy = derive_orchestration_policy(&c, "edit the json config", false);

    assert_eq!(policy.validation_policy, ValidationPolicy::TargetedTests);
}

// 6. DestructiveCommand blocks shell_mutation, deploy, restart_service
#[test]
fn test_destructive_command_blocks_mutation() {
    let c = classification(
        RequestIntent::Explain,
        vec![RequestRisk::DestructiveCommand],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(&c, "run rm -rf /tmp", false);

    assert!(policy
        .blocked_tools
        .contains(&ToolCapability::ShellMutation));
    assert!(policy.blocked_tools.contains(&ToolCapability::Deploy));
    assert!(policy
        .blocked_tools
        .contains(&ToolCapability::RestartService));
}

// 7. HighStakes adds high_stakes_guardrail and blocks mutations
#[test]
fn test_high_stakes_guardrail_and_blocks() {
    let c = classification(
        RequestIntent::Explain,
        vec![RequestRisk::HighStakes],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(&c, "explain this medical issue", false);

    assert!(policy
        .risk_policy
        .contains(&RiskPolicy::HighStakesGuardrail));
    assert!(policy.blocked_tools.contains(&ToolCapability::FileEdit));
    assert!(policy
        .blocked_tools
        .contains(&ToolCapability::ShellMutation));
    assert!(policy
        .blocked_tools
        .contains(&ToolCapability::DockerMutation));
    assert!(policy.blocked_tools.contains(&ToolCapability::Deploy));
    assert!(policy
        .blocked_tools
        .contains(&ToolCapability::RestartService));
    assert!(policy.blocked_tools.contains(&ToolCapability::GitWrite));
}

#[test]
fn test_implement_allows_edit_create_and_targeted_validation_only() {
    let c = classification(
        RequestIntent::Implement,
        vec![RequestRisk::None],
        RequestArtifactType::Code,
    );
    let policy = derive_orchestration_policy(&c, "implement the request classifier change", false);

    assert!(policy.allowed_tools.contains(&ToolCapability::RepoRead));
    assert!(policy.allowed_tools.contains(&ToolCapability::FileRead));
    assert!(policy.allowed_tools.contains(&ToolCapability::FileEdit));
    assert_eq!(policy.validation_policy, ValidationPolicy::TargetedTests);
    assert_eq!(policy.git_policy, GitPolicy::NoGitChanges);
    assert_eq!(policy.runtime_policy, RuntimePolicy::NoRestart);
    assert!(policy.blocked_tools.contains(&ToolCapability::ShellRead));
    assert!(policy
        .blocked_tools
        .contains(&ToolCapability::ShellMutation));
    assert!(policy.blocked_tools.contains(&ToolCapability::GitWrite));
    assert!(policy.blocked_tools.contains(&ToolCapability::Deploy));
    assert!(policy
        .blocked_tools
        .contains(&ToolCapability::RestartService));
    assert!(policy
        .scope_policy
        .contains(&ScopePolicy::IgnoreUnrelatedDirtyChanges));
}

// 8. Unknown minimal read-only empty allowed_tools
#[test]
fn test_unknown_minimal() {
    let c = classification(
        RequestIntent::Unknown,
        vec![RequestRisk::None],
        RequestArtifactType::Unknown,
    );
    let policy = derive_orchestration_policy(&c, "???", false);

    assert!(policy.allowed_tools.is_empty());
    assert_eq!(policy.edit_policy, EditPolicy::ReadOnly);
}

// 9. NoScp always present
#[test]
fn test_no_scp_always_present() {
    for intent in [
        RequestIntent::Explain,
        RequestIntent::Debug,
        RequestIntent::Implement,
        RequestIntent::ModifyConfig,
        RequestIntent::GenerateConfig,
        RequestIntent::OperateTool,
        RequestIntent::Plan,
        RequestIntent::Summarize,
        RequestIntent::Classify,
        RequestIntent::Search,
        RequestIntent::Unknown,
    ] {
        let c = classification(
            intent,
            vec![RequestRisk::None],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "test", false);
        assert!(
            policy.scope_policy.contains(&ScopePolicy::NoScp),
            "{intent:?} must include no_scp"
        );
    }
}

// 11. prompt/spec review => multi_pass_review, read_only, narrowed tools
#[test]
fn test_prompt_review_overlay() {
    let c = classification(
        RequestIntent::Explain,
        vec![RequestRisk::None],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(
        &c,
        "Review the prompt spec and rewrite it — is this good?",
        false,
    );

    assert_eq!(
        policy.prompt_refinement_policy,
        PromptRefinementPolicy::MultiPassReview
    );
    assert_eq!(policy.edit_policy, EditPolicy::ReadOnly);
    assert_eq!(policy.git_policy, GitPolicy::NoGitChanges);
    assert_eq!(policy.runtime_policy, RuntimePolicy::NoRestart);
    // narrowed to repo_read only (no ExternalCurrentInfoRequired)
    assert!(policy.allowed_tools.contains(&ToolCapability::RepoRead));
    assert!(
        !policy.allowed_tools.contains(&ToolCapability::WebSearch),
        "web_search should not be present without external_web_required"
    );
}

#[test]
fn test_prompt_review_overlay_with_external_info() {
    let c = classification(
        RequestIntent::Explain,
        vec![RequestRisk::ExternalCurrentInfoRequired],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(
        &c,
        "Review the prompt spec and rewrite it — is this good?",
        false,
    );

    assert_eq!(
        policy.prompt_refinement_policy,
        PromptRefinementPolicy::MultiPassReview
    );
    // web_search should be present because ExternalCurrentInfoRequired adds it
    assert!(policy.allowed_tools.contains(&ToolCapability::WebSearch));
}

// -----------------------------------------------------------------------
// Phase 3 — blocked_tools authority & required⊆allowed invariants
// -----------------------------------------------------------------------

// 1. Summarize + ExternalCurrentInfoRequired => web_search is both allowed and required.
#[test]
fn test_summarize_external_info_requires_and_allows_web_search() {
    let c = classification(
        RequestIntent::Summarize,
        vec![RequestRisk::ExternalCurrentInfoRequired],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(&c, "summarize the latest release", false);

    // web_search must be in allowed_tools
    assert!(
        policy.allowed_tools.contains(&ToolCapability::WebSearch),
        "web_search must be allowed for Summarize+ExternalCurrentInfoRequired"
    );
    // web_search must be in required_tools
    assert!(
        policy.required_tools.contains(&ToolCapability::WebSearch),
        "web_search must be required for Summarize+ExternalCurrentInfoRequired"
    );
}

// 2. Debug + Code + HighStakes => file_edit is blocked and not allowed.
#[test]
fn test_debug_code_high_stakes_blocks_file_edit() {
    let c = classification(
        RequestIntent::Debug,
        vec![RequestRisk::HighStakes],
        RequestArtifactType::Code,
    );
    let policy = derive_orchestration_policy(&c, "fix the bug in main.rs", false);

    // file_edit must be blocked
    assert!(
        policy.blocked_tools.contains(&ToolCapability::FileEdit),
        "file_edit must be blocked under HighStakes"
    );
    // file_edit must NOT be in allowed_tools
    assert!(
        !policy.allowed_tools.contains(&ToolCapability::FileEdit),
        "file_edit must not be allowed when blocked"
    );
    // file_edit must NOT be in required_tools
    assert!(
        !policy.required_tools.contains(&ToolCapability::FileEdit),
        "file_edit must not be required when blocked"
    );
}

// 3. DestructiveCommand on OperateTool => shell_mutation/deploy/restart_service blocked and absent.
#[test]
fn test_destructive_command_operate_tool_blocks_mutation_tools() {
    let c = classification(
        RequestIntent::OperateTool,
        vec![RequestRisk::DestructiveCommand],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(&c, "run rm -rf /tmp", false);

    let blocked_caps = [
        ToolCapability::ShellMutation,
        ToolCapability::Deploy,
        ToolCapability::RestartService,
    ];

    for cap in &blocked_caps {
        assert!(
            policy.blocked_tools.contains(cap),
            "{cap:?} must be blocked under DestructiveCommand"
        );
        assert!(
            !policy.allowed_tools.contains(cap),
            "{cap:?} must not be allowed when blocked"
        );
        assert!(
            !policy.required_tools.contains(cap),
            "{cap:?} must not be required when blocked"
        );
    }
}
