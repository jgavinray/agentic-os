use super::test_support::classification;
use super::*;
use crate::request_classification::{RequestArtifactType, RequestIntent, RequestRisk};

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
    assert!(policy.allowed_tools.contains(&ToolCapability::Validation));
    assert_eq!(policy.validation_policy, ValidationPolicy::TargetedTests);
    assert_eq!(policy.git_policy, GitPolicy::NoGitChanges);
    assert_eq!(policy.runtime_policy, RuntimePolicy::NoRestart);
    assert!(
        policy.allowed_tools.contains(&ToolCapability::ShellRead),
        "implement must expose read/validation shell so clients without a \
         dedicated test tool can satisfy targeted_tests"
    );
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

#[test]
fn test_summarize_external_info_requires_and_allows_web_search() {
    let c = classification(
        RequestIntent::Summarize,
        vec![RequestRisk::ExternalCurrentInfoRequired],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(&c, "summarize the latest release", false);

    assert!(
        policy.allowed_tools.contains(&ToolCapability::WebSearch),
        "web_search must be allowed for Summarize+ExternalCurrentInfoRequired"
    );
    assert!(
        policy.required_tools.contains(&ToolCapability::WebSearch),
        "web_search must be required for Summarize+ExternalCurrentInfoRequired"
    );
}
