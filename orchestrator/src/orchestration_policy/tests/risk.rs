use super::test_support::classification;
use super::*;
use crate::request_classification::{RequestArtifactType, RequestIntent, RequestRisk};

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
fn test_debug_code_high_stakes_blocks_file_edit() {
    let c = classification(
        RequestIntent::Debug,
        vec![RequestRisk::HighStakes],
        RequestArtifactType::Code,
    );
    let policy = derive_orchestration_policy(&c, "fix the bug in main.rs", false);

    assert!(
        policy.blocked_tools.contains(&ToolCapability::FileEdit),
        "file_edit must be blocked under HighStakes"
    );
    assert!(
        !policy.allowed_tools.contains(&ToolCapability::FileEdit),
        "file_edit must not be allowed when blocked"
    );
    assert!(
        !policy.required_tools.contains(&ToolCapability::FileEdit),
        "file_edit must not be required when blocked"
    );
}

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
