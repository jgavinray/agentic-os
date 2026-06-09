use super::*;

#[test]
fn policy_allows_file_read_when_repo_read_allowed() {
    use crate::orchestration_policy::{
        EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
    };

    let policy = crate::orchestration_policy::OrchestrationPolicy {
        context_sources: vec![],
        allowed_tools: vec![PolicyCap::RepoRead],
        required_tools: vec![],
        blocked_tools: vec![],
        edit_policy: EditPolicy::ReadOnly,
        validation_policy: crate::orchestration_policy::ValidationPolicy::None,
        git_policy: GitPolicy::NoGitChanges,
        runtime_policy: RuntimePolicy::NoRestart,
        scope_policy: vec![],
        prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
        risk_policy: vec![],
    };

    assert!(policy_allows_tool_capability(
        &policy,
        ToolCapability::FileRead
    ));
}

#[test]
fn policy_blocks_text_search_when_repo_read_blocked() {
    use crate::orchestration_policy::{
        EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
    };

    let policy = crate::orchestration_policy::OrchestrationPolicy {
        context_sources: vec![],
        allowed_tools: vec![],
        required_tools: vec![],
        blocked_tools: vec![PolicyCap::RepoRead],
        edit_policy: EditPolicy::ReadOnly,
        validation_policy: crate::orchestration_policy::ValidationPolicy::None,
        git_policy: GitPolicy::NoGitChanges,
        runtime_policy: RuntimePolicy::NoRestart,
        scope_policy: vec![],
        prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
        risk_policy: vec![],
    };

    assert!(policy_blocks_tool_capability(
        &policy,
        ToolCapability::TextSearch
    ));
}

#[test]
fn policy_maps_validation_to_named_validation_not_shell_read() {
    use crate::orchestration_policy::{
        EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
    };

    let policy = crate::orchestration_policy::OrchestrationPolicy {
        context_sources: vec![],
        allowed_tools: vec![PolicyCap::Validation],
        required_tools: vec![],
        blocked_tools: vec![],
        edit_policy: EditPolicy::ReadOnly,
        validation_policy: crate::orchestration_policy::ValidationPolicy::None,
        git_policy: GitPolicy::NoGitChanges,
        runtime_policy: RuntimePolicy::NoRestart,
        scope_policy: vec![],
        prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
        risk_policy: vec![],
    };

    assert!(policy_allows_tool_capability(
        &policy,
        ToolCapability::Validation
    ));
    assert!(!policy_allows_tool_capability(
        &policy,
        ToolCapability::Shell
    ));
}

#[test]
fn policy_maps_publishing_to_git_write() {
    use crate::orchestration_policy::{
        EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
    };

    let policy = crate::orchestration_policy::OrchestrationPolicy {
        context_sources: vec![],
        allowed_tools: vec![PolicyCap::GitWrite],
        required_tools: vec![],
        blocked_tools: vec![],
        edit_policy: EditPolicy::ReadOnly,
        validation_policy: crate::orchestration_policy::ValidationPolicy::None,
        git_policy: GitPolicy::NoGitChanges,
        runtime_policy: RuntimePolicy::NoRestart,
        scope_policy: vec![],
        prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
        risk_policy: vec![],
    };

    assert!(policy_allows_tool_capability(
        &policy,
        ToolCapability::Publishing
    ));
}

#[test]
fn policy_unknown_is_not_allowed_and_not_blocked() {
    use crate::orchestration_policy::{
        EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
    };

    let policy = crate::orchestration_policy::OrchestrationPolicy {
        context_sources: vec![],
        allowed_tools: vec![PolicyCap::FileRead],
        required_tools: vec![],
        blocked_tools: vec![PolicyCap::FileRead],
        edit_policy: EditPolicy::ReadOnly,
        validation_policy: crate::orchestration_policy::ValidationPolicy::None,
        git_policy: GitPolicy::NoGitChanges,
        runtime_policy: RuntimePolicy::NoRestart,
        scope_policy: vec![],
        prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
        risk_policy: vec![],
    };

    assert!(!policy_allows_tool_capability(
        &policy,
        ToolCapability::Unknown
    ));
    assert!(!policy_blocks_tool_capability(
        &policy,
        ToolCapability::Unknown
    ));
}
