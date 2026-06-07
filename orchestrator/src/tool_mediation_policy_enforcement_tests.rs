use super::*;
use serde_json::json;

#[test]
fn policy_blocks_file_read_denies_read() {
    use crate::orchestration_policy::{
        EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
    };

    let policy = crate::orchestration_policy::OrchestrationPolicy {
        context_sources: vec![],
        allowed_tools: vec![],
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

    let req = ToolAuthorizeRequest {
        session_id: None,
        repo: None,
        task: None,
        trajectory_id: None,
        attempt_index: None,
        parent_event_id: None,
        user_intent: Some("read README.md".to_string()),
        tool_name: "Read".to_string(),
        arguments: json!({}),
        available_tools: vec!["Read".to_string()],
    };

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    assert_eq!(response.decision, "deny");
    assert_eq!(response.reason, "policy_blocked_tool");
    assert_eq!(response.capability, "file_read");
    assert_eq!(response.attempted_tool, "Read");
    assert_eq!(response.preferred_tool, None);
    assert_eq!(
        response.message,
        "Tool call denied by orchestration policy."
    );
}

#[test]
fn policy_not_allowing_shell_read_denies_bash() {
    use crate::orchestration_policy::{EditPolicy, GitPolicy, RuntimePolicy};

    // Empty allowed_tools and empty blocked_tools means nothing is allowed.
    let policy = crate::orchestration_policy::OrchestrationPolicy {
        context_sources: vec![],
        allowed_tools: vec![],
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

    let req = ToolAuthorizeRequest {
        session_id: None,
        repo: None,
        task: None,
        trajectory_id: None,
        attempt_index: None,
        parent_event_id: None,
        user_intent: Some("run a command".to_string()),
        tool_name: "Bash".to_string(),
        arguments: json!({"command": "echo hello"}),
        available_tools: vec!["Bash".to_string()],
    };

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    assert_eq!(response.decision, "deny");
    assert_eq!(response.reason, "policy_tool_not_allowed");
    assert_eq!(response.capability, "shell_mutation");
    assert_eq!(response.attempted_tool, "Bash");
    assert_eq!(response.preferred_tool, None);
    assert_eq!(
        response.message,
        "Tool call is not allowed by orchestration policy."
    );
}

#[test]
fn policy_shell_read_does_not_allow_generic_bash() {
    use crate::orchestration_policy::{
        EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
    };

    let policy = crate::orchestration_policy::OrchestrationPolicy {
        context_sources: vec![],
        allowed_tools: vec![PolicyCap::ShellRead],
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

    let req = ToolAuthorizeRequest {
        session_id: None,
        repo: None,
        task: None,
        trajectory_id: None,
        attempt_index: None,
        parent_event_id: None,
        user_intent: Some("run a command".to_string()),
        tool_name: "Bash".to_string(),
        arguments: json!({"command": "echo hello"}),
        available_tools: vec!["Bash".to_string()],
    };

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    assert_eq!(response.decision, "deny");
    assert_eq!(response.reason, "policy_tool_not_allowed");
    assert_eq!(response.capability, "shell_mutation");
    assert_eq!(response.attempted_tool, "Bash");
    assert_eq!(response.preferred_tool, None);
    assert_eq!(
        response.message,
        "Tool call is not allowed by orchestration policy."
    );
}

#[test]
fn policy_allows_file_read_but_canonical_read_exists() {
    use crate::orchestration_policy::{
        EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
    };

    let policy = crate::orchestration_policy::OrchestrationPolicy {
        context_sources: vec![],
        allowed_tools: vec![PolicyCap::FileRead],
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

    let req = ToolAuthorizeRequest {
        session_id: None,
        repo: None,
        task: None,
        trajectory_id: None,
        attempt_index: None,
        parent_event_id: None,
        user_intent: Some("read README.md".to_string()),
        tool_name: "Bash".to_string(),
        arguments: json!({"command": "cat README.md"}),
        available_tools: vec!["Read".to_string(), "Bash".to_string()],
    };

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    // Policy allows FileRead, but canonical-tool preference still applies.
    assert_eq!(response.decision, "deny");
    assert_eq!(response.reason, "prefer_canonical_tool");
    assert_eq!(response.capability, "file_read");
    assert_eq!(response.preferred_tool.as_deref(), Some("Read"));
}

#[test]
fn policy_blocks_shell_mutation_denies_mutating_bash() {
    use crate::orchestration_policy::{
        EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
    };

    let policy = crate::orchestration_policy::OrchestrationPolicy {
        context_sources: vec![],
        allowed_tools: vec![],
        required_tools: vec![],
        blocked_tools: vec![PolicyCap::ShellMutation],
        edit_policy: EditPolicy::ReadOnly,
        validation_policy: crate::orchestration_policy::ValidationPolicy::None,
        git_policy: GitPolicy::NoGitChanges,
        runtime_policy: RuntimePolicy::NoRestart,
        scope_policy: vec![],
        prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
        risk_policy: vec![],
    };

    let req = ToolAuthorizeRequest {
        session_id: None,
        repo: None,
        task: None,
        trajectory_id: None,
        attempt_index: None,
        parent_event_id: None,
        user_intent: Some("remove a file".to_string()),
        tool_name: "Bash".to_string(),
        arguments: json!({"command": "rm -rf /tmp/example"}),
        available_tools: vec!["Bash".to_string()],
    };

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    assert_eq!(response.decision, "deny");
    assert_eq!(response.reason, "policy_blocked_tool");
    assert_eq!(response.capability, "shell_mutation");
    assert_eq!(response.attempted_tool, "Bash");
    assert_eq!(response.preferred_tool, None);
    assert_eq!(
        response.message,
        "Tool call denied by orchestration policy."
    );
}

#[test]
fn policy_shell_read_only_does_not_allow_mutating_bash() {
    use crate::orchestration_policy::{
        EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
    };

    let policy = crate::orchestration_policy::OrchestrationPolicy {
        context_sources: vec![],
        allowed_tools: vec![PolicyCap::ShellRead],
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

    let req = ToolAuthorizeRequest {
        session_id: None,
        repo: None,
        task: None,
        trajectory_id: None,
        attempt_index: None,
        parent_event_id: None,
        user_intent: Some("remove a file".to_string()),
        tool_name: "Bash".to_string(),
        arguments: json!({"command": "rm -rf /tmp/example"}),
        available_tools: vec!["Bash".to_string()],
    };

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    assert_eq!(response.decision, "deny");
    assert_eq!(response.reason, "policy_tool_not_allowed");
    assert_eq!(response.capability, "shell_mutation");
    assert_eq!(response.attempted_tool, "Bash");
    assert_eq!(response.preferred_tool, None);
    assert_eq!(
        response.message,
        "Tool call is not allowed by orchestration policy."
    );
}

#[test]
fn policy_repo_read_allows_rg_shell_as_text_search() {
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

    let req = ToolAuthorizeRequest {
        session_id: None,
        repo: None,
        task: None,
        trajectory_id: None,
        attempt_index: None,
        parent_event_id: None,
        user_intent: Some("search for pattern".to_string()),
        tool_name: "Bash".to_string(),
        arguments: json!({"command": "rg pattern src"}),
        available_tools: vec!["Bash".to_string()],
    };

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    assert_eq!(response.decision, "allow");
    assert_eq!(response.capability, "text_search");
    assert_ne!(response.capability, "shell_mutation");
}
