use super::test_support::{policy_with_tools, tool_request};
use super::*;
use crate::orchestration_policy::{EditPolicy, ToolCapability as PolicyCap};
use serde_json::json;

fn explicit_file_only_policy() -> crate::orchestration_policy::OrchestrationPolicy {
    let mut policy = policy_with_tools(vec![PolicyCap::FileEdit, PolicyCap::RepoRead], vec![]);
    policy.edit_policy = EditPolicy::ExplicitFileOnly;
    policy
}

#[test]
fn explicit_file_only_denies_edit_outside_named_file() {
    let policy = explicit_file_only_policy();
    let req = tool_request(
        "Update docs/config.md to describe the new flag",
        "Edit",
        json!({"file_path": "src/main.rs", "old_string": "a", "new_string": "b"}),
        vec!["Edit", "Read"],
    );

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    assert_eq!(response.decision, "deny");
    assert_eq!(response.reason, "edit_scope_violation");
    assert_eq!(response.capability, "file_edit");
    assert!(
        response.message.contains("src/main.rs"),
        "message must name the rejected path: {}",
        response.message
    );
}

#[test]
fn explicit_file_only_allows_edit_of_named_file() {
    let policy = explicit_file_only_policy();
    let req = tool_request(
        "Update docs/config.md to describe the new flag",
        "Edit",
        json!({"file_path": "docs/config.md", "old_string": "a", "new_string": "b"}),
        vec!["Edit", "Read"],
    );

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    assert_eq!(response.decision, "allow");
}

#[test]
fn single_file_edit_enforces_named_file_scope() {
    let mut policy = policy_with_tools(vec![PolicyCap::FileEdit, PolicyCap::RepoRead], vec![]);
    policy.edit_policy = EditPolicy::SingleFileEdit;
    let req = tool_request(
        "Fix the off-by-one in token_limits.rs",
        "Write",
        json!({"file_path": "src/handlers/usage.rs", "content": "x"}),
        vec!["Write", "Read"],
    );

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    assert_eq!(response.decision, "deny");
    assert_eq!(response.reason, "edit_scope_violation");
}

#[test]
fn edit_scope_enforcement_skips_when_request_names_no_files() {
    let policy = explicit_file_only_policy();
    let req = tool_request(
        "fix the typo in the greeting",
        "Edit",
        json!({"file_path": "src/main.rs", "old_string": "helo", "new_string": "hello"}),
        vec!["Edit", "Read"],
    );

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    assert_eq!(
        response.decision, "allow",
        "no named files means no scope evidence to enforce"
    );
}

#[test]
fn policy_blocks_file_read_denies_read() {
    let policy = policy_with_tools(vec![], vec![PolicyCap::FileRead]);
    let req = tool_request("read README.md", "Read", json!({}), vec!["Read"]);

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
    let policy = policy_with_tools(vec![], vec![]);
    let req = tool_request(
        "run a command",
        "Bash",
        json!({"command": "echo hello"}),
        vec!["Bash"],
    );

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    assert_eq!(response.decision, "deny");
    assert_eq!(response.reason, "policy_tool_not_allowed");
    assert_eq!(response.capability, "shell");
    assert_eq!(response.attempted_tool, "Bash");
    assert_eq!(response.preferred_tool, None);
    assert_eq!(
        response.message,
        "Tool call is not allowed by orchestration policy."
    );
}

#[test]
fn policy_shell_read_does_not_allow_generic_bash() {
    let policy = policy_with_tools(vec![PolicyCap::ShellRead], vec![]);
    let req = tool_request(
        "run a command",
        "Bash",
        json!({"command": "echo hello"}),
        vec!["Bash"],
    );

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
fn policy_blocks_shell_mutation_denies_mutating_bash() {
    let policy = policy_with_tools(vec![PolicyCap::ShellRead], vec![PolicyCap::ShellMutation]);
    let req = tool_request(
        "remove a file",
        "Bash",
        json!({"command": "rm -rf /tmp/example"}),
        vec!["Bash"],
    );

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
    let policy = policy_with_tools(vec![PolicyCap::ShellRead], vec![]);
    let req = tool_request(
        "remove a file",
        "Bash",
        json!({"command": "rm -rf /tmp/example"}),
        vec!["Bash"],
    );

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
