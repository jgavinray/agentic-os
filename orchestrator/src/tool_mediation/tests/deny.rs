use super::test_support::{policy_with_tools, tool_request};
use super::*;
use crate::orchestration_policy::ToolCapability as PolicyCap;
use serde_json::json;

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
    let policy = policy_with_tools(vec![], vec![PolicyCap::ShellMutation]);
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
