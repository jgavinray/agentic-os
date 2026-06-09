use super::test_support::{policy_with_tools, tool_request};
use super::*;
use crate::orchestration_policy::ToolCapability as PolicyCap;
use serde_json::json;

#[test]
fn policy_allows_file_read_but_canonical_read_exists() {
    let policy = policy_with_tools(vec![PolicyCap::FileRead], vec![]);
    let req = tool_request(
        "read README.md",
        "Bash",
        json!({"command": "cat README.md"}),
        vec!["Read", "Bash"],
    );

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    assert_eq!(response.decision, "deny");
    assert_eq!(response.reason, "policy_tool_not_allowed");
    assert_eq!(response.capability, "shell");
    assert_eq!(response.preferred_tool, None);
}

#[test]
fn policy_repo_read_allows_named_rg_as_text_search() {
    let policy = policy_with_tools(vec![PolicyCap::RepoRead], vec![]);
    let req = tool_request(
        "search for pattern",
        "rg",
        json!({"pattern": "pattern", "path": "src"}),
        vec!["rg"],
    );

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    assert_eq!(response.decision, "allow");
    assert_eq!(response.capability, "text_search");
    assert_ne!(response.capability, "shell_mutation");
}

#[test]
fn policy_repo_read_does_not_allow_bash_wrapped_rg() {
    let policy = policy_with_tools(vec![PolicyCap::RepoRead], vec![]);
    let req = tool_request(
        "search for pattern",
        "Bash",
        json!({"command": "rg pattern src"}),
        vec!["Bash", "rg"],
    );

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    assert_eq!(response.decision, "deny");
    assert_eq!(response.reason, "policy_tool_not_allowed");
    assert_eq!(response.capability, "shell");
}
