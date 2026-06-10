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

fn implementation_policy() -> crate::orchestration_policy::OrchestrationPolicy {
    let text = "Implement the change in src/module.rs";
    let classification = crate::request_classification::classify_request_text(
        "agentic-os",
        "session-t",
        text,
        None,
        "user_message",
    );
    crate::orchestration_policy::derive_orchestration_policy(&classification, text, false)
}

#[test]
fn implementation_policy_allows_bash_validation_commands() {
    let policy = implementation_policy();
    for command in [
        "cargo test",
        "cargo test -p orchestrator",
        "cargo build",
        "cargo check --all-targets",
        "cargo clippy -- -D warnings",
        "go test ./...",
        "npm test",
        "pytest tests/",
        "make test",
    ] {
        let req = tool_request(
            "implement and validate the change",
            "Bash",
            json!({"command": command}),
            vec!["Read", "Grep", "Edit", "Write", "Bash"],
        );

        let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

        assert_eq!(response.decision, "allow", "command: {command}");
        assert_eq!(response.capability, "validation", "command: {command}");
    }
}

#[test]
fn implementation_policy_denies_bash_mutation_commands() {
    let policy = implementation_policy();
    for command in [
        "rm -rf target",
        "git push origin main",
        "curl -X POST http://x",
    ] {
        let req = tool_request(
            "implement the change",
            "Bash",
            json!({"command": command}),
            vec!["Read", "Grep", "Edit", "Write", "Bash"],
        );

        let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

        assert_eq!(response.decision, "deny", "command: {command}");
        assert_eq!(response.capability, "shell_mutation", "command: {command}");
        assert_eq!(response.reason, "policy_blocked_tool", "command: {command}");
    }
}

#[test]
fn implementation_policy_steers_bash_reads_to_canonical_tools() {
    let policy = implementation_policy();
    let req = tool_request(
        "implement the change",
        "Bash",
        json!({"command": "cat README.md"}),
        vec!["Read", "Grep", "Edit", "Write", "Bash"],
    );

    let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

    assert_eq!(response.decision, "deny");
    assert_eq!(response.reason, "prefer_canonical_tool");
    assert_eq!(response.preferred_tool.as_deref(), Some("Read"));
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
