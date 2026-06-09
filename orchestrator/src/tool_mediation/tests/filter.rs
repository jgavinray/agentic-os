use super::test_support::policy_with_tools;
use super::*;
use serde_json::json;

#[test]
fn openai_bash_only_with_empty_policy_hides_all() {
    let policy = policy_with_tools(vec![], vec![]);
    let mut req = json!({
        "messages": [{"role": "user", "content": "run a command"}],
        "tools": [
            {"type": "function", "function": {"name": "Bash"}}
        ],
        "tool_choice": {"type": "function", "function": {"name": "Bash"}}
    });

    let outcome = shape_openai_request_with_policy(&mut req, "run a command", Some(&policy));

    assert_eq!(outcome.decision, "shape");
    assert_eq!(outcome.reason, "policy_filtered_all_tools");
    assert_eq!(req["tools"].as_array().unwrap().len(), 0);
    assert!(outcome.allowed_tools.is_empty());
    assert_eq!(outcome.hidden_tools.len(), 1);
    assert_eq!(outcome.hidden_tools[0].name, "Bash");
}

#[test]
fn openai_bash_only_with_empty_policy_tool_choice_auto() {
    let policy = policy_with_tools(vec![], vec![]);
    let mut req = json!({
        "messages": [{"role": "user", "content": "run a command"}],
        "tools": [
            {"type": "function", "function": {"name": "Bash"}}
        ],
        "tool_choice": {"type": "function", "function": {"name": "Bash"}}
    });

    let outcome = shape_openai_request_with_policy(&mut req, "run a command", Some(&policy));

    assert_eq!(req["tool_choice"], "auto");
    assert!(outcome.tool_choice_changed);
}

#[test]
fn anthropic_bash_only_with_empty_policy_hides_all() {
    let policy = policy_with_tools(vec![], vec![]);
    let mut req = json!({
        "messages": [],
        "tools": [
            {"name": "bash", "input_schema": {}}
        ],
        "tool_choice": {"type": "tool", "name": "bash"}
    });

    let outcome = shape_anthropic_request_with_policy(&mut req, "run a command", Some(&policy));

    assert_eq!(outcome.decision, "shape");
    assert_eq!(outcome.reason, "policy_filtered_all_tools");
    assert_eq!(req["tools"].as_array().unwrap().len(), 0);
    assert!(outcome.allowed_tools.is_empty());
    assert_eq!(outcome.hidden_tools.len(), 1);
    assert_eq!(outcome.hidden_tools[0].name, "bash");
}

// ---------------------------------------------------------------------------
// Implementation policy: concrete repo navigation / search tools preserved
// ---------------------------------------------------------------------------

fn implementation_policy() -> crate::orchestration_policy::OrchestrationPolicy {
    // Derive the real implementation policy from a representative request.
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
fn implementation_policy_preserves_read_glob_ls_list_grep_search_rg_edit_write_openai() {
    let policy = implementation_policy();
    let mut req = json!({
        "messages": [],
        "tools": [
            {"type": "function", "function": {"name": "Read"}},
            {"type": "function", "function": {"name": "Glob"}},
            {"type": "function", "function": {"name": "LS"}},
            {"type": "function", "function": {"name": "List"}},
            {"type": "function", "function": {"name": "Grep"}},
            {"type": "function", "function": {"name": "Search"}},
            {"type": "function", "function": {"name": "rg"}},
            {"type": "function", "function": {"name": "Edit"}},
            {"type": "function", "function": {"name": "Write"}},
            {"type": "function", "function": {"name": "run_tests"}}
        ]
    });

    let outcome = shape_openai_request_with_policy(&mut req, "Implement the change", Some(&policy));
    let allowed = req["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap())
        .collect::<Vec<_>>();

    // All concrete repo navigation and editing tools must be preserved.
    assert!(allowed.contains(&"Read"), "Read should be allowed");
    assert!(allowed.contains(&"Glob"), "Glob should be allowed");
    assert!(allowed.contains(&"LS"), "LS should be allowed");
    assert!(allowed.contains(&"List"), "List should be allowed");
    assert!(allowed.contains(&"Grep"), "Grep should be allowed");
    assert!(allowed.contains(&"Search"), "Search should be allowed");
    assert!(allowed.contains(&"rg"), "rg should be allowed");
    assert!(allowed.contains(&"Edit"), "Edit should be allowed");
    assert!(allowed.contains(&"Write"), "Write should be allowed");
    assert!(
        allowed.contains(&"run_tests"),
        "run_tests should be allowed"
    );
    assert_eq!(allowed.len(), 10, "all 10 tools should be allowed");
    assert!(outcome.hidden_tools.is_empty(), "no tools should be hidden");
}

#[test]
fn implementation_policy_preserves_read_glob_grep_search_edit_write_anthropic() {
    let policy = implementation_policy();
    let mut req = json!({
        "messages": [],
        "tools": [
            {"name": "Read", "input_schema": {}},
            {"name": "Glob", "input_schema": {}},
            {"name": "LS", "input_schema": {}},
            {"name": "List", "input_schema": {}},
            {"name": "Grep", "input_schema": {}},
            {"name": "Search", "input_schema": {}},
            {"name": "rg", "input_schema": {}},
            {"name": "Edit", "input_schema": {}},
            {"name": "Write", "input_schema": {}}
        ]
    });

    let outcome =
        shape_anthropic_request_with_policy(&mut req, "Implement the change", Some(&policy));
    let allowed = req["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect::<Vec<_>>();

    assert!(allowed.contains(&"Read"), "Read should be allowed");
    assert!(allowed.contains(&"Glob"), "Glob should be allowed");
    assert!(allowed.contains(&"LS"), "LS should be allowed");
    assert!(allowed.contains(&"List"), "List should be allowed");
    assert!(allowed.contains(&"Grep"), "Grep should be allowed");
    assert!(allowed.contains(&"Search"), "Search should be allowed");
    assert!(allowed.contains(&"rg"), "rg should be allowed");
    assert!(allowed.contains(&"Edit"), "Edit should be allowed");
    assert!(allowed.contains(&"Write"), "Write should be allowed");
    assert_eq!(allowed.len(), 9, "all 9 tools should be allowed");
    assert!(outcome.hidden_tools.is_empty(), "no tools should be hidden");
}

// ---------------------------------------------------------------------------
// Implementation policy: dangerous / broad tools hidden
// ---------------------------------------------------------------------------

#[test]
fn implementation_policy_hides_bash_shell_delete_publishing_deploy_unknown_openai() {
    let policy = implementation_policy();
    let mut req = json!({
        "messages": [],
        "tools": [
            {"type": "function", "function": {"name": "Read"}},
            {"type": "function", "function": {"name": "Grep"}},
            {"type": "function", "function": {"name": "Edit"}},
            {"type": "function", "function": {"name": "Write"}},
            {"type": "function", "function": {"name": "Bash"}},
            {"type": "function", "function": {"name": "Shell"}},
            {"type": "function", "function": {"name": "Delete"}},
            {"type": "function", "function": {"name": "MultiEdit"}},
            {"type": "function", "function": {"name": "CreatePR"}},
            {"type": "function", "function": {"name": "Push"}},
            {"type": "function", "function": {"name": "Deploy"}},
            {"type": "function", "function": {"name": "RestartService"}},
            {"type": "function", "function": {"name": "RunCommand"}},
            {"type": "function", "function": {"name": "ExecuteCommand"}},
            {"type": "function", "function": {"name": "Terminal"}},
            {"type": "function", "function": {"name": "mcp__plugin_ecc_github__create_or_update_file"}},
            {"type": "function", "function": {"name": "mcp__plugin_ecc_github__push_files"}},
            {"type": "function", "function": {"name": "mcp__plugin_ecc_playwright__browser_click"}}
        ]
    });

    let outcome = shape_openai_request_with_policy(&mut req, "Implement the change", Some(&policy));
    let allowed = req["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap())
        .collect::<Vec<_>>();
    let hidden = outcome
        .hidden_tools
        .iter()
        .map(|t| t.name.as_str())
        .collect::<Vec<_>>();

    // Concrete repo tools preserved.
    assert!(allowed.contains(&"Read"));
    assert!(allowed.contains(&"Grep"));
    assert!(allowed.contains(&"Edit"));
    assert!(allowed.contains(&"Write"));

    // Bash / shell tools hidden.
    assert!(hidden.contains(&"Bash"), "Bash should be hidden");
    assert!(hidden.contains(&"Shell"), "Shell should be hidden");
    assert!(
        hidden.contains(&"RunCommand"),
        "RunCommand should be hidden"
    );
    assert!(
        hidden.contains(&"ExecuteCommand"),
        "ExecuteCommand should be hidden"
    );
    assert!(hidden.contains(&"Terminal"), "Terminal should be hidden");

    // Delete / remove-style tools hidden.
    assert!(hidden.contains(&"Delete"), "Delete should be hidden");

    // Unknown broad tools hidden.
    assert!(hidden.contains(&"MultiEdit"), "MultiEdit should be hidden");

    // Publishing / git-write tools hidden.
    assert!(hidden.contains(&"CreatePR"), "CreatePR should be hidden");
    assert!(hidden.contains(&"Push"), "Push should be hidden");

    // Deploy / runtime mutation tools hidden.
    assert!(hidden.contains(&"Deploy"), "Deploy should be hidden");
    assert!(
        hidden.contains(&"RestartService"),
        "RestartService should be hidden"
    );

    // External MCP tools hidden (Unknown capability).
    assert!(hidden.contains(&"mcp__plugin_ecc_github__create_or_update_file"));
    assert!(hidden.contains(&"mcp__plugin_ecc_github__push_files"));
    assert!(hidden.contains(&"mcp__plugin_ecc_playwright__browser_click"));
}

#[test]
fn implementation_policy_hides_bash_shell_delete_publishing_deploy_unknown_anthropic() {
    let policy = implementation_policy();
    let mut req = json!({
        "messages": [],
        "tools": [
            {"name": "Read", "input_schema": {}},
            {"name": "Grep", "input_schema": {}},
            {"name": "Edit", "input_schema": {}},
            {"name": "Write", "input_schema": {}},
            {"name": "Bash", "input_schema": {}},
            {"name": "Shell", "input_schema": {}},
            {"name": "Delete", "input_schema": {}},
            {"name": "MultiEdit", "input_schema": {}},
            {"name": "CreatePR", "input_schema": {}},
            {"name": "Push", "input_schema": {}},
            {"name": "Deploy", "input_schema": {}},
            {"name": "RestartService", "input_schema": {}},
            {"name": "RunCommand", "input_schema": {}},
            {"name": "ExecuteCommand", "input_schema": {}},
            {"name": "Terminal", "input_schema": {}}
        ]
    });

    let outcome =
        shape_anthropic_request_with_policy(&mut req, "Implement the change", Some(&policy));
    let allowed = req["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect::<Vec<_>>();
    let hidden = outcome
        .hidden_tools
        .iter()
        .map(|t| t.name.as_str())
        .collect::<Vec<_>>();

    // Concrete repo tools preserved.
    assert!(allowed.contains(&"Read"));
    assert!(allowed.contains(&"Grep"));
    assert!(allowed.contains(&"Edit"));
    assert!(allowed.contains(&"Write"));

    // Bash / shell tools hidden.
    assert!(hidden.contains(&"Bash"), "Bash should be hidden");
    assert!(hidden.contains(&"Shell"), "Shell should be hidden");
    assert!(
        hidden.contains(&"RunCommand"),
        "RunCommand should be hidden"
    );
    assert!(
        hidden.contains(&"ExecuteCommand"),
        "ExecuteCommand should be hidden"
    );
    assert!(hidden.contains(&"Terminal"), "Terminal should be hidden");

    // Delete / remove-style tools hidden.
    assert!(hidden.contains(&"Delete"), "Delete should be hidden");

    // Unknown broad tools hidden.
    assert!(hidden.contains(&"MultiEdit"), "MultiEdit should be hidden");

    // Publishing / git-write tools hidden.
    assert!(hidden.contains(&"CreatePR"), "CreatePR should be hidden");
    assert!(hidden.contains(&"Push"), "Push should be hidden");

    // Deploy / runtime mutation tools hidden.
    assert!(hidden.contains(&"Deploy"), "Deploy should be hidden");
    assert!(
        hidden.contains(&"RestartService"),
        "RestartService should be hidden"
    );
}

#[test]
fn implementation_tool_surface_reports_missing_repo_navigation() {
    let policy = implementation_policy();
    assert!(policy_requires_implementation_tool_surface(&policy));
    let mut req = json!({
        "messages": [],
        "tools": [
            {"name": "Read", "input_schema": {}},
            {"name": "Edit", "input_schema": {}},
            {"name": "Write", "input_schema": {}}
        ]
    });

    let outcome =
        shape_anthropic_request_with_policy(&mut req, "Implement the change", Some(&policy));

    assert_eq!(
        missing_implementation_tool_capabilities(&outcome),
        vec!["text_search", "file_list"]
    );
}

#[test]
fn implementation_tool_surface_accepts_minimum_concrete_profile() {
    let policy = implementation_policy();
    let mut req = json!({
        "messages": [],
        "tools": [
            {"name": "Read", "input_schema": {}},
            {"name": "Grep", "input_schema": {}},
            {"name": "Glob", "input_schema": {}},
            {"name": "Edit", "input_schema": {}}
        ]
    });

    let outcome =
        shape_anthropic_request_with_policy(&mut req, "Implement the change", Some(&policy));

    assert!(missing_implementation_tool_capabilities(&outcome).is_empty());
}

#[test]
fn read_only_debug_policy_does_not_require_implementation_tool_surface() {
    let text = "Please explain the Docker compose error";
    let classification = crate::request_classification::classify_request_text(
        "agentic-os",
        "session-t",
        text,
        Some("ERROR failed to connect to http://localhost:8088"),
        "user_message",
    );
    let policy =
        crate::orchestration_policy::derive_orchestration_policy(&classification, text, false);

    assert!(!policy_requires_implementation_tool_surface(&policy));
}
