use super::*;
use serde_json::json;

#[test]
fn shape_openai_request_with_policy_matches_shape_openai_request() {
    let mut req1 = json!({
        "messages": [{"role": "user", "content": "Read README.md"}],
        "tools": [
            {"type": "function", "function": {"name": "Read"}},
            {"type": "function", "function": {"name": "Bash"}}
        ],
        "tool_choice": {"type": "function", "function": {"name": "Bash"}}
    });
    let mut req2 = serde_json::to_value(&req1).unwrap();

    let outcome1 = shape_openai_request(&mut req1, "Read README.md");
    let outcome2 = shape_openai_request_with_policy(&mut req2, "Read README.md", None);

    assert_eq!(outcome1.decision, outcome2.decision);
    assert_eq!(outcome1.reason, outcome2.reason);
    assert_eq!(outcome1.hidden_tools.len(), outcome2.hidden_tools.len());
    assert_eq!(outcome1.hidden_tools[0].name, outcome2.hidden_tools[0].name);
}

#[test]
fn authorize_tool_call_with_policy_matches_authorize_tool_call() {
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

    let response1 = authorize_tool_call(&req, true);
    let response2 = authorize_tool_call_with_policy(&req, true, None);

    assert_eq!(response1.decision, response2.decision);
    assert_eq!(response1.reason, response2.reason);
    assert_eq!(response1.preferred_tool, response2.preferred_tool);
    assert_eq!(response1.replacement, response2.replacement);
}

#[test]
fn implementation_policy_hides_broad_and_unknown_tools() {
    let text = "Implement the request classifier change in src/request_classification.rs";
    let classification = crate::request_classification::classify_request_text(
        "agentic-os",
        "session-a",
        text,
        None,
        "user_message",
    );
    let policy =
        crate::orchestration_policy::derive_orchestration_policy(&classification, text, false);
    let mut req = json!({
        "messages": [],
        "tools": [
            {"type": "function", "function": {"name": "Read"}},
            {"type": "function", "function": {"name": "Grep"}},
            {"type": "function", "function": {"name": "Edit"}},
            {"type": "function", "function": {"name": "Write"}},
            {"type": "function", "function": {"name": "Bash"}},
            {"type": "function", "function": {"name": "MultiEdit"}},
            {"type": "function", "function": {"name": "Delete"}},
            {"type": "function", "function": {"name": "CreatePR"}}
        ]
    });

    let outcome = shape_openai_request_with_policy(&mut req, text, Some(&policy));
    let allowed = req["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["function"]["name"].as_str().unwrap())
        .collect::<Vec<_>>();
    let hidden = outcome
        .hidden_tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(classification.intent.as_str(), "implement");
    assert_eq!(allowed, vec!["Read", "Grep", "Edit", "Write"]);
    assert!(hidden.contains(&"Bash"));
    assert!(hidden.contains(&"MultiEdit"));
    assert!(hidden.contains(&"Delete"));
    assert!(hidden.contains(&"CreatePR"));
}

#[test]
fn implementation_policy_keeps_tight_claude_code_file_tools() {
    let text = "Continue the prompt-intervention implementation, run relevant checks, and leave a concise summary.";
    let classification = crate::request_classification::classify_request_text(
        "agentic-os",
        "session-a",
        text,
        None,
        "user_message",
    );
    let policy =
        crate::orchestration_policy::derive_orchestration_policy(&classification, text, false);
    let mut req = json!({
        "messages": [],
        "tools": [
            {"name": "Read", "input_schema": {}},
            {"name": "Glob", "input_schema": {}},
            {"name": "Grep", "input_schema": {}},
            {"name": "Edit", "input_schema": {}},
            {"name": "Write", "input_schema": {}},
            {"name": "run_tests", "input_schema": {}},
            {"name": "Bash", "input_schema": {}},
            {"name": "MultiEdit", "input_schema": {}},
            {"name": "Delete", "input_schema": {}},
            {"name": "mcp__plugin_ecc_github__create_or_update_file", "input_schema": {}},
            {"name": "mcp__plugin_ecc_github__push_files", "input_schema": {}},
            {"name": "mcp__plugin_ecc_playwright__browser_click", "input_schema": {}}
        ]
    });

    let outcome = shape_anthropic_request_with_policy(&mut req, text, Some(&policy));
    let allowed = req["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect::<Vec<_>>();
    let hidden = outcome
        .hidden_tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(classification.intent.as_str(), "implement");
    assert_eq!(
        allowed,
        vec!["Read", "Glob", "Grep", "Edit", "Write", "run_tests"]
    );
    assert!(hidden.contains(&"Bash"));
    assert!(hidden.contains(&"MultiEdit"));
    assert!(hidden.contains(&"Delete"));
    assert!(hidden.contains(&"mcp__plugin_ecc_github__create_or_update_file"));
    assert!(hidden.contains(&"mcp__plugin_ecc_github__push_files"));
    assert!(hidden.contains(&"mcp__plugin_ecc_playwright__browser_click"));
}
