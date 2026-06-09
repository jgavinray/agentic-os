use super::*;
use serde_json::json;

#[test]
fn openai_file_read_hides_shell_when_read_exists() {
    let mut req = json!({
        "messages": [{"role": "user", "content": "Read README.md"}],
        "tools": [
            {"type": "function", "function": {"name": "Read"}},
            {"type": "function", "function": {"name": "Bash"}}
        ],
        "tool_choice": {"type": "function", "function": {"name": "Bash"}}
    });

    let outcome = shape_openai_request(&mut req, "Read README.md");

    assert_eq!(outcome.decision, "shape");
    assert_eq!(outcome.reason, "prefer_canonical_tool");
    assert_eq!(outcome.hidden_tools[0].name, "Bash");
    assert_eq!(req["tools"].as_array().unwrap().len(), 1);
    assert_eq!(req["tools"][0]["function"]["name"], "Read");
    assert_eq!(req["tool_choice"], "auto");
}

#[test]
fn anthropic_file_read_hides_shell_when_read_exists() {
    let mut req = json!({
        "messages": [],
        "tools": [
            {"name": "read", "input_schema": {}},
            {"name": "bash", "input_schema": {}}
        ],
        "tool_choice": {"type": "tool", "name": "bash"}
    });

    let outcome = shape_anthropic_request(&mut req, "show the file docs/TRAJECTORIES.md");

    assert_eq!(outcome.decision, "shape");
    assert_eq!(outcome.hidden_tools[0].name, "bash");
    assert_eq!(req["tools"].as_array().unwrap().len(), 1);
    assert_eq!(req["tools"][0]["name"], "read");
    assert_eq!(req["tool_choice"], json!({"type": "auto"}));
}

#[test]
fn shell_stays_available_when_no_canonical_tool_exists() {
    let mut req = json!({
        "messages": [],
        "tools": [
            {"type": "function", "function": {"name": "Bash"}}
        ]
    });

    let outcome = shape_openai_request(&mut req, "Read README.md");

    assert_eq!(outcome.decision, "pass");
    assert_eq!(outcome.reason, "not_applicable");
    assert_eq!(req["tools"].as_array().unwrap().len(), 1);
}

#[test]
fn authorization_denies_shell_file_read_with_read_available() {
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

    let response = authorize_tool_call(&req, true);

    assert_eq!(response.decision, "deny");
    assert_eq!(response.reason, "prefer_canonical_tool");
    assert_eq!(response.preferred_tool.as_deref(), Some("Read"));
    assert_eq!(
        response.replacement.unwrap(),
        json!({"tool_name": "Read", "arguments": {"path": "README.md"}})
    );
}

#[test]
fn authorization_allows_shell_file_read_without_read_available() {
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
        available_tools: vec!["Bash".to_string()],
    };

    let response = authorize_tool_call(&req, true);

    assert_eq!(response.decision, "allow");
    assert_eq!(response.reason, "not_applicable");
}
