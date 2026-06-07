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

// -----------------------------------------------------------------------
// policy_filtered_all_tools — OpenAI
// -----------------------------------------------------------------------

#[test]
fn openai_bash_only_with_empty_policy_hides_all() {
    use crate::orchestration_policy::{EditPolicy, GitPolicy, RuntimePolicy};

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
    use crate::orchestration_policy::{EditPolicy, GitPolicy, RuntimePolicy};

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

// -----------------------------------------------------------------------
// policy_filtered_all_tools — Anthropic
// -----------------------------------------------------------------------

#[test]
fn anthropic_bash_only_with_empty_policy_hides_all() {
    use crate::orchestration_policy::{EditPolicy, GitPolicy, RuntimePolicy};

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
