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
