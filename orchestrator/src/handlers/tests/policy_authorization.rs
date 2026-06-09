use super::*;

#[test]
fn derive_tool_authorization_policy_uses_request_fields() {
    let req = crate::tool_mediation::ToolAuthorizeRequest {
        session_id: Some("session-a".to_string()),
        repo: Some("repo-a".to_string()),
        task: None,
        trajectory_id: None,
        attempt_index: None,
        parent_event_id: None,
        user_intent: Some("explain the logs".to_string()),
        tool_name: "Read".to_string(),
        arguments: json!({}),
        available_tools: vec![],
    };

    let (classification, policy) = derive_tool_authorization_policy(&req, "namespace-a", false);

    assert_eq!(classification.repo, "repo-a");
    assert_eq!(classification.session_id, "session-a");
    assert!(policy
        .scope_policy
        .contains(&orchestration_policy::ScopePolicy::NoScp));
}

#[test]
fn derive_tool_authorization_policy_falls_back_to_namespace_and_tool_name() {
    let req = crate::tool_mediation::ToolAuthorizeRequest {
        session_id: None,
        repo: None,
        task: None,
        trajectory_id: None,
        attempt_index: None,
        parent_event_id: None,
        user_intent: None,
        tool_name: "Bash".to_string(),
        arguments: json!({}),
        available_tools: vec![],
    };

    let (classification, policy) = derive_tool_authorization_policy(&req, "namespace-a", false);

    assert_eq!(classification.repo, "namespace-a");
    assert_eq!(classification.session_id, "unknown");
    assert!(policy
        .scope_policy
        .contains(&orchestration_policy::ScopePolicy::NoScp));
}

#[test]
fn derive_tool_authorization_policy_sees_destructive_command_arguments() {
    let req = crate::tool_mediation::ToolAuthorizeRequest {
        session_id: Some("session-a".to_string()),
        repo: Some("repo-a".to_string()),
        task: None,
        trajectory_id: None,
        attempt_index: None,
        parent_event_id: None,
        user_intent: Some("run a command".to_string()),
        tool_name: "Bash".to_string(),
        arguments: json!({"command":"rm -rf /tmp/example"}),
        available_tools: vec![],
    };

    let (classification, policy) = derive_tool_authorization_policy(&req, "namespace-a", false);

    assert!(classification
        .risk
        .contains(&crate::request_classification::RequestRisk::DestructiveCommand));
    assert!(policy
        .blocked_tools
        .contains(&orchestration_policy::ToolCapability::ShellMutation));
}

#[test]
fn derive_tool_authorization_policy_allows_search_command_arguments_without_destructive_risk() {
    let req = crate::tool_mediation::ToolAuthorizeRequest {
        session_id: Some("session-a".to_string()),
        repo: Some("repo-a".to_string()),
        task: None,
        trajectory_id: None,
        attempt_index: None,
        parent_event_id: None,
        user_intent: None,
        tool_name: "Bash".to_string(),
        arguments: json!({"command":"rg pattern src"}),
        available_tools: vec![],
    };

    let (classification, policy) = derive_tool_authorization_policy(&req, "namespace-a", false);

    assert!(!classification
        .risk
        .contains(&crate::request_classification::RequestRisk::DestructiveCommand));
    assert!(policy
        .scope_policy
        .contains(&orchestration_policy::ScopePolicy::NoScp));
}

#[test]
fn tool_authorization_classification_text_ignores_unknown_argument_keys() {
    let req = crate::tool_mediation::ToolAuthorizeRequest {
        session_id: None,
        repo: None,
        task: None,
        trajectory_id: None,
        attempt_index: None,
        parent_event_id: None,
        user_intent: None,
        tool_name: "Bash".to_string(),
        arguments: json!({"secret":"sk-test", "command":"echo ok"}),
        available_tools: vec![],
    };

    let text = tool_authorization_classification_text(&req);

    assert!(!text.contains("sk-test"));
    assert!(text.contains("command:echo ok"));
}
