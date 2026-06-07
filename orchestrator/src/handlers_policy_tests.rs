use super::*;
use serde_json::json;

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
fn live_classification_overrides_generic_task_for_context_category() {
    let classification = crate::request_classification::classify_request_text(
        "agentic-os",
        "session-a",
        "Read README.md",
        None,
        "user_message",
    );

    assert_eq!(
        context_task_category("default task", Some(&classification)),
        crate::state::TaskCategory::Narrow
    );
}

#[test]
fn orchestration_policy_narrows_context_memory_levels() {
    let classification = crate::request_classification::classify_request_text(
        "agentic-os",
        "session-a",
        "Read README.md",
        None,
        "user_message",
    );
    let orchestration_policy =
        orchestration_policy::derive_orchestration_policy(&classification, "Read README.md", false);
    let mut context_policy =
        crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Moderate);

    apply_orchestration_context_limits(&mut context_policy, &orchestration_policy);

    assert_eq!(context_policy.l2_limit, 0);
    assert_eq!(context_policy.l3_limit, 0);
    assert_eq!(context_policy.failure_limit, 0);
    assert!(!orchestration_policy
        .context_sources
        .contains(&orchestration_policy::ContextSource::CompiledSummaries));
    assert!(!orchestration_policy
        .context_sources
        .contains(&orchestration_policy::ContextSource::TotalRecall));
}

// ── Tool authorization classification text ────────────────────────

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

// ── Auth headers ──────────────────────────────────────────────────

#[test]
fn provided_api_token_accepts_bearer_authorization() {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::AUTHORIZATION,
        "Bearer sk-test".parse().expect("valid header"),
    );

    assert_eq!(provided_api_token(&headers), "sk-test");
}

#[test]
fn provided_api_token_accepts_anthropic_x_api_key() {
    let mut headers = HeaderMap::new();
    headers.insert("x-api-key", "sk-test".parse().expect("valid header"));

    assert_eq!(provided_api_token(&headers), "sk-test");
}
