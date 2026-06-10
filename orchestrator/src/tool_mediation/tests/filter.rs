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

fn claude_ops_tools() -> Vec<serde_json::Value> {
    vec![
        json!({"name": "Agent", "input_schema": {}}),
        json!({"name": "AskUserQuestion", "input_schema": {}}),
        json!({"name": "Bash", "input_schema": {}}),
        json!({"name": "CronCreate", "input_schema": {}}),
        json!({"name": "CronDelete", "input_schema": {}}),
        json!({"name": "CronList", "input_schema": {}}),
        json!({"name": "Edit", "input_schema": {}}),
        json!({"name": "EnterPlanMode", "input_schema": {}}),
        json!({"name": "EnterWorktree", "input_schema": {}}),
        json!({"name": "ExitPlanMode", "input_schema": {}}),
        json!({"name": "ExitWorktree", "input_schema": {}}),
        json!({"name": "LSP", "input_schema": {}}),
        json!({"name": "Monitor", "input_schema": {}}),
        json!({"name": "NotebookEdit", "input_schema": {}}),
        json!({"name": "PushNotification", "input_schema": {}}),
        json!({"name": "Read", "input_schema": {}}),
        json!({"name": "ScheduleWakeup", "input_schema": {}}),
        json!({"name": "Skill", "input_schema": {}}),
        json!({"name": "TaskCreate", "input_schema": {}}),
        json!({"name": "TaskGet", "input_schema": {}}),
        json!({"name": "TaskList", "input_schema": {}}),
        json!({"name": "TaskOutput", "input_schema": {}}),
        json!({"name": "TaskStop", "input_schema": {}}),
        json!({"name": "TaskUpdate", "input_schema": {}}),
        json!({"name": "WebFetch", "input_schema": {}}),
        json!({"name": "WebSearch", "input_schema": {}}),
        json!({"name": "Workflow", "input_schema": {}}),
        json!({"name": "Write", "input_schema": {}}),
    ]
}

fn policy_for_intent(
    intent: crate::request_classification::RequestIntent,
    artifact_type: crate::request_classification::RequestArtifactType,
) -> crate::orchestration_policy::OrchestrationPolicy {
    let mut classification = crate::request_classification::RequestClassification::deterministic(
        format!("event-{}", intent.as_str()),
        "agentic-os".to_string(),
        "session-tool-mediation".to_string(),
        chrono::Utc::now(),
    );
    classification.intent = intent;
    classification.artifact_type = artifact_type;
    crate::orchestration_policy::derive_orchestration_policy(
        &classification,
        "representative request",
        false,
    )
}

fn allowed_anthropic_tool_names(req: &serde_json::Value) -> Vec<&str> {
    req["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect()
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

#[test]
fn implementation_policy_preserves_tight_claude_code_ops_surface_anthropic() {
    let policy = implementation_policy();
    let mut req = json!({
        "messages": [],
        "tools": claude_ops_tools()
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

    assert_eq!(allowed, vec!["Bash", "Edit", "LSP", "Read", "Write"]);
    assert!(
        !hidden.contains(&"Bash"),
        "Bash must be exposed so implement requests can run validation commands"
    );
    assert!(hidden.contains(&"Agent"), "Agent should stay hidden");
    assert!(hidden.contains(&"Workflow"), "Workflow should stay hidden");
    assert!(hidden.contains(&"Monitor"), "Monitor should stay hidden");
    assert!(
        hidden.contains(&"WebSearch"),
        "WebSearch should stay hidden"
    );
    assert!(
        hidden.contains(&"CronCreate"),
        "CronCreate should stay hidden"
    );

    assert_eq!(
        outcome
            .allowed_tools
            .iter()
            .map(|tool| (tool.name.as_str(), tool.capability))
            .collect::<Vec<_>>(),
        vec![
            ("Bash", "shell"),
            ("Edit", "file_edit"),
            ("LSP", "text_search"),
            ("Read", "file_read"),
            ("Write", "file_edit")
        ]
    );
    assert_eq!(
        missing_implementation_tool_capabilities(&outcome),
        Vec::<&'static str>::new()
    );
}

#[test]
fn all_intents_validate_the_full_claude_code_tool_surface() {
    use crate::request_classification::{RequestArtifactType, RequestIntent};

    struct Case {
        name: &'static str,
        intent: RequestIntent,
        artifact_type: RequestArtifactType,
        expected_allowed: &'static [&'static str],
    }

    let cases = [
        Case {
            name: "explain",
            intent: RequestIntent::Explain,
            artifact_type: RequestArtifactType::PlainText,
            expected_allowed: &["LSP", "Read", "WebFetch", "WebSearch"],
        },
        Case {
            name: "debug_code",
            intent: RequestIntent::Debug,
            artifact_type: RequestArtifactType::Code,
            expected_allowed: &["Bash", "Edit", "LSP", "Read", "Write"],
        },
        Case {
            name: "debug_logs",
            intent: RequestIntent::Debug,
            artifact_type: RequestArtifactType::Logs,
            expected_allowed: &["Bash", "LSP", "Read"],
        },
        Case {
            name: "implement",
            intent: RequestIntent::Implement,
            artifact_type: RequestArtifactType::Code,
            expected_allowed: &["Bash", "Edit", "LSP", "Read", "Write"],
        },
        Case {
            name: "generate_config",
            intent: RequestIntent::GenerateConfig,
            artifact_type: RequestArtifactType::Yaml,
            expected_allowed: &["Bash", "Edit", "LSP", "Read", "Write"],
        },
        Case {
            name: "modify_config",
            intent: RequestIntent::ModifyConfig,
            artifact_type: RequestArtifactType::Json,
            expected_allowed: &["Bash", "Edit", "LSP", "Read", "Write"],
        },
        Case {
            name: "summarize",
            intent: RequestIntent::Summarize,
            artifact_type: RequestArtifactType::Markdown,
            expected_allowed: &["LSP", "Read"],
        },
        Case {
            name: "classify",
            intent: RequestIntent::Classify,
            artifact_type: RequestArtifactType::PlainText,
            expected_allowed: &["LSP", "Read"],
        },
        Case {
            name: "search",
            intent: RequestIntent::Search,
            artifact_type: RequestArtifactType::PlainText,
            expected_allowed: &["LSP", "Read", "WebFetch", "WebSearch"],
        },
        Case {
            name: "plan",
            intent: RequestIntent::Plan,
            artifact_type: RequestArtifactType::Code,
            expected_allowed: &["LSP", "Read", "WebFetch", "WebSearch"],
        },
        Case {
            name: "operate_tool",
            intent: RequestIntent::OperateTool,
            artifact_type: RequestArtifactType::PlainText,
            expected_allowed: &["Bash"],
        },
        Case {
            name: "unknown",
            intent: RequestIntent::Unknown,
            artifact_type: RequestArtifactType::Unknown,
            expected_allowed: &[],
        },
    ];

    for case in cases {
        let policy = policy_for_intent(case.intent, case.artifact_type);
        let mut req = json!({
            "messages": [],
            "tools": claude_ops_tools()
        });

        let outcome = shape_anthropic_request_with_policy(&mut req, case.name, Some(&policy));
        let allowed = allowed_anthropic_tool_names(&req);
        assert_eq!(allowed, case.expected_allowed, "case {}", case.name);

        for tool in &outcome.allowed_tools {
            let capability =
                crate::tool_mediation::classification::capability_for_tool_name(&tool.name);
            assert!(
                policy_allows_tool_capability(&policy, capability),
                "case {} allowed {} without policy capability {}",
                case.name,
                tool.name,
                tool.capability
            );
            assert!(
                !policy_blocks_tool_capability(&policy, capability),
                "case {} allowed {} despite blocked capability {}",
                case.name,
                tool.name,
                tool.capability
            );
        }

        for tool in &outcome.hidden_tools {
            let capability =
                crate::tool_mediation::classification::capability_for_tool_name(&tool.name);
            assert!(
                capability == ToolCapability::Unknown
                    || policy_blocks_tool_capability(&policy, capability)
                    || !policy_allows_tool_capability(&policy, capability),
                "case {} hid {} even though capability {} is allowed and unblocked",
                case.name,
                tool.name,
                tool.capability
            );
        }
    }
}

#[test]
fn file_edit_requests_mentioning_commit_are_not_publishing_intent() {
    // "commit" as an incidental word ("do not commit") must not flip the
    // tool intent to Publishing; the edit verbs decide.
    let intent =
        detect_tool_intent("Fix the bug in src/lib.rs and update the tests, but do not commit.");

    assert_eq!(intent, ToolIntent::FileEdit);
}

// ---------------------------------------------------------------------------
// Implementation policy: dangerous / broad tools hidden
// ---------------------------------------------------------------------------

#[test]
fn implementation_policy_exposes_shell_hides_delete_publishing_deploy_unknown_openai() {
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

    // Shell tools exposed in the menu; per-command authorization still gates
    // what they may run (validation allowed, mutation denied).
    assert!(allowed.contains(&"Bash"), "Bash should be exposed");
    assert!(allowed.contains(&"Shell"), "Shell should be exposed");
    assert!(
        allowed.contains(&"RunCommand"),
        "RunCommand should be exposed"
    );
    assert!(
        allowed.contains(&"ExecuteCommand"),
        "ExecuteCommand should be exposed"
    );
    assert!(allowed.contains(&"Terminal"), "Terminal should be exposed");

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
fn implementation_policy_exposes_shell_hides_delete_publishing_deploy_unknown_anthropic() {
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

    // Shell tools exposed in the menu; per-command authorization still gates
    // what they may run (validation allowed, mutation denied).
    assert!(allowed.contains(&"Bash"), "Bash should be exposed");
    assert!(allowed.contains(&"Shell"), "Shell should be exposed");
    assert!(
        allowed.contains(&"RunCommand"),
        "RunCommand should be exposed"
    );
    assert!(
        allowed.contains(&"ExecuteCommand"),
        "ExecuteCommand should be exposed"
    );
    assert!(allowed.contains(&"Terminal"), "Terminal should be exposed");

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
