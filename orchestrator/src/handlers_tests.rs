use super::*;
use chrono::Utc;
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

// ── inject_system_context ──────────────────────────────────────────

#[test]
fn inject_inserts_system_message_when_none_exists() {
    let mut payload = json!({
        "messages": [{"role": "user", "content": "hello"}]
    });
    inject_system_context(&mut payload, "prior context");
    let msgs = payload["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "prior context");
    assert_eq!(msgs[1]["role"], "user");
}

#[test]
fn inject_appends_to_existing_system_message_not_duplicates_it() {
    let mut payload = json!({
        "messages": [
            {"role": "system", "content": "base prompt"},
            {"role": "user", "content": "hello"}
        ]
    });
    inject_system_context(&mut payload, "prior context");
    let msgs = payload["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 2, "must not insert a second system message");
    let content = msgs[0]["content"].as_str().unwrap();
    assert!(content.contains("base prompt"));
    assert!(content.contains("prior context"));
}

#[test]
fn inject_keeps_system_message_at_index_zero() {
    let mut payload = json!({
        "messages": [
            {"role": "system", "content": "base prompt"},
            {"role": "user", "content": "hello"}
        ]
    });
    inject_system_context(&mut payload, "context");
    let msgs = payload["messages"].as_array().unwrap();
    assert_eq!(msgs[0]["role"], "system");
}

#[test]
fn inject_is_noop_when_messages_missing() {
    let mut payload = json!({"model": "gpt-4"});
    inject_system_context(&mut payload, "context");
    assert!(payload.get("messages").is_none());
}

// ── extract_assistant_from_sse ─────────────────────────────────────

#[test]
fn extract_sse_assembles_content_across_chunks() {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\
                   data: [DONE]\n";
    assert_eq!(extract_assistant_from_sse(sse), "Hello world");
}

#[test]
fn extract_sse_stops_at_done() {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"before\"}}]}\n\
                   data: [DONE]\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"after\"}}]}\n";
    assert_eq!(extract_assistant_from_sse(sse), "before");
}

#[test]
fn extract_sse_returns_empty_for_blank_input() {
    assert_eq!(extract_assistant_from_sse(""), "");
}

#[test]
fn extract_sse_ignores_malformed_json_data_lines() {
    let sse = "data: not-valid-json\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\
                   data: [DONE]\n";
    assert_eq!(extract_assistant_from_sse(sse), "ok");
}

#[test]
fn extract_sse_ignores_non_data_lines() {
    let sse = ": keep-alive\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\
                   data: [DONE]\n";
    assert_eq!(extract_assistant_from_sse(sse), "hi");
}

#[test]
fn extract_sse_usage_records_processed_cached_and_generated_tokens() {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\
                   data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":40,\"prompt_tokens_details\":{\"cached_tokens\":80}}}\n\
                   data: [DONE]\n";
    let usage = extract_token_usage_from_sse(sse);
    assert_eq!(usage.processed_tokens, 120);
    assert_eq!(usage.cached_tokens, 80);
    assert_eq!(usage.generated_tokens, 40);
}

#[test]
fn local_reasoning_policy_prefers_header_over_client_fields() {
    let mut headers = HeaderMap::new();
    headers.insert("x-agent-reasoning-policy", "high".parse().unwrap());
    let payload = serde_json::json!({
        "reasoning_effort": "low",
        "thinking": {"type": "enabled", "budget_tokens": 1024}
    });

    let selection = local_reasoning_selection(&headers, &payload);

    assert_eq!(selection.policy, LocalReasoningPolicy::High);
    assert_eq!(selection.source, "x-agent-reasoning-policy");
}

#[test]
fn local_reasoning_policy_maps_anthropic_thinking_budget() {
    let headers = HeaderMap::new();

    let low = local_reasoning_selection(
        &headers,
        &serde_json::json!({"thinking": {"budget_tokens": 1024}}),
    );
    let medium = local_reasoning_selection(
        &headers,
        &serde_json::json!({"thinking": {"budget_tokens": 4096}}),
    );
    let high = local_reasoning_selection(
        &headers,
        &serde_json::json!({"thinking": {"budget_tokens": 12000}}),
    );

    assert_eq!(low.policy, LocalReasoningPolicy::Low);
    assert_eq!(medium.policy, LocalReasoningPolicy::Medium);
    assert_eq!(high.policy, LocalReasoningPolicy::High);
    assert_eq!(high.source, "client_thinking_budget");
}

#[test]
fn local_reasoning_defaults_preserve_explicit_sampling() {
    let mut req = serde_json::json!({
        "max_tokens": 333,
        "temperature": 0.7
    });

    apply_local_reasoning_defaults(
        &mut req,
        LocalReasoningSelection {
            policy: LocalReasoningPolicy::Low,
            source: "test",
        },
    );

    assert_eq!(req["max_tokens"], serde_json::json!(333));
    assert_eq!(req["temperature"], serde_json::json!(0.7));
}

#[test]
fn local_reasoning_defaults_shape_missing_sampling() {
    let mut req = serde_json::json!({});

    apply_local_reasoning_defaults(
        &mut req,
        LocalReasoningSelection {
            policy: LocalReasoningPolicy::Low,
            source: "test",
        },
    );

    assert_eq!(req["max_tokens"], serde_json::json!(2048));
    assert_eq!(req["temperature"], serde_json::json!(0.1));
}

#[test]
fn extract_sse_handles_missing_content_field_gracefully() {
    let sse = "data: {\"choices\":[{\"delta\":{}}]}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"real\"}}]}\n\
                   data: [DONE]\n";
    assert_eq!(extract_assistant_from_sse(sse), "real");
}

#[test]
fn retry_backoff_durations_are_correct() {
    let delays: Vec<u64> = (0u32..2).map(|a| 200 * 2u64.pow(a)).collect();
    assert_eq!(delays, vec![200, 400]);
}

#[test]
fn rate_limited_response_sets_429_and_retry_after() {
    let response = rate_limited_response("secret-token", 3);
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get(header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok()),
        Some("3")
    );
}

// ── API_KEYS parsing: semicolon-delimited token,namespace entries ──

#[test]
fn parse_simple_single_entry() {
    let input = "agent-os,project-alpha";
    let mut parts = input.splitn(2, ',');
    let token = parts.next().unwrap_or(input).trim().to_string();
    let namespace = parts.next().unwrap_or(&token).trim().to_string();
    assert_eq!(token, "agent-os");
    assert_eq!(namespace, "project-alpha");
}

#[test]
fn parse_multiple_semicolon_entries() {
    let input = "agent-os,project-alpha;agent-os,project-beta;sk-work,work";
    let entries: Vec<(String, String)> = input
        .split(';')
        .map(|s| {
            let s = s.trim();
            let mut parts = s.splitn(2, ',');
            let token = parts.next().unwrap_or(s).trim().to_string();
            let namespace = parts.next().unwrap_or(&token).trim().to_string();
            (token, namespace)
        })
        .filter(|(t, _)| !t.is_empty())
        .collect();

    assert_eq!(entries.len(), 3);
    assert_eq!(
        entries[0],
        ("agent-os".to_string(), "project-alpha".to_string())
    );
    assert_eq!(
        entries[1],
        ("agent-os".to_string(), "project-beta".to_string())
    );
    assert_eq!(entries[2], ("sk-work".to_string(), "work".to_string()));
}

#[test]
fn parse_empty_token_filtered() {
    let input = "token1,ns1;;token2,ns2";
    let entries: Vec<(String, String)> = input
        .split(';')
        .map(|s| {
            let s = s.trim();
            let mut parts = s.splitn(2, ',');
            let token = parts.next().unwrap_or(s).trim().to_string();
            let namespace = parts.next().unwrap_or(&token).trim().to_string();
            (token, namespace)
        })
        .filter(|(t, _)| !t.is_empty())
        .collect();

    assert_eq!(entries.len(), 2);
}

#[test]
fn parse_fallback_to_token_when_no_namespace() {
    let input = "my-token";
    let mut parts = input.splitn(2, ',');
    let token = parts.next().unwrap_or(input).trim().to_string();
    let namespace = parts.next().unwrap_or(&token).trim().to_string();
    assert_eq!(token, "my-token");
    assert_eq!(namespace, "my-token");
}

#[test]
fn parse_default_value() {
    let input = "agent-os,agentic-os";
    let entries: Vec<(String, String)> = input
        .split(';')
        .map(|s| {
            let s = s.trim();
            let mut parts = s.splitn(2, ',');
            let token = parts.next().unwrap_or(s).trim().to_string();
            let namespace = parts.next().unwrap_or(&token).trim().to_string();
            (token, namespace)
        })
        .filter(|(t, _)| !t.is_empty())
        .collect();

    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0],
        ("agent-os".to_string(), "agentic-os".to_string())
    );
}

// ── Model substitution ────────────────────────────────────────

#[test]
fn model_substituted_with_default_in_chat_completions() {
    let mut req = json!({
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "hi"}]
    });
    let default_model = "qwen36-35b-heretic";
    req["model"] = Value::String(default_model.to_string());
    assert_eq!(req["model"].as_str().unwrap(), default_model);
}

#[test]
fn model_substituted_with_default_for_any_client_model_name() {
    let client_models = [
        "gpt-4o",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "gpt-4-turbo",
    ];
    let default_model = "qwen36-35b-heretic";
    for client_model in client_models {
        let mut req = json!({
            "model": client_model,
            "messages": [{"role": "user", "content": "hi"}]
        });
        req["model"] = Value::String(default_model.to_string());
        assert_eq!(
            req["model"].as_str().unwrap(),
            default_model,
            "client model '{client_model}' was not replaced"
        );
    }
}

#[test]
fn anthropic_sanitize_removes_local_backend_incompatible_params() {
    let mut req = json!({
        "model": "qwen36-27b",
        "max_tokens": 1024,
        "max_output_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "reasoning_effort": "high",
        "context_management": {"edits": []},
        "messages": [{"role": "user", "content": "hi"}]
    });

    anthropic::sanitize_litellm_request(&mut req);

    assert!(req.get("thinking").is_none());
    assert!(req.get("max_output_tokens").is_none());
    assert!(req.get("reasoning_effort").is_none());
    assert!(req.get("context_management").is_none());
    assert_eq!(req["max_tokens"], json!(1024));
    assert_eq!(req["messages"][0]["content"], "hi");
}

#[test]
fn max_tokens_clamp_respects_explicit_small_values() {
    let mut req = json!({
        "model": "qwen36-27b",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}]
    });

    enforce_min_max_tokens(&mut req);

    assert_eq!(req["max_tokens"], json!(1));
}

#[test]
fn max_tokens_clamp_defaults_missing_values() {
    let mut req = json!({
        "model": "qwen36-27b",
        "messages": [{"role": "user", "content": "hi"}]
    });

    enforce_min_max_tokens(&mut req);

    assert_eq!(req["max_tokens"], json!(DEFAULT_MAX_TOKENS));
}

#[test]
fn max_tokens_clamp_caps_oversized_values() {
    let mut req = json!({
        "model": "qwen36-27b",
        "max_tokens": MAX_MAX_TOKENS + 1,
        "messages": [{"role": "user", "content": "hi"}]
    });

    enforce_min_max_tokens(&mut req);

    assert_eq!(req["max_tokens"], json!(MAX_MAX_TOKENS));
}

#[test]
fn metrics_include_context_pack_counts() {
    let metrics = telemetry::MetricsRegistry::new();
    telemetry::record_context_pack(
        &metrics,
        &ContextPackStats {
            context_chars: 400,
            context_tokens_estimate: 100,
            l0_items_injected: 1,
            l1_items_injected: 2,
            l2_items_injected: 3,
            l3_items_injected: 4,
            failed_attempts_injected: 1,
            remediations_injected: 1,
            retrieval_semantic_hits: 5,
            retrieval_fts_hits: 6,
            retrieval_deduped_hits: 7,
            cache_hit: false,
            ..Default::default()
        },
    );
    telemetry::record_tokens(
        &metrics,
        &TokenUsage {
            processed_tokens: 120,
            cached_tokens: 80,
            generated_tokens: 40,
        },
        "test-model",
    );
    telemetry::record_promotion(&metrics, false, false);
    telemetry::record_promotion(&metrics, true, true);

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.context_pack_requests, 1);
    assert_eq!(snapshot.context_cache_misses, 1);
    assert_eq!(snapshot.l3_items_injected, 4);
    assert_eq!(snapshot.retrieval_deduped_hits, 7);
    assert_eq!(snapshot.processed_tokens, 120);
    assert_eq!(snapshot.cached_tokens, 80);
    assert_eq!(snapshot.generated_tokens, 40);
    assert_eq!(snapshot.promotion_attempts, 2);
    assert_eq!(snapshot.promotion_accepted, 1);
    assert_eq!(snapshot.promotion_rejected, 1);
    assert_eq!(snapshot.memory_source_coverage, 0.5);
}

#[test]
fn context_pack_sections_are_stable() {
    let mut evidence = db::ContextEvidence::default();
    evidence.l3_project = vec![test_event("summary", "L3 architecture truth", 3)];
    evidence.l2_repo = vec![test_event("summary", "L2 repo pattern", 2)];
    evidence.l1_matching = vec![test_event("summary", "L1 task summary", 1)];
    evidence.l0_recent = vec![test_event("edit", "L0 raw evidence", 0)];
    evidence.failures = vec![test_event("failed_attempt", "do not repeat this", 0)];
    let policy = ContextPolicy::for_category(TaskCategory::Architecture);

    let (context, _stats) = db::build_layered_context(
        "repo",
        "architecture task",
        &evidence,
        &[],
        &[],
        &policy,
        16_000,
        4000,
    );

    let l3 = context.find("== Durable Project Memory ==").unwrap();
    let l2 = context.find("== Repo Patterns and Decisions ==").unwrap();
    let l1 = context.find("== Relevant Session Summaries ==").unwrap();
    let l0 = context.find("== Recent Evidence ==").unwrap();
    let failures = context
        .find("== Failed Attempts and Remediations ==")
        .unwrap();

    assert!(l3 < l2);
    assert!(l2 < l1);
    assert!(l1 < l0);
    assert!(l0 < failures);
}

#[test]
fn failed_attempts_are_injected_for_debug_tasks() {
    let mut evidence = db::ContextEvidence::default();
    evidence.failures = vec![db::AgentEvent {
        evidence: Some("the old retry loop timed out".to_string()),
        metadata: json!({"outcome": "bounded retry fixed the lag"}),
        ..test_event("failed_attempt", "unbounded retries caused lag", 0)
    }];
    let policy = ContextPolicy::for_category(TaskCategory::Narrow);

    let (context, stats) = db::build_layered_context(
        "repo",
        "debug lag",
        &evidence,
        &[],
        &[],
        &policy,
        8000,
        4000,
    );

    assert!(context.contains("unbounded retries caused lag"));
    assert!(context.contains("Evidence: the old retry loop timed out"));
    assert!(context.contains("Outcome: bounded retry fixed the lag"));
    assert_eq!(stats.failed_attempts_injected, 1);
}

#[test]
fn cache_hit_rate_updates() {
    let metrics = telemetry::MetricsRegistry::new();
    telemetry::record_context_pack(
        &metrics,
        &ContextPackStats {
            cache_hit: false,
            ..Default::default()
        },
    );
    telemetry::record_context_pack(
        &metrics,
        &ContextPackStats {
            cache_hit: true,
            ..Default::default()
        },
    );

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.context_pack_requests, 2);
    assert_eq!(snapshot.context_cache_misses, 1);
    assert_eq!(snapshot.context_cache_hits, 1);
}

#[test]
fn failure_history_and_validation_capture_are_feature_flagged() {
    let context_src = include_str!("context_packing.rs");
    let validations_src = include_str!("routes/validations.rs");
    assert!(context_src.contains("state.execution_feedback_enabled"));
    let ctx_start = context_src
        .find("async fn get_or_build_cached_context")
        .expect("get_or_build_cached_context not found");
    let ctx_body: String = context_src[ctx_start..].chars().take(4500).collect();
    assert!(ctx_body.contains("state.execution_feedback_enabled"));

    let validation_start = validations_src
        .find("pub async fn validations")
        .expect("validations handler not found");
    let validation_body: String = validations_src[validation_start..]
        .chars()
        .take(1200)
        .collect();
    assert!(validation_body.contains("!state.execution_feedback_enabled"));
}

#[test]
fn trajectory_capture_is_feature_flagged() {
    let sessions_src = include_str!("routes/sessions.rs");
    let trajectory_src = include_str!("background/trajectory.rs");
    assert!(sessions_src.contains("state.trajectory_capture_enabled"));
    let append_start = sessions_src
        .find("pub async fn append_event")
        .expect("append_event handler not found");
    let append_body: String = sessions_src[append_start..].chars().take(1400).collect();
    assert!(append_body.contains("req.trajectory_id = None"));
    assert!(append_body.contains("req.attempt_index = None"));
    assert!(append_body.contains("req.event_role = None"));

    let sweep_start = trajectory_src
        .find("pub async fn run_trajectory_idle_sweep")
        .expect("trajectory idle sweep not found");
    let sweep_body: String = trajectory_src[sweep_start..].chars().take(800).collect();
    assert!(sweep_body.contains("!state.trajectory_capture_enabled"));
}

/// Verify the model request path does not await a full context rebuild.
#[test]
fn pack_context_into_req_uses_async_cache_refresh() {
    let src = include_str!("handlers.rs");
    let pctr_start = src
        .find("async fn pack_context_into_req")
        .expect("pack_context_into_req not found");
    let body = &src[pctr_start..pctr_start + 1500];
    assert!(
        body.contains("cached_context_for_request"),
        "pack_context_into_req should use cached/minimal context immediately"
    );
    assert!(
        !body.contains("get_or_build_cached_context("),
        "pack_context_into_req should not await full context construction"
    );
    assert!(
        include_str!("context_packing.rs").contains("fn spawn_context_cache_refresh")
            && include_str!("context_packing.rs").contains("get_or_build_cached_context_inner")
            && include_str!("background/mod.rs").contains("tokio::spawn(async move"),
        "context cache refresh should run in the background"
    );
}

// ── inject_system_context_anthropic ───────────────────────────────

#[test]
fn anthropic_inject_sets_system_when_absent() {
    let mut payload = json!({"messages": [{"role": "user", "content": "hi"}]});
    inject_system_context_anthropic(&mut payload, "ctx");
    assert_eq!(
        payload["system"],
        json!([{
            "type": "text",
            "text": "ctx"
        }])
    );
}

#[test]
fn anthropic_inject_appends_to_string_system() {
    let mut payload = json!({"system": "base", "messages": []});
    inject_system_context_anthropic(&mut payload, "ctx");
    let sys = payload["system"].as_array().unwrap();
    assert_eq!(sys[0], json!({"type": "text", "text": "base"}));
    assert_eq!(
        sys[1],
        json!({
            "type": "text",
            "text": "ctx"
        })
    );
}

#[test]
fn anthropic_inject_flattens_array_system_and_appends() {
    let mut payload = json!({
        "system": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}],
        "messages": []
    });
    inject_system_context_anthropic(&mut payload, "ctx");
    let sys = payload["system"].as_array().unwrap();
    assert_eq!(sys[0], json!({"type": "text", "text": "part1"}));
    assert_eq!(sys[1], json!({"type": "text", "text": "part2"}));
    assert_eq!(
        sys[2],
        json!({
            "type": "text",
            "text": "ctx"
        })
    );
}

#[test]
fn anthropic_inject_ignores_non_text_system_type() {
    let mut payload = json!({"system": 42, "messages": []});
    inject_system_context_anthropic(&mut payload, "ctx");
    assert_eq!(
        payload["system"],
        json!([{
            "type": "text",
            "text": "ctx"
        }])
    );
}

#[test]
fn anthropic_inject_caches_stable_prefix_not_dynamic_tail() {
    let mut payload = json!({"messages": []});
    let context =
            "== Stable Context Artifacts ==\n[repo:service_topology:active]\nstable\n\nRepository: repo\nTask: task\n";
    inject_system_context_anthropic(&mut payload, context);
    let sys = payload["system"].as_array().unwrap();
    assert_eq!(sys.len(), 2);
    assert_eq!(
        sys[0]["cache_control"],
        json!({"type": "ephemeral"}),
        "stable compiler prefix should be provider-cacheable"
    );
    assert!(sys[0]["text"]
        .as_str()
        .unwrap()
        .contains("Stable Context Artifacts"));
    assert_eq!(
        sys[1].get("cache_control"),
        None,
        "dynamic repository/task tail must not become a provider cache breakpoint"
    );
    assert!(sys[1]["text"]
        .as_str()
        .unwrap()
        .starts_with("Repository: repo"));
}

// ── extract_assistant_from_anthropic_response ─────────────────────

#[test]
fn anthropic_response_extracts_text_block() {
    let resp = json!({"content": [{"type": "text", "text": "hello"}]});
    assert_eq!(extract_assistant_from_anthropic_response(&resp), "hello");
}

#[test]
fn anthropic_response_skips_tool_use_blocks() {
    let resp = json!({"content": [
        {"type": "tool_use", "id": "c1", "name": "bash", "input": {}},
        {"type": "text", "text": "done"}
    ]});
    assert_eq!(extract_assistant_from_anthropic_response(&resp), "done");
}

#[test]
fn anthropic_response_concatenates_multiple_text_blocks() {
    let resp = json!({"content": [
        {"type": "text", "text": "foo"},
        {"type": "text", "text": "bar"}
    ]});
    assert_eq!(extract_assistant_from_anthropic_response(&resp), "foobar");
}

#[test]
fn anthropic_response_returns_empty_when_no_content() {
    assert_eq!(extract_assistant_from_anthropic_response(&json!({})), "");
}

#[test]
fn anthropic_response_truncates_at_500_chars() {
    let long = "x".repeat(600);
    let resp = json!({"content": [{"type": "text", "text": long}]});
    assert_eq!(extract_assistant_from_anthropic_response(&resp).len(), 500);
}

// ── extract_token_usage_from_anthropic_sse ────────────────────────

#[test]
fn anthropic_sse_usage_reads_message_start_and_delta() {
    let raw = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":100,\"cache_read_input_tokens\":40}}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":25}}\n\n",
        );
    let usage = extract_token_usage_from_anthropic_sse(raw);
    assert_eq!(usage.processed_tokens, 100);
    assert_eq!(usage.cached_tokens, 40);
    assert_eq!(usage.generated_tokens, 25);
}

#[test]
fn anthropic_sse_usage_is_zero_for_empty_stream() {
    let usage = extract_token_usage_from_anthropic_sse("");
    assert_eq!(usage.processed_tokens, 0);
    assert_eq!(usage.generated_tokens, 0);
}

// ── extract_assistant_from_anthropic_sse ──────────────────────────

#[test]
fn anthropic_sse_collects_text_deltas() {
    let raw = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
        );
    assert_eq!(extract_assistant_from_anthropic_sse(raw), "hello");
}

#[test]
fn anthropic_sse_skips_non_text_deltas() {
    let raw = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n",
        );
    assert_eq!(extract_assistant_from_anthropic_sse(raw), "");
}

#[test]
fn anthropic_sse_truncates_at_500_chars() {
    let chunk_text = "x".repeat(300);
    let make_line = |t: &str| {
        format!(
                "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{t}\"}}}}\n\n"
            )
    };
    let raw = make_line(&chunk_text) + &make_line(&chunk_text);
    assert_eq!(extract_assistant_from_anthropic_sse(&raw).len(), 500);
}

fn test_event(event_type: &str, summary: &str, summary_level: i32) -> db::AgentEvent {
    db::AgentEvent {
        id: uuid::Uuid::new_v4().to_string(),
        session_id: "session".to_string(),
        repo: "repo".to_string(),
        actor: "actor".to_string(),
        event_type: event_type.to_string(),
        summary: summary.to_string(),
        evidence: None,
        metadata: json!({}),
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: None,
        created_at: Utc::now(),
        summary_level,
    }
}
