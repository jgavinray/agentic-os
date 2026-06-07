use super::*;
use chrono::Utc;
use serde_json::json;
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

#[test]
fn retry_backoff_durations_are_correct() {
    let delays: Vec<u64> = (0u32..2).map(|a| 200 * 2u64.pow(a)).collect();
    assert_eq!(delays, vec![200, 400]);
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
    let src = include_str!("handlers_context.rs");
    let pctr_start = src
        .find("async fn pack_context_into_req")
        .expect("pack_context_into_req not found");
    let body: String = src[pctr_start..].chars().take(1500).collect();
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
