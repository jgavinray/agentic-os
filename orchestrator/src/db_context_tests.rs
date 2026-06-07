use super::*;

#[test]
fn context_policy_small_task_includes_l0_and_l1() {
    let mut evidence = ContextEvidence::default();
    evidence.l0_recent = vec![event_at_level("edit", "fresh implementation detail", 0)];
    evidence.l1_matching = vec![event_at_level("summary", "matching prior task summary", 1)];
    evidence.l2_repo = vec![event_at_level("summary", "repo convention", 2)];

    let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
    let (out, stats) =
        build_layered_context("r", "fix bug", &evidence, &[], &[], &policy, 8000, 4000);

    assert!(out.contains("== Recent Evidence =="));
    assert!(out.contains("fresh implementation detail"));
    assert!(out.contains("== Relevant Session Summaries =="));
    assert!(out.contains("matching prior task summary"));
    assert_eq!(stats.l0_items_injected, 1);
    assert_eq!(stats.l1_items_injected, 1);
}

#[test]
fn context_policy_broad_task_includes_l1_l2_and_l0_tail() {
    let mut evidence = ContextEvidence::default();
    evidence.l0_recent = vec![event_at_level("edit", "latest local change", 0)];
    evidence.l1_matching = vec![event_at_level("summary", "session summary", 1)];
    evidence.l2_repo = vec![event_at_level("summary", "repo-level pattern", 2)];

    let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Broad);
    let (out, _stats) = build_layered_context(
        "r",
        "large refactor",
        &evidence,
        &[],
        &[],
        &policy,
        8000,
        4000,
    );

    assert!(out.contains("latest local change"));
    assert!(out.contains("session summary"));
    assert!(out.contains("repo-level pattern"));
}

#[test]
fn context_policy_architecture_task_prioritizes_l2_l3() {
    let mut evidence = ContextEvidence::default();
    evidence.l2_repo = vec![event_at_level("summary", "repository decision record", 2)];
    evidence.l3_project = vec![event_at_level("summary", "durable architecture truth", 3)];

    let policy =
        crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Architecture);
    let (out, stats) = build_layered_context(
        "r",
        "architecture review",
        &evidence,
        &[],
        &[],
        &policy,
        8000,
        4000,
    );

    let l3_pos = out.find("durable architecture truth").unwrap();
    let l2_pos = out.find("repository decision record").unwrap();
    assert!(l3_pos < l2_pos);
    assert_eq!(stats.l2_items_injected, 1);
    assert_eq!(stats.l3_items_injected, 1);
}

#[test]
fn higher_levels_do_not_eliminate_l0_tail() {
    let mut evidence = ContextEvidence::default();
    evidence.l0_recent = vec![event_at_level("edit", "raw event still included", 0)];
    evidence.l3_project = vec![event_at_level("summary", "durable compressed memory", 3)];

    let policy =
        crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Architecture);
    let (out, _stats) = build_layered_context(
        "r",
        "architecture",
        &evidence,
        &[],
        &[],
        &policy,
        8000,
        4000,
    );

    assert!(out.contains("raw event still included"));
    assert!(out.contains("durable compressed memory"));
}

#[test]
fn failed_attempt_preserves_evidence_and_outcome() {
    let mut evidence = ContextEvidence::default();
    evidence.failures = vec![failed_event(
        "retrying the old cache key caused stale context",
        "observed repeated stale pack after append_event",
        "new key includes event count",
    )];

    let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
    let (out, stats) =
        build_layered_context("r", "fix cache", &evidence, &[], &[], &policy, 8000, 4000);

    assert!(out.contains("== Failed Attempts and Remediations =="));
    assert!(out.contains("observed repeated stale pack after append_event"));
    assert!(out.contains("Outcome: new key includes event count"));
    assert!(out.contains("Source IDs: 2"));
    assert_eq!(stats.failed_attempts_injected, 1);
}

#[test]
fn failure_history_section_sits_between_failures_and_open_questions() {
    let mut evidence = ContextEvidence::default();
    evidence.failures = vec![failed_event(
        "old patch broke cache invalidation",
        "cargo test failed",
        "invalidate by repo prefix",
    )];
    evidence.failure_history = vec![FailureHistoryItem {
        signature: "rust:type-mismatch".to_string(),
        category: "type_error".to_string(),
        failure: AgentEvent {
            event_type: crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT.to_string(),
            summary: "failed compile `rust` target `cargo` errors=1 warnings=0".to_string(),
            metadata: serde_json::json!({
                "payload": {
                    "signature": "rust:type-mismatch",
                    "signature_category": "type_error",
                    "fingerprint_version": 1,
                    "failure_summaries": ["error[E0308]: mismatched types"]
                }
            }),
            ..event_at_level(
                crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT,
                "failed compile `rust` target `cargo` errors=1 warnings=0",
                0,
            )
        },
        remediation: Some(AgentEvent {
            event_type: crate::execution_feedback::EVENT_TYPE_REMEDIATION.to_string(),
            summary: "changed generic bound to match caller".to_string(),
            metadata: serde_json::json!({
                "payload": {"signature": "rust:type-mismatch"}
            }),
            ..event_at_level(
                crate::execution_feedback::EVENT_TYPE_REMEDIATION,
                "changed generic bound to match caller",
                0,
            )
        }),
    }];
    evidence.l0_recent = vec![AgentEvent {
        event_type: "checkpoint".to_string(),
        metadata: serde_json::json!({"open_questions": ["Should cache TTL change?"]}),
        ..event_at_level("checkpoint", "checkpoint", 0)
    }];

    let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
    let (out, stats) = build_layered_context(
        "r",
        "error[E0308]",
        &evidence,
        &[],
        &[],
        &policy,
        8000,
        4000,
    );

    let failed = out.find("== Failed Attempts and Remediations ==").unwrap();
    let history = out.find("== Failure History ==").unwrap();
    let open = out.find("== Open Questions ==").unwrap();
    assert!(failed < history);
    assert!(history < open);
    assert!(out.contains("changed generic bound to match caller"));
    assert_eq!(stats.failure_history_items_injected, 1);
    assert_eq!(
        stats.failure_history_remediation_signatures,
        vec!["rust:type-mismatch".to_string()]
    );
}

#[test]
fn operational_constraints_section_sits_above_failure_history() {
    let mut evidence = ContextEvidence::default();
    evidence.operational_constraints = vec![crate::feature_extraction::OperationalConstraint {
        constraint_type: "use_known_endpoint".to_string(),
        text: "Do not use `localhost` for host-side orchestrator testing. The correct endpoint for this environment is `http://host.docker.internal:8088`.".to_string(),
    }];
    evidence.failure_history = vec![FailureHistoryItem {
        signature: "process:non-zero-exit".to_string(),
        category: "unknown".to_string(),
        failure: event_at_level(
            crate::execution_feedback::EVENT_TYPE_TOOL_RESULT,
            "failed tool `Bash` exit_code=1",
            0,
        ),
        remediation: None,
    }];

    let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
    let (out, stats) =
        build_layered_context("r", "debug", &evidence, &[], &[], &policy, 8000, 4000);

    let constraints = out.find("Operational Constraints:").unwrap();
    let history = out.find("== Failure History ==").unwrap();
    assert!(constraints < history);
    assert!(out.contains("- Do not use `localhost`"));
    assert_eq!(stats.operational_constraints_injected, 1);
}

#[test]
fn failure_history_section_omitted_when_empty() {
    let evidence = ContextEvidence::default();
    let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
    let (out, stats) =
        build_layered_context("r", "no failures", &evidence, &[], &[], &policy, 8000, 4000);

    assert!(!out.contains("== Failure History =="));
    assert_eq!(stats.failure_history_items_injected, 0);
}

#[test]
fn failure_history_section_respects_budget() {
    let mut evidence = ContextEvidence::default();
    evidence.failure_history = vec![FailureHistoryItem {
        signature: "unknown".to_string(),
        category: "unknown".to_string(),
        failure: AgentEvent {
            event_type: crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT.to_string(),
            summary: "x".repeat(200),
            metadata: serde_json::json!({
                "payload": {
                    "signature": "unknown",
                    "signature_category": "unknown",
                    "fingerprint_version": 1,
                    "failure_reason": "y".repeat(200)
                }
            }),
            ..event_at_level(
                crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT,
                "long failure",
                0,
            )
        },
        remediation: None,
    }];
    let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
    let (out, _stats) =
        build_layered_context("r", "unknown", &evidence, &[], &[], &policy, 8000, 50);
    assert!(out.contains("truncated: failure history budget exceeded"));
}

#[test]
fn event_chain_orders_from_root_to_leaf() {
    let correlation_id = uuid::Uuid::new_v4();
    let root_id = uuid::Uuid::new_v4();
    let response_id = uuid::Uuid::new_v4();
    let tool_id = uuid::Uuid::new_v4();
    let validation_id = uuid::Uuid::new_v4();

    let mut root = event_at_level("user_message", "request", 0);
    root.id = root_id.to_string();
    root.correlation_id = Some(correlation_id);
    root.parent_event_id = None;

    let mut response = event_at_level("assistant_message", "patch", 0);
    response.id = response_id.to_string();
    response.correlation_id = Some(correlation_id);
    response.parent_event_id = Some(root_id);

    let mut tool = event_at_level(
        crate::execution_feedback::EVENT_TYPE_TOOL_RESULT,
        "apply",
        0,
    );
    tool.id = tool_id.to_string();
    tool.correlation_id = Some(correlation_id);
    tool.parent_event_id = Some(response_id);

    let mut validation = event_at_level(
        crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT,
        "schema ok",
        0,
    );
    validation.id = validation_id.to_string();
    validation.correlation_id = Some(correlation_id);
    validation.parent_event_id = Some(tool_id);

    let ordered = order_event_chain(
        vec![
            validation.clone(),
            tool.clone(),
            root.clone(),
            response.clone(),
        ],
        &validation.id,
    );
    let ids: Vec<String> = ordered.into_iter().map(|event| event.id).collect();
    assert_eq!(
        ids,
        vec![
            root_id.to_string(),
            response_id.to_string(),
            tool_id.to_string(),
            validation_id.to_string()
        ]
    );
}

#[test]
fn failure_history_context_build_p99_under_five_ms() {
    let mut durations = Vec::new();
    let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
    let mut evidence = ContextEvidence::default();
    evidence.failure_history = (0..50)
        .map(|idx| FailureHistoryItem {
            signature: format!("typescript:TS{idx:04}"),
            category: "typescript".to_string(),
            failure: event_at_level(
                crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT,
                &format!("failure {idx}"),
                0,
            ),
            remediation: Some(event_at_level(
                crate::execution_feedback::EVENT_TYPE_REMEDIATION,
                &format!("remediation {idx}"),
                0,
            )),
        })
        .collect();

    for _ in 0..100 {
        let started = std::time::Instant::now();
        let _ = build_layered_context(
            "r",
            "typescript:TS2322",
            &evidence,
            &[],
            &[],
            &policy,
            8000,
            4000,
        );
        durations.push(started.elapsed());
    }
    durations.sort();
    let p99 = durations[durations.len() - 1];
    let threshold = if cfg!(debug_assertions) {
        std::time::Duration::from_millis(20)
    } else {
        std::time::Duration::from_millis(5)
    };
    assert!(p99 < threshold, "p99 was {p99:?}");
}
