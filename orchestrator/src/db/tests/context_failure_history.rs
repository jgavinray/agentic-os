use super::*;

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
