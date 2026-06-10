use super::*;

#[test]
fn worker_scope_check_marker_is_recorded() {
    let metadata = annotate_event_metadata(
        "assistant_message",
        "SCOPE-CHECK: orchestrator/src/db.rs — schema change needed to add the column",
        None,
        json!({}),
    );

    let feedback = metadata.get("harness_feedback").unwrap();
    assert!(feedback["signals"]
        .as_array()
        .unwrap()
        .contains(&Value::String("worker_scope_check".to_string())));
    assert_eq!(feedback["quarantined"], false);
}

#[test]
fn worker_stuck_marker_is_recorded() {
    let metadata = annotate_event_metadata(
        "assistant_message",
        "STUCK: tried changing the import twice — cargo check still fails with E0433",
        None,
        json!({}),
    );

    let feedback = metadata.get("harness_feedback").unwrap();
    assert!(feedback["signals"]
        .as_array()
        .unwrap()
        .contains(&Value::String("worker_stuck".to_string())));
}

#[test]
fn completion_with_open_validation_gap_is_flagged_and_quarantined() {
    let metadata = annotate_event_metadata(
        "assistant_message",
        "Done. I updated the handler and everything works.",
        None,
        json!({
            "finish_reason": "end_turn",
            "validation_gate": {
                "required": "targeted_tests",
                "edits_observed": true,
                "validation_observed": false,
                "nudge_injected": true
            }
        }),
    );

    let feedback = metadata.get("harness_feedback").unwrap();
    assert!(feedback["signals"]
        .as_array()
        .unwrap()
        .contains(&Value::String("completion_without_validation".to_string())));
    assert_eq!(feedback["quarantined"], true);
    assert_eq!(feedback["quarantine_reason"], "contract_violation");
}

#[test]
fn mid_loop_tool_use_response_with_gap_is_not_flagged() {
    // finish_reason tool_use means the trajectory is still working; only a
    // final answer with the gap open is a contract violation.
    let metadata = annotate_event_metadata(
        "assistant_message",
        "Editing the handler now.",
        None,
        json!({
            "finish_reason": "tool_use",
            "validation_gate": {
                "edits_observed": true,
                "validation_observed": false
            }
        }),
    );

    let signals = metadata
        .get("harness_feedback")
        .and_then(|feedback| feedback.get("signals"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert!(!signals.contains(&Value::String("completion_without_validation".to_string())));
}

#[test]
fn validated_completion_is_not_flagged() {
    let metadata = annotate_event_metadata(
        "assistant_message",
        "Done. cargo test passed.",
        None,
        json!({
            "finish_reason": "end_turn",
            "validation_gate": {
                "edits_observed": true,
                "validation_observed": true
            }
        }),
    );

    let signals = metadata
        .get("harness_feedback")
        .and_then(|feedback| feedback.get("signals"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert!(!signals.contains(&Value::String("completion_without_validation".to_string())));
}

#[test]
fn context_ledger_path_typo_is_quarantined() {
    let metadata = annotate_event_metadata(
        "failed_attempt",
        "Read failed: /repo/orchestrator/src/context_leder/enums.rs not found",
        Some("No such file or directory"),
        json!({}),
    );

    let feedback = metadata.get("harness_feedback").unwrap();
    assert_eq!(feedback["schema_version"], HARNESS_FEEDBACK_SCHEMA_VERSION);
    assert_eq!(feedback["quarantined"], true);
    assert_eq!(feedback["quarantine_reason"], "poisoned_path");
    assert_eq!(feedback["memory_policy"], MEMORY_POLICY_EXCLUDE);
    assert!(feedback["signals"]
        .as_array()
        .unwrap()
        .contains(&Value::String("path_typo_context_ledger".to_string())));
    assert!(is_quarantined(&metadata));
}

#[test]
fn requested_path_mismatch_with_poisoned_path_is_quarantined() {
    let metadata = annotate_event_metadata(
        "failed_attempt",
        "path mismatch",
        None,
        json!({
            "harness": {
                "requested_path": "orchestrator/src/context_ledger.rs",
                "actual_path": "orchestrator/src/context_leder.rs"
            }
        }),
    );

    let feedback = metadata.get("harness_feedback").unwrap();
    assert_eq!(feedback["quarantined"], true);
    assert_eq!(feedback["quarantine_reason"], "poisoned_path");
    let signals = signal_names(&metadata);
    assert!(signals.contains(&"requested_path_mismatch"));
    assert!(signals.contains(&"poisoned_path_attempt"));
    assert!(signals.contains(&"path_typo_context_ledger"));
}

#[test]
fn false_path_success_is_quarantined() {
    let metadata = annotate_event_metadata(
        "assistant_message",
        "Read successful. Path: orchestrator/src/context_leder.rs. Safe to proceed.",
        None,
        json!({}),
    );

    let feedback = metadata.get("harness_feedback").unwrap();
    assert_eq!(feedback["quarantined"], true);
    assert_eq!(feedback["quarantine_reason"], "poisoned_path");
    let signals = signal_names(&metadata);
    assert!(signals.contains(&"false_path_success"));
    assert!(signals.contains(&"poisoned_path_attempt"));
}

#[test]
fn continued_after_stop_is_quarantined() {
    let metadata = annotate_event_metadata(
        "assistant_message",
        "Stop immediately. Do not retry. Read 1 file after the stop instruction.",
        None,
        json!({}),
    );

    let feedback = metadata.get("harness_feedback").unwrap();
    assert_eq!(feedback["quarantined"], true);
    assert_eq!(feedback["quarantine_reason"], "tool_recovery_failed");
    assert!(signal_names(&metadata).contains(&"continued_after_stop"));
}

#[test]
fn self_correction_after_violation_is_recorded_without_overriding_quarantine() {
    let metadata = annotate_event_metadata(
        "assistant_message",
        "The requested path was orchestrator/src/context_ledger.rs. I used the forbidden spelling orchestrator/src/context_leder.rs. Wait, failed.",
        None,
        json!({}),
    );

    let feedback = metadata.get("harness_feedback").unwrap();
    assert_eq!(feedback["quarantined"], true);
    assert_eq!(feedback["quarantine_reason"], "poisoned_path");
    let signals = signal_names(&metadata);
    assert!(signals.contains(&"self_correction_after_violation"));
    assert!(signals.contains(&"requested_path_mismatch"));
}

#[test]
fn guardrail_metadata_is_attached_to_blocked_events() {
    let metadata = annotate_event_metadata(
        "assistant_message",
        "Final report.",
        None,
        json!({
            "harness": {
                "required_tests": ["cargo test context_ledger"],
                "tests_run": []
            }
        }),
    );

    assert_eq!(metadata["harness_guardrail"]["action"], "block");
    assert_eq!(
        metadata["harness_guardrail"]["reason"],
        "contract_violation"
    );
    assert_eq!(metadata["harness_feedback"]["quarantined"], true);
}

#[test]
fn structured_manual_quarantine_is_preserved_with_bounded_reason() {
    let metadata = annotate_event_metadata(
        "checkpoint",
        "benchmark run",
        None,
        json!({
            "harness": {
                "signals": ["gate_blocked", "unknown-user-value"],
                "quarantine": true,
                "quarantine_reason": "raw user supplied reason"
            }
        }),
    );

    let feedback = metadata.get("harness_feedback").unwrap();
    assert_eq!(feedback["quarantined"], true);
    assert_eq!(feedback["quarantine_reason"], "manual_quarantine");
    assert!(feedback["signals"]
        .as_array()
        .unwrap()
        .contains(&Value::String("gate_blocked".to_string())));
    assert!(feedback["signals"]
        .as_array()
        .unwrap()
        .contains(&Value::String("unknown".to_string())));
}

#[test]
fn ordinary_events_are_not_annotated() {
    let metadata = annotate_event_metadata("user_message", "please update docs", None, json!({}));
    assert!(metadata.get("harness_feedback").is_none());
    assert!(!is_quarantined(&metadata));
}

#[test]
fn annotation_is_idempotent() {
    let once = annotate_event_metadata(
        "failed_attempt",
        "context_ledler path typo caused failed read",
        None,
        json!({}),
    );
    let twice = annotate_event_metadata(
        "failed_attempt",
        "context_ledler path typo caused failed read",
        None,
        once.clone(),
    );
    assert_eq!(once, twice);
}
