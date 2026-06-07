use super::*;
use serde_json::{json, Value};

fn signal_names(metadata: &Value) -> Vec<&str> {
    metadata["harness_feedback"]["signals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect()
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
fn runtime_guardrail_blocks_poisoned_path_attempt() {
    let decision = evaluate_runtime_guardrail(
        "tool_intent",
        "read path",
        None,
        &json!({
            "harness": {
                "requested_path": "orchestrator/src/context_ledger.rs",
                "actual_path": "orchestrator/src/context_leder.rs"
            }
        }),
    );

    assert_eq!(decision.action, GUARDRAIL_ACTION_BLOCK);
    assert_eq!(decision.reason, "poisoned_path");
    assert!(!decision.should_stop);
    assert!(decision.signals.contains(&"requested_path_mismatch"));
    assert!(decision.signals.contains(&"poisoned_path_attempt"));
}

#[test]
fn runtime_guardrail_terminates_after_stop_violation() {
    let decision = evaluate_runtime_guardrail(
        "assistant_message",
        "Stop immediately. Do not retry. Read 1 file after the stop instruction.",
        None,
        &json!({}),
    );

    assert_eq!(decision.action, GUARDRAIL_ACTION_TERMINATE);
    assert_eq!(decision.reason, "tool_recovery_failed");
    assert!(decision.should_stop);
    assert!(decision.signals.contains(&"continued_after_stop"));
}

#[test]
fn runtime_guardrail_blocks_success_without_change() {
    let decision = evaluate_runtime_guardrail(
        "assistant_message",
        "Final report: edit succeeded.",
        None,
        &json!({
            "harness": {
                "claimed_success": true,
                "files_changed": 0
            }
        }),
    );

    assert_eq!(decision.action, GUARDRAIL_ACTION_BLOCK);
    assert_eq!(decision.reason, "contract_violation");
    assert!(decision.signals.contains(&"success_without_change"));
}

#[test]
fn runtime_guardrail_blocks_missing_required_validation() {
    let decision = evaluate_runtime_guardrail(
        "assistant_message",
        "Final report.",
        None,
        &json!({
            "harness": {
                "required_tests": ["cargo fmt --check", "cargo test context_ledger"],
                "tests_run": ["cargo fmt --check"]
            }
        }),
    );

    assert_eq!(decision.action, GUARDRAIL_ACTION_BLOCK);
    assert_eq!(decision.reason, "contract_violation");
    assert!(decision.signals.contains(&"missing_required_validation"));
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
