use super::*;

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
