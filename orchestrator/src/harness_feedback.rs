//! Deterministic harness feedback classification.
//!
//! The harness feedback layer records bounded execution-harness signals on
//! `agent_events` metadata. It does not replace the audit log and it never
//! changes prompt assembly directly; quarantine only prevents known poisoned
//! harness artifacts from being reused as memory.

use serde_json::{json, Value};
use std::collections::BTreeSet;

use crate::harness_feedback_metadata::{
    bool_path, changed_file_count, contains_any, is_poisoned_context_ledger_path, object_or_empty,
    string_array_path, string_path,
};
pub use crate::harness_feedback_store::{run_backfill, BackfillOptions, BackfillReport};
pub use crate::harness_feedback_taxonomy::{
    bounded_guardrail_action, bounded_guardrail_reason, bounded_learning_status,
    bounded_quarantine_reason, bounded_signal_type, GUARDRAIL_ACTION_ALLOW, GUARDRAIL_ACTION_BLOCK,
    GUARDRAIL_ACTION_TERMINATE, GUARDRAIL_ACTION_WARN, HARNESS_FEEDBACK_SCHEMA_VERSION,
    HARNESS_GUARDRAIL_ACTIONS, HARNESS_GUARDRAIL_REASONS, HARNESS_LEARNING_STATUSES,
    HARNESS_QUARANTINE_REASONS, HARNESS_SIGNAL_TYPES,
};
use crate::harness_feedback_taxonomy::{
    guardrail_action_for_signals, GUARDRAIL_REASON_NONE, MEMORY_POLICY_EXCLUDE,
    MEMORY_POLICY_INCLUDE,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeGuardrailDecision {
    pub action: &'static str,
    pub reason: &'static str,
    pub should_stop: bool,
    pub signals: Vec<&'static str>,
}

impl RuntimeGuardrailDecision {
    fn allow(classification: HarnessClassification) -> Self {
        Self {
            action: GUARDRAIL_ACTION_ALLOW,
            reason: GUARDRAIL_REASON_NONE,
            should_stop: false,
            signals: classification.signals.iter().copied().collect(),
        }
    }

    fn from_classification(classification: HarnessClassification) -> Self {
        let signals: Vec<&'static str> = classification.signals.iter().copied().collect();
        let reason = classification
            .quarantine_reason
            .unwrap_or(GUARDRAIL_REASON_NONE);
        let action = guardrail_action_for_signals(&classification.signals);
        Self {
            action,
            reason,
            should_stop: action == GUARDRAIL_ACTION_TERMINATE,
            signals,
        }
    }

    pub fn as_json(&self) -> Value {
        json!({
            "schema_version": HARNESS_FEEDBACK_SCHEMA_VERSION,
            "action": self.action,
            "reason": self.reason,
            "should_stop": self.should_stop,
            "signals": self.signals,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HarnessClassification {
    signals: BTreeSet<&'static str>,
    quarantined: bool,
    quarantine_reason: Option<&'static str>,
}

impl HarnessClassification {
    fn empty() -> Self {
        Self {
            signals: BTreeSet::new(),
            quarantined: false,
            quarantine_reason: None,
        }
    }

    fn signal(&mut self, signal: &'static str) {
        self.signals.insert(signal);
    }

    fn quarantine(&mut self, reason: &'static str) {
        self.quarantined = true;
        if self.quarantine_reason.is_none() {
            self.quarantine_reason = Some(reason);
        }
    }
}

pub fn annotate_event_metadata(
    event_type: &str,
    summary: &str,
    evidence: Option<&str>,
    metadata: Value,
) -> Value {
    let classification = classify_event(event_type, summary, evidence, &metadata);
    let guardrail = RuntimeGuardrailDecision::from_classification(classification.clone());
    if classification.signals.is_empty()
        && !classification.quarantined
        && metadata.get("harness_feedback").is_none()
        && guardrail.action == GUARDRAIL_ACTION_ALLOW
    {
        return metadata;
    }

    let mut metadata = object_or_empty(metadata);
    let mut feedback = metadata
        .get("harness_feedback")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let signals: Vec<Value> = classification
        .signals
        .iter()
        .map(|signal| Value::String((*signal).to_string()))
        .collect();
    let learning_status = if classification.quarantined {
        "quarantined"
    } else if signals.is_empty() {
        "ignored"
    } else {
        "recorded"
    };

    feedback.insert(
        "schema_version".to_string(),
        Value::Number(HARNESS_FEEDBACK_SCHEMA_VERSION.into()),
    );
    feedback.insert("signals".to_string(), Value::Array(signals));
    feedback.insert(
        "signal_count".to_string(),
        Value::Number((classification.signals.len() as u64).into()),
    );
    feedback.insert(
        "quarantined".to_string(),
        Value::Bool(classification.quarantined),
    );
    feedback.insert(
        "quarantine_reason".to_string(),
        classification
            .quarantine_reason
            .map(|reason| Value::String(reason.to_string()))
            .unwrap_or(Value::Null),
    );
    feedback.insert(
        "memory_policy".to_string(),
        Value::String(
            if classification.quarantined {
                MEMORY_POLICY_EXCLUDE
            } else {
                MEMORY_POLICY_INCLUDE
            }
            .to_string(),
        ),
    );
    feedback.insert(
        "learning_status".to_string(),
        Value::String(learning_status.to_string()),
    );

    metadata["harness_feedback"] = Value::Object(feedback);
    if guardrail.action != GUARDRAIL_ACTION_ALLOW {
        metadata["harness_guardrail"] = guardrail.as_json();
    }
    metadata
}

pub fn evaluate_runtime_guardrail(
    event_type: &str,
    summary: &str,
    evidence: Option<&str>,
    metadata: &Value,
) -> RuntimeGuardrailDecision {
    let classification = classify_event(event_type, summary, evidence, metadata);
    if classification.signals.is_empty() && !classification.quarantined {
        RuntimeGuardrailDecision::allow(classification)
    } else {
        RuntimeGuardrailDecision::from_classification(classification)
    }
}

pub fn is_quarantined(metadata: &Value) -> bool {
    metadata
        .get("harness_feedback")
        .and_then(|value| value.get("quarantined"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || metadata
            .get("harness_feedback")
            .and_then(|value| value.get("memory_policy"))
            .and_then(Value::as_str)
            .is_some_and(|policy| policy == MEMORY_POLICY_EXCLUDE)
}

pub fn record_metadata_metrics(metadata: &Value) {
    if let Some(guardrail) = metadata.get("harness_guardrail") {
        let action = guardrail
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or(GUARDRAIL_ACTION_ALLOW);
        let reason = guardrail
            .get("reason")
            .and_then(Value::as_str)
            .unwrap_or(GUARDRAIL_REASON_NONE);
        crate::telemetry::record_harness_guardrail_decision(action, reason);
    }

    let Some(feedback) = metadata.get("harness_feedback") else {
        return;
    };
    let signals = feedback
        .get("signals")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for signal in signals {
        if let Some(signal) = signal.as_str() {
            crate::telemetry::record_harness_feedback_signal(signal);
        }
    }
    if is_quarantined(metadata) {
        let reason = feedback
            .get("quarantine_reason")
            .and_then(Value::as_str)
            .unwrap_or("manual_quarantine");
        crate::telemetry::record_harness_feedback_quarantine(reason);
    }
    let status = feedback
        .get("learning_status")
        .and_then(Value::as_str)
        .unwrap_or("ignored");
    crate::telemetry::record_harness_feedback_learning(status);
}

fn classify_event(
    event_type: &str,
    summary: &str,
    evidence: Option<&str>,
    metadata: &Value,
) -> HarnessClassification {
    let mut classification = HarnessClassification::empty();
    collect_structured_metadata(metadata, &mut classification);
    collect_text_signals(event_type, summary, evidence, &mut classification);
    apply_quarantine_rules(&mut classification);
    classification
}

fn collect_structured_metadata(metadata: &Value, classification: &mut HarnessClassification) {
    if bool_path(metadata, &["benchmark"]) || bool_path(metadata, &["harness", "benchmark"]) {
        classification.signal("benchmark_trace");
    }

    for path in [
        &["harness_feedback", "signals"][..],
        &["harness", "signals"][..],
        &["harness", "failure_signals"][..],
    ] {
        for signal in string_array_path(metadata, path) {
            classification.signal(bounded_signal_type(signal));
        }
    }

    for path in [
        &["harness_feedback", "signal"][..],
        &["harness", "signal"][..],
        &["harness", "failure_class"][..],
    ] {
        if let Some(signal) = string_path(metadata, path) {
            classification.signal(bounded_signal_type(signal));
        }
    }

    collect_structured_path_signals(metadata, classification);
    collect_structured_contract_signals(metadata, classification);

    if bool_path(metadata, &["harness_feedback", "quarantined"])
        || bool_path(metadata, &["harness", "quarantine"])
        || bool_path(metadata, &["harness", "quarantined"])
    {
        let reason = string_path(metadata, &["harness_feedback", "quarantine_reason"])
            .or_else(|| string_path(metadata, &["harness", "quarantine_reason"]))
            .map(bounded_quarantine_reason)
            .unwrap_or("manual_quarantine");
        classification.quarantine(reason);
    }
}

fn collect_structured_contract_signals(
    metadata: &Value,
    classification: &mut HarnessClassification,
) {
    let claimed_success = bool_path(metadata, &["harness", "claimed_success"])
        || bool_path(metadata, &["harness_feedback", "claimed_success"])
        || bool_path(metadata, &["harness", "final_report", "claimed_success"]);
    if claimed_success && changed_file_count(metadata) == Some(0) {
        classification.signal("success_without_change");
        classification.signal("fake_success_claim");
    }

    let required_tests = string_array_path(metadata, &["harness", "required_tests"]);
    if !required_tests.is_empty() {
        let tests_run = string_array_path(metadata, &["harness", "tests_run"]);
        if tests_run.is_empty()
            || required_tests
                .iter()
                .any(|required| !tests_run.iter().any(|ran| ran == required))
        {
            classification.signal("missing_required_validation");
        }
    }
}

fn collect_structured_path_signals(metadata: &Value, classification: &mut HarnessClassification) {
    let requested_path = string_path(metadata, &["harness", "requested_path"])
        .or_else(|| string_path(metadata, &["harness", "requested_file"]))
        .or_else(|| string_path(metadata, &["harness_feedback", "requested_path"]));
    let actual_path = string_path(metadata, &["harness", "actual_path"])
        .or_else(|| string_path(metadata, &["harness", "tool_path"]))
        .or_else(|| string_path(metadata, &["harness_feedback", "actual_path"]));

    if let (Some(requested_path), Some(actual_path)) = (requested_path, actual_path) {
        if requested_path != actual_path {
            classification.signal("requested_path_mismatch");
        }
        if is_poisoned_context_ledger_path(actual_path) {
            classification.signal("poisoned_path_attempt");
            classification.signal("path_typo_context_ledger");
        }
    }
}

fn collect_text_signals(
    event_type: &str,
    summary: &str,
    evidence: Option<&str>,
    classification: &mut HarnessClassification,
) {
    let haystack = format!(
        "{}\n{}\n{}",
        event_type,
        summary,
        evidence.unwrap_or_default()
    )
    .to_ascii_lowercase();

    let has_correct_context_ledger_path = haystack.contains("context_ledger");
    let has_poisoned_context_ledger_path = contains_any(
        &haystack,
        &["context_leder", "context_ledler", "context-ledger"],
    );

    if has_poisoned_context_ledger_path {
        classification.signal("path_typo_context_ledger");
        classification.signal("poisoned_path_attempt");
    }
    if has_correct_context_ledger_path && has_poisoned_context_ledger_path {
        classification.signal("requested_path_mismatch");
    }
    if contains_any(
        &haystack,
        &[
            "no such file or directory",
            "file not found",
            "read tool: file",
        ],
    ) && contains_any(
        &haystack,
        &["context_ledger", "context_leder", "context_ledler", ".tmp/"],
    ) {
        classification.signal("read_missing_path");
    }
    if contains_any(&haystack, &["error writing file", "error editing file"]) {
        classification.signal("edit_failed");
    }
    if contains_any(
        &haystack,
        &["[fact-forcing gate]", "gateguard", "fact-forcing gate"],
    ) {
        classification.signal("gate_blocked");
    }
    if haystack.contains("gate") && contains_any(&haystack, &["fallback", "failed to recover"]) {
        classification.signal("gate_recovery_failed");
    }
    if haystack.contains("gate") && contains_any(&haystack, &["recovered", "satisfy gateguard"]) {
        classification.signal("gate_recovered");
    }
    if contains_any(&haystack, &["do not use bash", "bash fallback"]) {
        classification.signal("forbidden_tool_fallback");
    }
    if contains_any(
        &haystack,
        &["do not retry", "do not search", "do not guess paths"],
    ) && contains_any(&haystack, &["retry", "searched for", "finds the actual"])
    {
        classification.signal("command_boundary_violation");
    }
    if contains_any(
        &haystack,
        &["invalid patch", "fake hash", "a1b2c3d4e5f60718"],
    ) || (haystack.contains("unified diff patch")
        && contains_any(&haystack, &["context_leder", "context_ledler"]))
    {
        classification.signal("patch_invalid");
    }
    if contains_any(
        &haystack,
        &["claimed success", "false confidence", "misleading"],
    ) && contains_any(&haystack, &["test", "green", "passed"])
    {
        classification.signal("fake_success_claim");
    }
    if has_poisoned_context_ledger_path
        && contains_any(
            &haystack,
            &[
                "read successful",
                "file read successfully",
                "safe to proceed",
                "file changed",
                "edit succeeded",
            ],
        )
    {
        classification.signal("false_path_success");
    }
    if contains_any(
        &haystack,
        &["made no progress", "no progress", "still incomplete"],
    ) {
        classification.signal("no_progress_timeout");
    }
    if contains_any(&haystack, &["sanity check: passed", "sanity_check=passed"]) {
        classification.signal("diagnosis_success");
    }
    if contains_any(
        &haystack,
        &["hard-stop compliance", "stopped after the first failure"],
    ) {
        classification.signal("hard_stop_compliance");
    }
    if contains_any(
        &haystack,
        &[
            "stop immediately",
            "do not retry",
            "do not continue",
            "do not attempt recovery",
            "stop after",
        ],
    ) && contains_any(
        &haystack,
        &[
            "read 1 file",
            "read tool",
            "bash(",
            "searched for",
            "let me re-read",
            "retry",
        ],
    ) {
        classification.signal("continued_after_stop");
    }
    if contains_any(
        &haystack,
        &[
            "scope creep",
            "outside the allowed files",
            "scope violation",
        ],
    ) {
        classification.signal("scope_violation");
    }
    if contains_any(
        &haystack,
        &["enum_variant_hash", "placeholder", "early return"],
    ) && contains_any(&haystack, &["test", "validation"])
    {
        classification.signal("misleading_green_tests");
    }
    if contains_any(
        &haystack,
        &[
            "poisoned memory",
            "poisoned cache",
            "memory/cache is perhaps poisoned",
        ],
    ) {
        classification.signal("memory_poison_indicator");
    }
    if has_poisoned_context_ledger_path
        && has_correct_context_ledger_path
        && contains_any(
            &haystack,
            &[
                "wait",
                "i need to be more careful",
                "let me be more careful",
                "failed",
                "forbidden spelling",
            ],
        )
    {
        classification.signal("self_correction_after_violation");
    }
}

fn apply_quarantine_rules(classification: &mut HarnessClassification) {
    if classification.signals.contains("poisoned_path_attempt")
        || classification.signals.contains("path_typo_context_ledger")
        || classification.signals.contains("requested_path_mismatch")
    {
        classification.quarantine("poisoned_path");
    } else if classification.signals.contains("repeated_invalid_path") {
        classification.quarantine("repeated_invalid_path");
    } else if classification.signals.contains("patch_invalid") {
        classification.quarantine("patch_invalid");
    } else if classification.signals.contains("success_without_change")
        || classification
            .signals
            .contains("missing_required_validation")
    {
        classification.quarantine("contract_violation");
    } else if classification.signals.contains("fake_success_claim")
        || classification.signals.contains("false_path_success")
    {
        classification.quarantine("fake_success_claim");
    } else if classification.signals.contains("gate_recovery_failed")
        || classification.signals.contains("forbidden_tool_fallback")
        || classification
            .signals
            .contains("command_boundary_violation")
        || classification.signals.contains("continued_after_stop")
        || classification.signals.contains("no_progress_timeout")
    {
        classification.quarantine("tool_recovery_failed");
    } else if classification.signals.contains("memory_poison_indicator") {
        classification.quarantine("memory_poison_indicator");
    } else if classification.signals.contains("benchmark_trace") {
        classification.quarantine("benchmark_session");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let metadata =
            annotate_event_metadata("user_message", "please update docs", None, json!({}));
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
}
