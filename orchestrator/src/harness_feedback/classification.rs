use serde_json::Value;
use std::collections::BTreeSet;

use crate::harness_feedback_metadata::{
    bool_path, changed_file_count, is_poisoned_context_ledger_path, string_array_path, string_path,
};
use crate::harness_feedback_taxonomy::{bounded_quarantine_reason, bounded_signal_type};
use crate::harness_feedback_text_signals::collect_text_signals;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HarnessClassification {
    pub(crate) signals: BTreeSet<&'static str>,
    pub(crate) quarantined: bool,
    pub(crate) quarantine_reason: Option<&'static str>,
}

impl HarnessClassification {
    pub(crate) fn empty() -> Self {
        Self {
            signals: BTreeSet::new(),
            quarantined: false,
            quarantine_reason: None,
        }
    }

    pub(crate) fn signal(&mut self, signal: &'static str) {
        self.signals.insert(signal);
    }

    fn quarantine(&mut self, reason: &'static str) {
        self.quarantined = true;
        if self.quarantine_reason.is_none() {
            self.quarantine_reason = Some(reason);
        }
    }
}

pub(crate) fn classify_event(
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
