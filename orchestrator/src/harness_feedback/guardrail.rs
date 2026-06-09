use serde_json::{json, Value};

use crate::harness_feedback_classification::{classify_event, HarnessClassification};
use crate::harness_feedback_taxonomy::{
    guardrail_action_for_signals, GUARDRAIL_ACTION_ALLOW, GUARDRAIL_ACTION_TERMINATE,
    GUARDRAIL_REASON_NONE, HARNESS_FEEDBACK_SCHEMA_VERSION,
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

    pub(crate) fn from_classification(classification: HarnessClassification) -> Self {
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
