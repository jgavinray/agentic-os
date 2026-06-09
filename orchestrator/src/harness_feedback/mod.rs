//! Deterministic harness feedback classification.
//!
//! The harness feedback layer records bounded execution-harness signals on
//! `agent_events` metadata. It does not replace the audit log and it never
//! changes prompt assembly directly; quarantine only prevents known poisoned
//! harness artifacts from being reused as memory.

use serde_json::Value;

pub mod classification;
pub mod guardrail;
pub mod metadata;
pub mod store;
pub mod taxonomy;
pub mod text_signals;

use classification::classify_event;
pub use guardrail::{evaluate_runtime_guardrail, RuntimeGuardrailDecision};
use metadata::object_or_empty;
pub use store::{run_backfill, BackfillOptions, BackfillReport};
pub use taxonomy::{
    bounded_guardrail_action, bounded_guardrail_reason, bounded_learning_status,
    bounded_quarantine_reason, bounded_signal_type, GUARDRAIL_ACTION_ALLOW, GUARDRAIL_ACTION_BLOCK,
    GUARDRAIL_ACTION_TERMINATE, GUARDRAIL_ACTION_WARN, HARNESS_FEEDBACK_SCHEMA_VERSION,
    HARNESS_GUARDRAIL_ACTIONS, HARNESS_GUARDRAIL_REASONS, HARNESS_LEARNING_STATUSES,
    HARNESS_QUARANTINE_REASONS, HARNESS_SIGNAL_TYPES,
};
use taxonomy::{GUARDRAIL_REASON_NONE, MEMORY_POLICY_EXCLUDE, MEMORY_POLICY_INCLUDE};

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

#[cfg(test)]
mod tests;
