//! Deterministic execution feedback capture.
//!
//! This module deliberately avoids LLM calls. It turns observed tool output and
//! explicit validation reports into structured `agent_events` payloads using
//! regexes, fixed rule ordering, and small parser helpers. The resulting events
//! can then flow through the existing Postgres, Qdrant, FTS, and metrics paths.

use crate::db::AgentEvent;
use uuid::Uuid;

pub mod chain;
pub mod compile_payloads;
pub mod events;
pub mod fingerprint_rules;
pub mod fingerprints;
pub mod labels;
pub mod lint_payloads;
pub mod parsing;
pub mod payloads;
pub mod reports;
pub mod signature_payloads;
pub mod summaries;
pub mod test_payloads;
pub mod tool_events;
pub mod tool_result;
pub mod validation_payloads;
pub mod validation_result_payloads;
pub mod validators;

pub use chain::{group_by_parent, patch_validation_ids_resolve};
pub use events::{
    build_execution_event, is_failure_outcome_event_type, ExecutionEventContext,
    ExecutionEventKind, EVENT_TYPE_COMPILE_RESULT, EVENT_TYPE_LINT_RESULT, EVENT_TYPE_PATCH_RESULT,
    EVENT_TYPE_REMEDIATION, EVENT_TYPE_TEST_RESULT, EVENT_TYPE_TOOL_RESULT,
    EVENT_TYPE_VALIDATION_RESULT, EXECUTION_EVENT_TYPES, FAILURE_OUTCOME_EVENT_TYPES,
};
pub use fingerprints::{
    backfill_inline_signature_metadata, extract_failure_signatures, fingerprint,
    inline_signature_payload, outcome_raw_text_from_payload, signature_category, summarize_text,
    FailureFingerprint, FINGERPRINT_VERSION,
};
pub use labels::{
    bounded_failure_category_label, bounded_failure_signature_label, bounded_validator_label,
    retry_trigger_category_from_payload, retry_trigger_category_from_signature, task_retry_type,
};
pub use parsing::{tool_results_from_sse, tool_results_from_value, CapturedToolResult};
pub use payloads::{
    compile_result_payload, lint_result_payload, patch_result_payload, remediation_payload,
    test_result_payload, tool_result_payload, validation_result_payload,
};
pub use reports::{
    events_for_validation_report, ValidationReportRequest, ValidationReportResponse,
};
use tool_events::{derive_tool_event, tool_payload};
pub use validators::{
    classify_validator, validate_validator_type, ValidationKind, ValidatorSpec, VALIDATOR_TYPES,
};

pub fn events_for_tool_result(
    ctx: &ExecutionEventContext,
    result: &CapturedToolResult,
) -> Vec<AgentEvent> {
    let derivation = derive_tool_event(result);
    let tool_event = build_execution_event(
        ctx,
        derivation.tool_kind,
        derivation.tool_success,
        tool_payload(result, &derivation),
    );
    let mut events = vec![tool_event.clone()];

    if let Some(validator_event) = derivation.validator_event {
        // A validator tool produces two linked memories: the observed tool run
        // and the normalized compile/test/lint/validation result derived from it.
        let validation_ctx = ctx.child_of(Uuid::parse_str(&tool_event.id).unwrap());
        let validation_event = build_execution_event(
            &validation_ctx,
            validator_event.kind,
            validator_event.success,
            validator_event.payload,
        );
        events.push(validation_event);
    }

    events
}

#[cfg(test)]
mod tests {
    pub(crate) use super::*;

    mod capture;
    mod event;
    mod fingerprint;
}
