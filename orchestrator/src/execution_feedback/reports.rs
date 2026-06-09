use crate::db::AgentEvent;
use crate::execution_feedback::events_for_tool_result;
use crate::execution_feedback_events::{
    build_execution_event, is_failure_outcome_event_type, ExecutionEventContext, ExecutionEventKind,
};
use crate::execution_feedback_fingerprints::inline_signature_payload_from_fingerprint;
use crate::execution_feedback_fingerprints::{
    fingerprint, inline_signature_payload, summarize_text,
};
use crate::execution_feedback_parsing::{exit_code_from_text, CapturedToolResult};
use crate::execution_feedback_payloads::validation_result_payload;
use crate::execution_feedback_summaries::event_summary;
use crate::execution_feedback_validators::{
    classify_validator, validate_validator_type, ValidationKind, ValidatorSpec,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct ValidationReportRequest {
    pub session_id: Option<String>,
    pub repo: String,
    pub task: String,
    pub actor: Option<String>,
    pub event_type: Option<String>,
    pub validator_name: String,
    pub validator_type: Option<String>,
    pub success: Option<bool>,
    pub exit_code: Option<i32>,
    pub content: Option<String>,
    pub payload: Option<Value>,
    pub correlation_id: Option<Uuid>,
    pub parent_event_id: Option<Uuid>,
    pub trajectory_id: Option<Uuid>,
    pub attempt_index: Option<i32>,
}

#[derive(Debug, Serialize)]
pub struct ValidationReportResponse {
    pub captured: bool,
    pub event_ids: Vec<String>,
    pub qdrant_indexed: bool,
}

pub fn events_for_validation_report(
    ctx: &ExecutionEventContext,
    report: &ValidationReportRequest,
) -> Result<Vec<AgentEvent>, String> {
    validate_validator_type(report.validator_type.as_deref())?;
    let content = report.content.as_deref().unwrap_or("");
    if let Some(event_type) = report.event_type.as_deref() {
        if let Some(kind) = ExecutionEventKind::from_str(event_type) {
            // Advanced clients can submit an already-normalized artifact event.
            // We still wrap it with the standard event envelope for consistency.
            let mut payload = report
                .payload
                .clone()
                .unwrap_or_else(|| validation_result_payload(&report.validator_name, true, ""));
            let success = report.success.unwrap_or_else(|| {
                payload
                    .get("success")
                    .and_then(Value::as_bool)
                    .unwrap_or(true)
            });
            if is_failure_outcome_event_type(kind.as_str()) {
                payload = inline_signature_payload(payload, success, content);
            }
            return Ok(vec![build_execution_event(ctx, kind, success, payload)]);
        }
    }

    let tool_result = CapturedToolResult {
        tool_name: report.validator_name.clone(),
        content: content.to_string(),
        exit_code: report
            .exit_code
            .unwrap_or_else(|| exit_code_from_text(content).unwrap_or(0)),
        duration_ms: 0,
        stdout_summary: summarize_text(content, 500),
        stderr_summary: String::new(),
    };

    let spec = explicit_validator_spec(report.validator_type.as_deref())
        .or_else(|| classify_validator(&report.validator_name, content));

    // If the validator maps to a known tool, reuse the stream parser path so
    // Path A and Path B produce equivalent event chains.
    let mut events = if spec.is_some() {
        events_for_tool_result(ctx, &tool_result)
    } else {
        vec![build_execution_event(
            ctx,
            ExecutionEventKind::ValidationResult,
            report.success.unwrap_or(true),
            report.payload.clone().unwrap_or_else(|| {
                validation_result_payload(
                    &report.validator_name,
                    report.success.unwrap_or(true),
                    report
                        .content
                        .as_deref()
                        .filter(|_| !report.success.unwrap_or(true))
                        .unwrap_or(""),
                )
            }),
        )]
    };

    if let Some(success) = report.success {
        // Explicit reports are allowed to override parser-derived success when
        // the client has stronger knowledge than our generic text parser.
        let override_fp = (!success).then(|| fingerprint(content));
        for event in &mut events {
            event.metadata["success"] = json!(success);
            if is_failure_outcome_event_type(&event.event_type) {
                let payload = event.metadata["payload"].clone();
                event.metadata["payload"] = inline_signature_payload_from_fingerprint(
                    payload,
                    success,
                    override_fp.as_ref(),
                );
                event.summary =
                    event_summary(&event.event_type, success, &event.metadata["payload"]);
                event.evidence = Some(summarize_text(&event.metadata["payload"].to_string(), 1000));
            }
        }
    }
    Ok(events)
}

fn explicit_validator_spec(kind: Option<&str>) -> Option<ValidatorSpec> {
    kind.and_then(|kind| match kind {
        "compile" => Some(ValidatorSpec {
            validator: "explicit",
            kind: ValidationKind::Compile,
        }),
        "test" => Some(ValidatorSpec {
            validator: "explicit",
            kind: ValidationKind::Test,
        }),
        "lint" => Some(ValidatorSpec {
            validator: "explicit",
            kind: ValidationKind::Lint,
        }),
        "type_check" => Some(ValidatorSpec {
            validator: "explicit",
            kind: ValidationKind::TypeCheck,
        }),
        "schema" => Some(ValidatorSpec {
            validator: "explicit",
            kind: ValidationKind::Schema,
        }),
        "static_analysis" => Some(ValidatorSpec {
            validator: "explicit",
            kind: ValidationKind::StaticAnalysis,
        }),
        "other" => Some(ValidatorSpec {
            validator: "explicit",
            kind: ValidationKind::Other,
        }),
        _ => None,
    })
}
