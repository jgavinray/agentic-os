use crate::db::AgentEvent;
use crate::trajectory_event_payload::{event_payload, metadata_i64, metadata_str, payload_str};
use crate::trajectory_summary_status::derive_status;
use crate::trajectory_types::{BoundaryReason, EventRole, FinalStatus};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use uuid::Uuid;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrajectoryResultSummary {
    pub trajectory_id: Uuid,
    pub final_status: FinalStatus,
    pub success: bool,
    pub retry_count: i64,
    pub total_model_calls: i64,
    pub total_tool_calls: i64,
    pub total_validations: i64,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub token_totals_complete: bool,
    pub total_latency_ms: i64,
    pub models_used: Vec<String>,
    pub validators_used: Vec<String>,
    pub files_touched: Vec<String>,
    pub failure_signatures: Vec<String>,
    pub remediation_count: i64,
    pub final_attempt_index: i32,
}

impl TrajectoryResultSummary {
    pub fn payload(&self) -> Value {
        json!({
            "trajectory_id": self.trajectory_id,
            "final_status": self.final_status.as_str(),
            "success": self.success,
            "retry_count": self.retry_count,
            "total_model_calls": self.total_model_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_validations": self.total_validations,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "token_totals_complete": self.token_totals_complete,
            "total_latency_ms": self.total_latency_ms,
            "models_used": self.models_used,
            "validators_used": self.validators_used,
            "files_touched": self.files_touched,
            "failure_signatures": self.failure_signatures,
            "remediation_count": self.remediation_count,
            "final_attempt_index": self.final_attempt_index,
        })
    }
}

pub fn summarize_trajectory(
    trajectory_id: Uuid,
    events: &[AgentEvent],
    boundary_reason: Option<BoundaryReason>,
) -> TrajectoryResultSummary {
    let mut max_attempt = 1;
    let mut model_calls = 0;
    let mut tool_calls = 0;
    let mut validations = 0;
    let mut input_total = 0;
    let mut output_total = 0;
    let mut token_totals_complete = true;
    let mut latency_total = 0;
    let mut models = BTreeSet::new();
    let mut validators = BTreeSet::new();
    let mut files = BTreeSet::new();
    let mut signatures = BTreeSet::new();
    let mut remediation_count = 0;

    for event in events {
        if let Some(attempt) = event.attempt_index {
            max_attempt = max_attempt.max(attempt);
        }
        let role = event.event_role.as_deref().and_then(EventRole::from_str);
        match role {
            Some(EventRole::ModelResponse) => {
                model_calls += 1;
                if let Some(model) = metadata_str(&event.metadata, "model_name") {
                    models.insert(model.to_string());
                }
                match metadata_i64(&event.metadata, "input_tokens") {
                    Some(value) => input_total += value,
                    None => token_totals_complete = false,
                }
                match metadata_i64(&event.metadata, "output_tokens") {
                    Some(value) => output_total += value,
                    None => token_totals_complete = false,
                }
                if let Some(value) = metadata_i64(&event.metadata, "latency_ms") {
                    latency_total += value;
                }
            }
            Some(EventRole::ToolCall) | Some(EventRole::ToolResult) => tool_calls += 1,
            Some(EventRole::Validation) => {
                validations += 1;
                let payload = event_payload(event);
                if let Some(name) = payload_str(payload, "validator_name")
                    .or_else(|| payload_str(payload, "framework"))
                    .or_else(|| payload_str(payload, "tool_name"))
                    .or_else(|| payload_str(payload, "target"))
                {
                    validators.insert(name.to_string());
                }
            }
            Some(EventRole::Patch) => {
                if let Some(arr) = event_payload(event)
                    .get("files_touched")
                    .and_then(Value::as_array)
                {
                    for file in arr.iter().filter_map(Value::as_str) {
                        files.insert(file.to_string());
                    }
                }
            }
            Some(EventRole::Remediation) => remediation_count += 1,
            _ => {}
        }

        let payload = event_payload(event);
        if let Some(signature) = payload_str(payload, "failure_signature")
            .or_else(|| payload_str(payload, "signature"))
            .filter(|signature| *signature != "unknown")
        {
            signatures.insert(signature.to_string());
        }
    }

    let final_status = derive_status(events, boundary_reason);
    TrajectoryResultSummary {
        trajectory_id,
        final_status,
        success: final_status == FinalStatus::Succeeded,
        retry_count: i64::from(max_attempt.saturating_sub(1)),
        total_model_calls: model_calls,
        total_tool_calls: tool_calls,
        total_validations: validations,
        total_input_tokens: input_total,
        total_output_tokens: output_total,
        token_totals_complete,
        total_latency_ms: latency_total,
        models_used: models.into_iter().collect(),
        validators_used: validators.into_iter().collect(),
        files_touched: files.into_iter().collect(),
        failure_signatures: signatures.into_iter().collect(),
        remediation_count,
        final_attempt_index: max_attempt,
    }
}
