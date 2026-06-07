//! Deterministic trajectory lineage helpers.
//!
//! Trajectories are stored as ordinary `agent_events`. This module only adds
//! bounded enums, event builders, and pure aggregation logic so lineage remains
//! replayable from the event log without learned classification or new storage.

use crate::db::AgentEvent;
use serde_json::Value;

pub use crate::trajectory_events::{
    context_pack_event, make_request_metadata, model_response_metadata, trajectory_result_event,
};
pub use crate::trajectory_roles::{
    default_role_for_event_type, is_trajectory_participating_event_type, validate_event_role,
};
pub use crate::trajectory_summary::{summarize_trajectory, TrajectoryResultSummary};
pub use crate::trajectory_types::{
    BoundaryReason, EventRole, FinalStatus, TrajectoryContext, DEFAULT_TRAJECTORY_IDLE_TIMEOUT_SEC,
    EVENT_ROLES, EVENT_TYPE_CONTEXT_PACK, EVENT_TYPE_TRAJECTORY_RESULT, FINAL_STATUSES,
};

pub fn optional_token_counts_from_value(value: &Value) -> (Option<i64>, Option<i64>) {
    let usage = value.get("usage");
    let input = usage
        .and_then(|u| u.get("prompt_tokens"))
        .or_else(|| usage.and_then(|u| u.get("input_tokens")))
        .and_then(Value::as_i64);
    let output = usage
        .and_then(|u| u.get("completion_tokens"))
        .or_else(|| usage.and_then(|u| u.get("output_tokens")))
        .and_then(Value::as_i64);
    (input, output)
}

pub fn model_finish_reason(value: &Value) -> Option<String> {
    value["choices"][0]["finish_reason"]
        .as_str()
        .or_else(|| value["stop_reason"].as_str())
        .map(str::to_string)
}

pub fn next_attempt_index(
    events: &[AgentEvent],
    current_attempt: i32,
    next_role: EventRole,
) -> i32 {
    if next_role != EventRole::ModelResponse {
        return current_attempt.max(1);
    }
    let latest_model_response_at = events
        .iter()
        .rposition(|event| event.event_role.as_deref() == Some(EventRole::ModelResponse.as_str()));
    let start = latest_model_response_at.map(|idx| idx + 1).unwrap_or(0);
    if events[start..].iter().any(is_failed_terminal_event) {
        current_attempt.max(1) + 1
    } else {
        current_attempt.max(1)
    }
}

pub fn is_failed_terminal_event(event: &AgentEvent) -> bool {
    match event.event_role.as_deref().and_then(EventRole::from_str) {
        Some(EventRole::Failure) => true,
        Some(EventRole::Validation) => !event_success(event),
        Some(EventRole::Patch) => {
            let payload = event_payload(event);
            payload_bool(payload, "patch_reverted").unwrap_or(false)
                || !payload_bool(payload, "patch_applied").unwrap_or_else(|| event_success(event))
        }
        Some(EventRole::Remediation) => !event_success(event),
        _ => false,
    }
}

fn event_success(event: &AgentEvent) -> bool {
    event
        .metadata
        .get("success")
        .and_then(Value::as_bool)
        .or_else(|| event_payload(event).get("success").and_then(Value::as_bool))
        .or_else(|| event_payload(event).get("pass").and_then(Value::as_bool))
        .unwrap_or(false)
}

fn event_payload(event: &AgentEvent) -> &Value {
    event.metadata.get("payload").unwrap_or(&event.metadata)
}

fn payload_bool(payload: &Value, key: &str) -> Option<bool> {
    payload.get(key).and_then(Value::as_bool)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use uuid::Uuid;

    fn event(role: EventRole, attempt: i32, payload: Value, success: bool) -> AgentEvent {
        AgentEvent {
            id: Uuid::new_v4().to_string(),
            session_id: "s".to_string(),
            repo: "r".to_string(),
            actor: "a".to_string(),
            event_type: role.as_str().to_string(),
            summary: role.as_str().to_string(),
            evidence: None,
            metadata: json!({"success": success, "payload": payload}),
            correlation_id: None,
            parent_event_id: None,
            trajectory_id: Some(Uuid::nil()),
            attempt_index: Some(attempt),
            event_role: Some(role.as_str().to_string()),
            created_at: chrono::Utc::now(),
            summary_level: 0,
        }
    }

    #[test]
    fn event_roles_are_restricted() {
        for role in EVENT_ROLES {
            assert!(validate_event_role(Some(role)).is_ok());
        }
        assert!(validate_event_role(Some("score")).is_err());
    }

    #[test]
    fn model_response_retry_increments_only_after_failed_terminal() {
        let events = vec![
            event(EventRole::ModelResponse, 1, json!({}), true),
            event(EventRole::Validation, 1, json!({"pass": false}), false),
        ];
        assert_eq!(next_attempt_index(&events, 1, EventRole::ModelResponse), 2);

        let events = vec![
            event(EventRole::ModelResponse, 1, json!({}), true),
            event(EventRole::ContextPack, 1, json!({}), true),
        ];
        assert_eq!(next_attempt_index(&events, 1, EventRole::ModelResponse), 1);
    }

    #[test]
    fn trajectory_result_sums_non_null_tokens_and_marks_incomplete() {
        let tid = Uuid::new_v4();
        let mut a = event(EventRole::ModelResponse, 1, json!({}), true);
        a.trajectory_id = Some(tid);
        a.metadata = json!({"model_name": "m1", "input_tokens": 10, "output_tokens": 5});
        let mut b = event(EventRole::ModelResponse, 1, json!({}), true);
        b.trajectory_id = Some(tid);
        b.metadata = json!({"model_name": "m1", "input_tokens": null, "output_tokens": 7});

        let result = summarize_trajectory(tid, &[a, b], Some(BoundaryReason::IdleTimeout));
        assert_eq!(result.total_input_tokens, 10);
        assert_eq!(result.total_output_tokens, 12);
        assert!(!result.token_totals_complete);
    }

    #[test]
    fn trajectory_statuses_are_deterministic() {
        let tid = Uuid::new_v4();
        let succeeded = summarize_trajectory(
            tid,
            &[
                event(
                    EventRole::Patch,
                    1,
                    json!({"patch_applied": true, "patch_reverted": false}),
                    true,
                ),
                event(EventRole::Validation, 1, json!({"pass": true}), true),
            ],
            Some(BoundaryReason::NewUserMessage),
        );
        assert_eq!(succeeded.final_status, FinalStatus::Succeeded);

        let abandoned = summarize_trajectory(
            tid,
            &[event(EventRole::ModelResponse, 1, json!({}), true)],
            Some(BoundaryReason::NewUserMessage),
        );
        assert_eq!(abandoned.final_status, FinalStatus::Abandoned);

        let unresolved = summarize_trajectory(
            tid,
            &[event(EventRole::ModelResponse, 1, json!({}), true)],
            Some(BoundaryReason::IdleTimeout),
        );
        assert_eq!(unresolved.final_status, FinalStatus::Unresolved);

        let reverted = summarize_trajectory(
            tid,
            &[
                event(EventRole::Patch, 1, json!({"patch_applied": true}), true),
                event(EventRole::Patch, 1, json!({"patch_reverted": true}), false),
            ],
            Some(BoundaryReason::IdleTimeout),
        );
        assert_eq!(reverted.final_status, FinalStatus::Reverted);
    }
}
