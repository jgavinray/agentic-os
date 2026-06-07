//! Deterministic trajectory lineage helpers.
//!
//! Trajectories are stored as ordinary `agent_events`. This module only adds
//! bounded enums, event builders, and pure aggregation logic so lineage remains
//! replayable from the event log without learned classification or new storage.

use crate::db::AgentEvent;
use serde_json::{json, Value};
use uuid::Uuid;

pub use crate::trajectory_summary::{summarize_trajectory, TrajectoryResultSummary};
pub use crate::trajectory_types::{
    BoundaryReason, EventRole, FinalStatus, TrajectoryContext, DEFAULT_TRAJECTORY_IDLE_TIMEOUT_SEC,
    EVENT_ROLES, EVENT_TYPE_CONTEXT_PACK, EVENT_TYPE_TRAJECTORY_RESULT, FINAL_STATUSES,
};

pub fn validate_event_role(role: Option<&str>) -> Result<(), anyhow::Error> {
    if let Some(role) = role {
        if EventRole::from_str(role).is_none() {
            anyhow::bail!("invalid event_role `{role}`");
        }
    }
    Ok(())
}

pub fn is_trajectory_participating_event_type(event_type: &str) -> bool {
    matches!(
        event_type,
        "user_message"
            | "assistant_message"
            | "failed_attempt"
            | EVENT_TYPE_CONTEXT_PACK
            | EVENT_TYPE_TRAJECTORY_RESULT
            | crate::execution_feedback::EVENT_TYPE_TOOL_RESULT
            | crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT
            | crate::execution_feedback::EVENT_TYPE_TEST_RESULT
            | crate::execution_feedback::EVENT_TYPE_LINT_RESULT
            | crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT
            | crate::execution_feedback::EVENT_TYPE_PATCH_RESULT
            | crate::execution_feedback::EVENT_TYPE_REMEDIATION
    )
}

pub fn default_role_for_event_type(event_type: &str) -> Option<EventRole> {
    Some(match event_type {
        "user_message" => EventRole::Request,
        "assistant_message" => EventRole::ModelResponse,
        "failed_attempt" => EventRole::Failure,
        EVENT_TYPE_CONTEXT_PACK => EventRole::ContextPack,
        EVENT_TYPE_TRAJECTORY_RESULT => EventRole::TrajectoryResult,
        crate::execution_feedback::EVENT_TYPE_TOOL_RESULT => EventRole::ToolResult,
        crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT
        | crate::execution_feedback::EVENT_TYPE_TEST_RESULT
        | crate::execution_feedback::EVENT_TYPE_LINT_RESULT
        | crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT => EventRole::Validation,
        crate::execution_feedback::EVENT_TYPE_PATCH_RESULT => EventRole::Patch,
        crate::execution_feedback::EVENT_TYPE_REMEDIATION => EventRole::Remediation,
        _ => return None,
    })
}

pub fn make_request_metadata(mut metadata: Option<Value>, trajectory: TrajectoryContext) -> Value {
    let mut out = metadata.take().unwrap_or_else(|| json!({}));
    out["event_type"] = json!("user_message");
    out["event_role"] = json!(EventRole::Request.as_str());
    out["trajectory_id"] = json!(trajectory.trajectory_id);
    out["attempt_index"] = json!(trajectory.attempt_index);
    out
}

#[allow(clippy::too_many_arguments)]
pub fn context_pack_event(
    session_id: &str,
    repo: &str,
    task: &str,
    trajectory: TrajectoryContext,
    parent_event_id: Option<Uuid>,
    stats: &crate::state::ContextPackStats,
    token_budget: usize,
) -> AgentEvent {
    let event_id = Uuid::new_v4();
    let truncated = stats.truncated;
    let payload = json!({
        "context_pack_id": event_id,
        "trajectory_id": trajectory.trajectory_id,
        "attempt_index": trajectory.attempt_index,
        "total_context_tokens": stats.context_tokens_estimate,
        "stable_prefix_hash": stats.stable_prefix_hash,
        "dynamic_tail_hash": stats.dynamic_tail_hash,
        "retrieved_event_ids": stats.retrieved_event_ids,
        "retrieval_strategy": "layered_rrf_decay",
        "semantic_hits": stats.retrieval_semantic_hits,
        "keyword_hits": stats.retrieval_fts_hits,
        "reranked_hits": stats.retrieval_deduped_hits,
        "memory_levels_used": stats.memory_levels_used,
        "injected_failure_signatures": stats.injected_failure_signatures,
        "token_budget": token_budget,
        "truncated": truncated,
    });
    AgentEvent {
        id: event_id.to_string(),
        session_id: session_id.to_string(),
        repo: repo.to_string(),
        actor: "context_builder".to_string(),
        event_type: EVENT_TYPE_CONTEXT_PACK.to_string(),
        summary: format!(
            "context pack tokens={} retrieved={} truncated={}",
            stats.context_tokens_estimate,
            stats.retrieved_event_ids.len(),
            truncated
        ),
        evidence: None,
        metadata: json!({
            "event_type": EVENT_TYPE_CONTEXT_PACK,
            "event_role": EventRole::ContextPack.as_str(),
            "trajectory_id": trajectory.trajectory_id,
            "attempt_index": trajectory.attempt_index,
            "repo": repo,
            "task": task,
            "payload": payload,
        }),
        correlation_id: Some(trajectory.trajectory_id),
        parent_event_id,
        trajectory_id: Some(trajectory.trajectory_id),
        attempt_index: Some(trajectory.attempt_index),
        event_role: Some(EventRole::ContextPack.as_str().to_string()),
        created_at: chrono::Utc::now(),
        summary_level: 0,
    }
}

pub fn trajectory_result_event(
    session_id: &str,
    repo: &str,
    summary: TrajectoryResultSummary,
) -> AgentEvent {
    let payload = summary.payload();
    AgentEvent {
        id: Uuid::new_v4().to_string(),
        session_id: session_id.to_string(),
        repo: repo.to_string(),
        actor: "trajectory".to_string(),
        event_type: EVENT_TYPE_TRAJECTORY_RESULT.to_string(),
        summary: format!(
            "trajectory {} status={} attempts={}",
            summary.trajectory_id,
            summary.final_status.as_str(),
            summary.final_attempt_index
        ),
        evidence: None,
        metadata: json!({
            "event_type": EVENT_TYPE_TRAJECTORY_RESULT,
            "event_role": EventRole::TrajectoryResult.as_str(),
            "trajectory_id": summary.trajectory_id,
            "attempt_index": summary.final_attempt_index,
            "payload": payload,
        }),
        correlation_id: Some(summary.trajectory_id),
        parent_event_id: None,
        trajectory_id: Some(summary.trajectory_id),
        attempt_index: Some(summary.final_attempt_index),
        event_role: Some(EventRole::TrajectoryResult.as_str().to_string()),
        created_at: chrono::Utc::now(),
        summary_level: 0,
    }
}

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

#[allow(clippy::too_many_arguments)]
pub fn model_response_metadata(
    model_name: &str,
    provider: &str,
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    latency_ms: Option<i64>,
    finish_reason: Option<String>,
    sampling_metadata: Option<Value>,
    context_pack_id: Option<Uuid>,
    trajectory: Option<TrajectoryContext>,
) -> Value {
    let mut metadata = json!({
        "model_name": model_name,
        "provider": provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "finish_reason": finish_reason,
        "sampling_params": Value::Null,
        "forwarded_sampling_params": Value::Null,
        "context_pack_id": context_pack_id,
    });
    if let Some(sampling) = sampling_metadata {
        metadata["sampling_params"] = sampling["sampling_params"].clone();
        metadata["forwarded_sampling_params"] = sampling["forwarded_sampling_params"].clone();
    }
    if let Some(trajectory) = trajectory {
        metadata["event_type"] = json!("assistant_message");
        metadata["event_role"] = json!(EventRole::ModelResponse.as_str());
        metadata["trajectory_id"] = json!(trajectory.trajectory_id);
        metadata["attempt_index"] = json!(trajectory.attempt_index);
    }
    metadata
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
