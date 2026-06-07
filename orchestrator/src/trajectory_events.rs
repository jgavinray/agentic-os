use crate::db::AgentEvent;
use crate::trajectory_types::{
    EventRole, TrajectoryContext, EVENT_TYPE_CONTEXT_PACK, EVENT_TYPE_TRAJECTORY_RESULT,
};
use serde_json::{json, Value};
use uuid::Uuid;

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
    summary: crate::trajectory_summary::TrajectoryResultSummary,
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
