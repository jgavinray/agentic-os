use super::*;
use crate::state::{ErrorRecord, SearchHit};
use chrono::Utc;

fn mem(event_type: &str, summary: &str, evidence: Option<&str>) -> crate::state::EventMemory {
    crate::state::EventMemory {
        event_type: event_type.to_string(),
        summary: summary.to_string(),
        evidence: evidence.map(str::to_string),
        metadata: serde_json::json!({}),
        created_at: Utc::now(),
        summary_level: 0,
    }
}

fn mem_with_meta(
    event_type: &str,
    summary: &str,
    metadata: serde_json::Value,
) -> crate::state::EventMemory {
    crate::state::EventMemory {
        event_type: event_type.to_string(),
        summary: summary.to_string(),
        evidence: None,
        metadata,
        created_at: Utc::now(),
        summary_level: 0,
    }
}

fn hit(event_type: &str, summary: &str) -> SearchHit {
    SearchHit {
        event_id: uuid::Uuid::new_v4().to_string(),
        event_type: event_type.to_string(),
        summary: summary.to_string(),
        created_at: Some(Utc::now()),
    }
}

fn err_rec(error_type: &str, description: &str, frequency: i64) -> ErrorRecord {
    ErrorRecord {
        id: "eid".to_string(),
        repo: "r".to_string(),
        task: "t".to_string(),
        error_type: error_type.to_string(),
        description: description.to_string(),
        severity: "medium".to_string(),
        frequency,
        last_seen: Utc::now(),
    }
}

fn event(summary: &str) -> AgentEvent {
    AgentEvent {
        id: "tid".to_string(),
        session_id: "tsession".to_string(),
        repo: "trepo".to_string(),
        actor: "tactor".to_string(),
        event_type: "decision".to_string(),
        summary: summary.to_string(),
        evidence: None,
        metadata: serde_json::json!({}),
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: None,
        created_at: Utc::now(),
        summary_level: 0,
    }
}

fn event_at_level(event_type: &str, summary: &str, level: i32) -> AgentEvent {
    AgentEvent {
        id: uuid::Uuid::new_v4().to_string(),
        session_id: "session".to_string(),
        repo: "repo".to_string(),
        actor: "actor".to_string(),
        event_type: event_type.to_string(),
        summary: summary.to_string(),
        evidence: None,
        metadata: serde_json::json!({}),
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: None,
        created_at: Utc::now(),
        summary_level: level,
    }
}

fn failed_event(summary: &str, evidence: &str, outcome: &str) -> AgentEvent {
    AgentEvent {
        evidence: Some(evidence.to_string()),
        metadata: serde_json::json!({
            "source_event_ids": ["a", "b"],
            "outcome": outcome,
        }),
        ..event_at_level("failed_attempt", summary, 0)
    }
}

#[cfg(test)]
#[path = "db_context_rendering_tests.rs"]
mod context_rendering_tests;

#[cfg(test)]
#[path = "db_context_failure_tests.rs"]
mod context_failure_tests;

#[cfg(test)]
#[path = "db_context_policy_tests.rs"]
mod context_policy_tests;

#[cfg(test)]
#[path = "db_record_tests.rs"]
mod record_tests;
