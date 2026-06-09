use super::*;
use crate::db;
use chrono::Utc;
use serde_json::json;

#[path = "context_pack.rs"]
mod context_pack_tests;

#[path = "context_source_path.rs"]
mod source_path_tests;

fn test_event(event_type: &str, summary: &str, summary_level: i32) -> db::AgentEvent {
    db::AgentEvent {
        id: uuid::Uuid::new_v4().to_string(),
        session_id: "session".to_string(),
        repo: "repo".to_string(),
        actor: "actor".to_string(),
        event_type: event_type.to_string(),
        summary: summary.to_string(),
        evidence: None,
        metadata: json!({}),
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: None,
        created_at: Utc::now(),
        summary_level,
    }
}
