use super::*;
use crate::db::AgentEvent;
use serde_json::{json, Value};

pub fn event(id: &str, summary: &str, evidence: Option<&str>) -> AgentEvent {
    AgentEvent {
        id: id.to_string(),
        session_id: "session-1".to_string(),
        repo: "agent-os".to_string(),
        actor: "user".to_string(),
        event_type: "user_message".to_string(),
        summary: summary.to_string(),
        evidence: evidence.map(str::to_string),
        metadata: json!({
            "payload": {
                "kind": "test"
            }
        }),
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: Some("request".to_string()),
        created_at: chrono::DateTime::parse_from_rfc3339("2026-05-23T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc),
        summary_level: 0,
    }
}

pub fn feature_string_values_are_bounded(value: &Value) -> bool {
    match value {
        Value::String(value) => enum_inventory()
            .iter()
            .any(|(_name, variants)| variants.contains(&value.as_str())),
        Value::Array(values) => values.iter().all(feature_string_values_are_bounded),
        Value::Object(values) => values.values().all(feature_string_values_are_bounded),
        _ => true,
    }
}
