use super::*;
use chrono::{DateTime, Duration, TimeZone, Utc};
use serde_json::json;
use serde_json::Value;

fn event(id: &str, event_type: &str, metadata: Value, created_at: DateTime<Utc>) -> AgentEvent {
    AgentEvent {
        id: id.to_string(),
        session_id: "s1".to_string(),
        repo: "repo".to_string(),
        actor: "agent".to_string(),
        event_type: event_type.to_string(),
        summary: event_type.to_string(),
        evidence: None,
        metadata,
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: None,
        created_at,
        summary_level: 0,
    }
}

fn config(now: DateTime<Utc>) -> ExtractionConfig {
    ExtractionConfig {
        evaluation_time: now,
        feature_window_sec: 3600,
        constraint_freshness_window_sec: 1800,
        max_operational_constraints: 5,
    }
}

#[path = "feature_extraction_live_health_tests.rs"]
mod live_health_tests;

#[path = "feature_extraction_bootstrap_detection_tests.rs"]
mod bootstrap_detection_tests;
