use serde::{Deserialize, Serialize};

use crate::state_context::EventMemory;

#[derive(Debug, Deserialize)]
pub struct StartSessionRequest {
    pub repo: String,
    pub task: String,
    pub actor: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AppendEventRequest {
    pub session_id: String,
    pub repo: String,
    pub actor: Option<String>,
    pub event_type: String,
    pub summary: String,
    pub evidence: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub correlation_id: Option<uuid::Uuid>,
    pub parent_event_id: Option<uuid::Uuid>,
    pub trajectory_id: Option<uuid::Uuid>,
    pub attempt_index: Option<i32>,
    pub event_role: Option<String>,
    pub task: Option<String>,
    pub error_type: Option<String>,
    pub error_description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct HarnessGuardrailRequest {
    pub event_type: Option<String>,
    pub summary: Option<String>,
    pub evidence: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct HarnessGuardrailResponse {
    pub action: String,
    pub reason: String,
    pub should_stop: bool,
    pub signals: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ContextPackRequest {
    pub repo: String,
    pub task: String,
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct CheckpointRequest {
    pub session_id: String,
    pub repo: String,
    pub summary: String,
    pub next_actions: Option<Vec<String>>,
    pub open_questions: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct StartSessionResponse {
    pub session_id: String,
}

#[derive(Debug, Serialize)]
pub struct ContextPackResponse {
    pub repo: String,
    pub task: String,
    pub context: String,
    pub memories: Vec<EventMemory>,
}

#[derive(Debug, Clone)]
pub struct ErrorRecord {
    #[allow(dead_code)]
    pub id: String,
    #[allow(dead_code)]
    pub repo: String,
    #[allow(dead_code)]
    pub task: String,
    pub error_type: String,
    pub description: String,
    #[allow(dead_code)]
    pub severity: String,
    pub frequency: i64,
    #[allow(dead_code)]
    pub last_seen: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub event_id: String,
    pub event_type: String,
    pub summary: String,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
}
