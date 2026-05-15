use deadpool_postgres::Pool;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Shared application state (cloned per request via Arc)
#[derive(Clone)]
pub struct AppState {
    /// Postgres connection pool
    pub pool: Pool,
    /// Postgres connection string
    pub db_url: String,
    /// Qdrant base URL
    pub qdrant_url: String,
    /// LiteLLM base URL
    pub litellm_url: String,
    /// LiteLLM API key
    pub litellm_key: String,
    /// Orchestrator API key (for auth enforcement)
    pub api_key: String,
    /// Default model name
    pub default_model: String,
    /// Shared HTTP client for upstream requests
    pub http: reqwest::Client,
}

// ── Request types ──────────────────────────────────────────────

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
    pub metadata: Option<Value>,
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

// ── Response types ─────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct StartSessionResponse {
    pub session_id: String,
}

#[derive(Debug, Serialize)]
pub struct EventMemory {
    pub event_type: String,
    pub summary: String,
    pub evidence: Option<String>,
    pub metadata: Value,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub struct ContextPackResponse {
    pub repo: String,
    pub task: String,
    pub context: String,
    pub memories: Vec<EventMemory>,
}
