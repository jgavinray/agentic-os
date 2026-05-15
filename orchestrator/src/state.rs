use deadpool_postgres::Pool;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::str::FromStr;

// ── Task categories for retrieval depth ────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TaskCategory {
    Narrow,
    Moderate,
    Broad,
}

#[derive(Debug, Deserialize)]
pub struct TaskContextConfig {
    pub task_category: TaskCategory,
    pub max_events: i64,
    pub semantic_limit: usize,
    pub char_budget: usize,
}

impl FromStr for TaskCategory {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "narrow" => Ok(TaskCategory::Narrow),
            "moderate" => Ok(TaskCategory::Moderate),
            "broad" => Ok(TaskCategory::Broad),
            _ => Err(format!("unknown task category: {s}")),
        }
    }
}

impl TaskContextConfig {
    pub fn for_category(cat: TaskCategory) -> Self {
        use TaskCategory::*;
        let (max_events, semantic_limit, char_budget) = match cat {
            Narrow => (3i64, 3usize, 5000usize),
            Moderate => (8i64, 5usize, 8000usize),
            Broad => (15i64, 10usize, 12000usize),
        };
        Self {
            task_category: cat,
            max_events,
            semantic_limit,
            char_budget,
        }
    }
}

impl TaskCategory {
    pub const NARROW_KEYWORDS: &'static [&'static str] = &[
        "fix", "bug", "debug", "error", "issue", "warn", "patch", "hotfix",
    ];
    pub const BROAD_KEYWORDS: &'static [&'static str] = &[
        "migrate", "rewrite", "refactor", "redesign", "architecture",
        "deploy", "infrastructure", "setup", "configure",
    ];

    pub fn from_task(task: &str) -> Self {
        let task_lower = task.to_lowercase();
        for kw in Self::BROAD_KEYWORDS {
            if task_lower.contains(kw) {
                return Self::Broad;
            }
        }
        for kw in Self::NARROW_KEYWORDS {
            if task_lower.contains(kw) {
                return Self::Narrow;
            }
        }
        Self::Moderate
    }

    pub fn max_events(&self) -> i64 {
        match self {
            Self::Narrow => 3,
            Self::Moderate => 8,
            Self::Broad => 15,
        }
    }
}

/// Shared application state (cloned per request via Arc)
#[derive(Clone)]
pub struct AppState {
    /// Postgres connection pool
    pub pool: Pool,
    /// Postgres connection string (kept for diagnostics)
    pub db_url: String,
    /// Qdrant base URL
    pub qdrant_url: String,
    /// LiteLLM base URL (must include /v1 suffix)
    pub litellm_url: String,
    /// LiteLLM API key
    pub litellm_key: String,
    /// Valid API keys — each key is a memory namespace (sk-work → namespace "work")
    pub api_keys: Vec<String>,
    /// Default model name for completions
    pub default_model: String,
    /// Default task when x-agent-task header is absent
    pub default_task: String,
    /// TEI embedding service base URL (e.g. http://embedding:80)
    pub embedding_url: String,
    /// Shared HTTP client for non-streaming upstream requests (has full timeouts)
    pub http: reqwest::Client,
    /// HTTP client for streaming upstream requests (no overall request timeout)
    pub http_stream: reqwest::Client,
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
    pub task: Option<String>,
    pub error_type: Option<String>,
    pub error_description: Option<String>,
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

#[derive(Debug, Clone)]
pub struct ErrorRecord {
    pub id: String,
    pub repo: String,
    pub task: String,
    pub error_type: String,
    pub description: String,
    pub severity: String,
    pub frequency: i64,
    pub last_seen: chrono::DateTime<chrono::Utc>,
}
