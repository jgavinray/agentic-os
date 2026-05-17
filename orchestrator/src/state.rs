use deadpool_postgres::Pool;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Minimum max_tokens for any upstream LiteLLM completion request.
/// vLLM backend exposes 262144 (256K) total context; 16K output leaves ~245K
/// for input and provides enough output budget for a complete LLM response.
pub const MIN_MAX_TOKENS: u64 = 16384;

/// Default TTL for cached context packs: 5 minutes.
pub const CONTEXT_CACHE_TTL_MS: u64 = 300_000;

/// Default exponential decay rate for search results.
///
/// The plan's acceptance criteria require a 1-hour-old result to score at most
/// 70% of an otherwise identical fresh result. A rate of 0.006 gives
/// e^(-0.006 * 60) ~= 0.698.
pub const DEFAULT_CONTEXT_DECAY_RATE: f64 = 0.006;

/// Cap source events consumed per summarization promotion pass.
pub const MAX_SUMMARIZER_EVENTS: i64 = 10;

/// Prompt templates for summary hierarchy promotion:
/// index 0 => raw events to L1, index 1 => L1 to L2, index 2 => L2 to L3.
pub const SUMMARY_PROMPTS: [&str; 3] = [
    "\
You are a precise technical summarizer. Extract information from conversation messages \
into these exact sections. Include only what is explicitly stated. \
Output nothing else — no preamble, no explanation.

DECISIONS:
(one decision per line, or the word \"none\")
OPEN_QUESTIONS:
(one question per line, or the word \"none\")
FAILED_APPROACHES:
(one failed approach per line, or the word \"none\")
KEY_FACTS:
(one key fact per line, or the word \"none\")

Messages:
{messages}",
    "\
You are consolidating event-level engineering summaries into a session-level summary. \
Keep recurring decisions, unresolved questions, failed approaches, and facts that would \
help resume the work later. Output only the consolidated summary.

Event summaries:
{messages}",
    "\
You are producing an executive engineering memory summary from session-level summaries. \
Keep durable architecture decisions, recurring risks, known failed approaches, and the \
current project state. Output only concise durable memory.

Session summaries:
{messages}",
];

/// Cached context pack with timestamp.
#[derive(Clone, Debug)]
pub struct CachedContext {
    pub context: String,
    pub memories: Vec<EventMemory>,
    pub cached_at: Instant,
}

/// Context cache keyed by repo:task:version.
#[derive(Clone)]
pub struct ContextCache {
    entries: Arc<std::sync::RwLock<HashMap<String, CachedContext>>>,
    ttl_ms: u64,
}

impl ContextCache {
    pub fn new(ttl_ms: u64) -> Self {
        Self {
            entries: Arc::new(std::sync::RwLock::new(HashMap::new())),
            ttl_ms,
        }
    }

    pub fn get(&self, key: &str) -> Option<CachedContext> {
        let entries = self.entries.read().unwrap();
        let entry = entries.get(key)?;
        if entry.cached_at.elapsed() < Duration::from_millis(self.ttl_ms) {
            Some(entry.clone())
        } else {
            None
        }
    }

    pub fn put(&self, key: String, value: CachedContext) {
        let mut entries = self.entries.write().unwrap();
        entries.insert(key, value);
    }

    /// Invalidate all entries matching a repo (wildcard: "repo:*").
    pub fn invalidate(&self, repo: &str, _task: &str) {
        let mut entries = self.entries.write().unwrap();
        entries.retain(|k, _| !k.starts_with(&format!("{repo}:")));
    }

    pub fn stats(&self) -> CacheStats {
        let entries = self.entries.read().unwrap();
        CacheStats {
            entries: entries.len(),
            ttl_ms: self.ttl_ms,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CacheStats {
    pub entries: usize,
    pub ttl_ms: u64,
}

pub fn context_cache_key(repo: &str, task: &str, event_count: i64) -> String {
    format!("{repo}:{task}:{event_count}")
}

// ── Task categories for retrieval depth ────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TaskCategory {
    Narrow,
    Moderate,
    Broad,
}

#[derive(Debug, Deserialize)]
pub struct TaskContextConfig {
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
        "migrate",
        "rewrite",
        "refactor",
        "redesign",
        "architecture",
        "deploy",
        "infrastructure",
        "setup",
        "configure",
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
}

/// Shared application state (cloned per request via Arc)
#[derive(Clone)]
pub struct AppState {
    /// Postgres connection pool
    pub pool: Pool,
    /// Qdrant base URL
    pub qdrant_url: String,
    /// LiteLLM base URL (must include /v1 suffix)
    pub litellm_url: String,
    /// LiteLLM API key
    pub litellm_key: String,
    /// Valid API keys — each entry is (token, namespace). Token authenticates; namespace
    /// partitions memory. Format in .env: `token,ns;token2,ns2`
    pub api_keys: Vec<(String, String)>,
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
    /// Context cache — keys by "repo:task:version"
    pub cache: ContextCache,
    /// Exponential decay rate used by hybrid search scoring.
    pub context_decay_rate: f64,
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

#[derive(Debug, Serialize, Clone)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_cache_key_includes_repo_task_and_event_count() {
        assert_eq!(
            context_cache_key("repo", "task", 42),
            "repo:task:42".to_string()
        );
    }

    #[test]
    fn context_cache_key_separates_repos() {
        let a = context_cache_key("repo-a", "task", 1);
        let b = context_cache_key("repo-b", "task", 1);
        assert_ne!(a, b);
    }

    #[test]
    fn context_cache_invalidate_removes_repo_entries() {
        let cache = ContextCache::new(CONTEXT_CACHE_TTL_MS);
        cache.put(
            context_cache_key("repo-a", "task", 1),
            CachedContext {
                context: "a".to_string(),
                memories: vec![],
                cached_at: Instant::now(),
            },
        );
        cache.put(
            context_cache_key("repo-b", "task", 1),
            CachedContext {
                context: "b".to_string(),
                memories: vec![],
                cached_at: Instant::now(),
            },
        );

        cache.invalidate("repo-a", "task");

        assert!(cache.get(&context_cache_key("repo-a", "task", 1)).is_none());
        assert!(cache.get(&context_cache_key("repo-b", "task", 1)).is_some());
    }
}
