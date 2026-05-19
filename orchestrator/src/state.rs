use deadpool_postgres::Pool;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Minimum max_tokens for proxied completion requests.
pub const MIN_MAX_TOKENS: u64 = 8192;

/// max_tokens used by the internal summarizer — summaries are short.
pub const SUMMARIZER_MAX_TOKENS: u64 = 2048;

/// Default TTL for cached context packs: 5 minutes.
pub const CONTEXT_CACHE_TTL_MS: u64 = 300_000;

/// Default exponential decay rate for search results.
///
/// The plan's acceptance criteria require a 1-hour-old result to score at most
/// 70% of an otherwise identical fresh result. A rate of 0.006 gives
/// e^(-0.006 * 60) ~= 0.698.
pub const DEFAULT_CONTEXT_DECAY_RATE: f64 = 0.006;

/// Default token budget for deterministic Failure History context.
pub const DEFAULT_FAILURE_HISTORY_TOKEN_BUDGET: usize = 1000;

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

#[derive(Debug, Clone, Serialize, Default)]
pub struct ContextPackStats {
    pub build_ms: u64,
    pub context_chars: usize,
    pub context_tokens_estimate: usize,
    pub l0_items_injected: usize,
    pub l1_items_injected: usize,
    pub l2_items_injected: usize,
    pub l3_items_injected: usize,
    pub failed_attempts_injected: usize,
    pub remediations_injected: usize,
    pub failure_history_items_injected: usize,
    #[serde(skip_serializing)]
    pub failure_history_remediation_signatures: Vec<String>,
    pub retrieval_semantic_hits: usize,
    pub retrieval_fts_hits: usize,
    pub retrieval_deduped_hits: usize,
    pub retrieved_event_ids: Vec<String>,
    pub memory_levels_used: Vec<String>,
    pub injected_failure_signatures: Vec<String>,
    pub token_budget: usize,
    pub truncated: bool,
    pub cache_hit: bool,
}

/// Cached context pack with timestamp.
#[derive(Clone, Debug)]
pub struct CachedContext {
    pub context: String,
    pub memories: Vec<EventMemory>,
    pub cached_at: Instant,
    pub stats: ContextPackStats,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskCategory {
    Narrow,
    Moderate,
    Broad,
    Architecture,
}

#[derive(Debug, Deserialize, Clone)]
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
            "architecture" => Ok(TaskCategory::Architecture),
            _ => Err(format!("unknown task category: {s}")),
        }
    }
}

impl TaskContextConfig {
    pub fn for_category(cat: TaskCategory) -> Self {
        use TaskCategory::*;
        let (max_events, semantic_limit, char_budget) = match cat {
            Narrow => (3i64, 3usize, 3000usize),
            Moderate => (8i64, 5usize, 5000usize),
            Broad => (15i64, 10usize, 7000usize),
            Architecture => (20i64, 12usize, 10000usize),
        };
        Self {
            max_events,
            semantic_limit,
            char_budget,
        }
    }
}

impl TaskCategory {
    pub const ARCHITECTURE_KEYWORDS: &'static [&'static str] = &[
        "architecture",
        "architectural",
        "design",
        "invariant",
        "durable",
        "project truth",
        "system design",
    ];
    pub const NARROW_KEYWORDS: &'static [&'static str] = &[
        "fix", "bug", "debug", "error", "issue", "warn", "patch", "hotfix",
    ];
    pub const BROAD_KEYWORDS: &'static [&'static str] = &[
        "migrate",
        "rewrite",
        "refactor",
        "redesign",
        "deploy",
        "infrastructure",
        "setup",
        "configure",
    ];

    pub fn from_task(task: &str) -> Self {
        let task_lower = task.to_lowercase();
        for kw in Self::ARCHITECTURE_KEYWORDS {
            if task_lower.contains(kw) {
                return Self::Architecture;
            }
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryLevel {
    L0 = 0,
    L1 = 1,
    L2 = 2,
    L3 = 3,
}

impl MemoryLevel {
    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum MemoryScope {
    Session,
    Repo,
    Project,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum PromotionReason {
    AutomaticSessionSummary,
    RepeatedPattern,
    ReferencedAgain,
    MarkedImportant,
    FailedAttemptRemediated,
    StableOverTime,
    ManualConfirmation,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ContextPolicy {
    pub task_category: TaskCategory,
    pub l0_recent_pct: u8,
    pub l1_matching_pct: u8,
    pub l2_repo_pct: u8,
    pub l3_project_pct: u8,
    pub failure_pct: u8,
    pub l0_recent_limit: i64,
    pub l1_limit: i64,
    pub l2_limit: i64,
    pub l3_limit: i64,
    pub failure_limit: i64,
}

impl ContextPolicy {
    pub fn for_task(task: &str) -> Self {
        Self::for_category(TaskCategory::from_task(task))
    }

    pub fn for_category(task_category: TaskCategory) -> Self {
        match task_category {
            TaskCategory::Narrow => Self {
                task_category,
                l0_recent_pct: 40,
                l1_matching_pct: 30,
                l2_repo_pct: 10,
                l3_project_pct: 0,
                failure_pct: 20,
                l0_recent_limit: 5,
                l1_limit: 4,
                l2_limit: 2,
                l3_limit: 0,
                failure_limit: 4,
            },
            TaskCategory::Moderate => Self {
                task_category,
                l0_recent_pct: 25,
                l1_matching_pct: 35,
                l2_repo_pct: 20,
                l3_project_pct: 5,
                failure_pct: 15,
                l0_recent_limit: 5,
                l1_limit: 6,
                l2_limit: 4,
                l3_limit: 2,
                failure_limit: 3,
            },
            TaskCategory::Broad => Self {
                task_category,
                l0_recent_pct: 10,
                l1_matching_pct: 35,
                l2_repo_pct: 35,
                l3_project_pct: 10,
                failure_pct: 10,
                l0_recent_limit: 3,
                l1_limit: 8,
                l2_limit: 8,
                l3_limit: 3,
                failure_limit: 3,
            },
            TaskCategory::Architecture => Self {
                task_category,
                l0_recent_pct: 5,
                l1_matching_pct: 20,
                l2_repo_pct: 35,
                l3_project_pct: 30,
                failure_pct: 10,
                l0_recent_limit: 2,
                l1_limit: 5,
                l2_limit: 8,
                l3_limit: 6,
                failure_limit: 3,
            },
        }
    }

    pub fn budget_for(&self, pct: u8, char_budget: usize) -> usize {
        (char_budget * pct as usize) / 100
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct TokenUsage {
    pub processed_tokens: u64,
    pub cached_tokens: u64,
    pub generated_tokens: u64,
}

impl TokenUsage {
    pub fn from_openai_value(value: &Value) -> Self {
        let usage = &value["usage"];
        let processed_tokens = usage["prompt_tokens"]
            .as_u64()
            .or_else(|| usage["input_tokens"].as_u64())
            .unwrap_or(0);
        let generated_tokens = usage["completion_tokens"]
            .as_u64()
            .or_else(|| usage["output_tokens"].as_u64())
            .unwrap_or(0);
        let cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .or_else(|| usage["input_token_details"]["cache_read"].as_u64())
            .or_else(|| usage["cache_read_input_tokens"].as_u64())
            .unwrap_or(0);

        Self {
            processed_tokens,
            cached_tokens,
            generated_tokens,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.processed_tokens == 0 && self.cached_tokens == 0 && self.generated_tokens == 0
    }
}

/// Shared application state (cloned per request via Arc)
#[derive(Clone)]
pub struct AppState {
    /// Postgres connection pool
    pub pool: Pool,
    /// Optional CPU sentiment classifier for detecting negative user feedback.
    pub sentiment: Option<Arc<crate::sentiment::SentimentClassifier>>,
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
    /// In-process ONNX embedding model (snowflake-arctic-embed-l-v2.0).
    pub embedder: Arc<crate::embedder::Embedder>,
    /// Shared HTTP client for non-streaming upstream requests (has full timeouts)
    pub http: reqwest::Client,
    /// HTTP client for streaming upstream requests (no overall request timeout)
    pub http_stream: reqwest::Client,
    /// Context cache — keys by "repo:task:version"
    pub cache: ContextCache,
    /// Exponential decay rate used by hybrid search scoring.
    pub context_decay_rate: f64,
    /// Per-API-key limiter for expensive inference routes.
    pub rate_limiter: crate::rate_limit::RateLimiter,
    /// Enables deterministic execution feedback capture and Failure History context.
    pub execution_feedback_enabled: bool,
    /// Enables deterministic trajectory lineage metadata and result emission.
    pub trajectory_capture_enabled: bool,
    /// Idle timeout after which an open trajectory is finalized as unresolved.
    pub trajectory_idle_timeout_sec: u64,
    /// Token budget for the Failure History context section.
    pub failure_history_token_budget: usize,
    /// Controls sampling parameter audit capture and override hook invocation.
    pub sampling_config: crate::sampling::SamplingConfig,
    /// Request-level sampling policy hook. Defaults to a no-op implementation.
    pub sampling_policy: Arc<dyn crate::sampling::SamplingPolicy>,
    /// Prometheus scrape handle.
    pub prometheus: metrics_exporter_prometheus::PrometheusHandle,
    /// JSON compatibility snapshot for legacy metrics callers.
    pub metrics: crate::telemetry::MetricsRegistry,
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
    pub summary_level: i32,
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
                stats: ContextPackStats::default(),
            },
        );
        cache.put(
            context_cache_key("repo-b", "task", 1),
            CachedContext {
                context: "b".to_string(),
                memories: vec![],
                cached_at: Instant::now(),
                stats: ContextPackStats::default(),
            },
        );

        cache.invalidate("repo-a", "task");

        assert!(cache.get(&context_cache_key("repo-a", "task", 1)).is_none());
        assert!(cache.get(&context_cache_key("repo-b", "task", 1)).is_some());
    }

    #[test]
    fn task_category_detects_architecture_before_broad() {
        assert_eq!(
            TaskCategory::from_task("architecture refactor for memory"),
            TaskCategory::Architecture
        );
    }

    #[test]
    fn context_policy_small_task_includes_l0_and_l1() {
        let policy = ContextPolicy::for_category(TaskCategory::Narrow);
        assert!(policy.l0_recent_pct > 0);
        assert!(policy.l1_matching_pct > 0);
        assert!(policy.failure_pct > 0);
    }

    #[test]
    fn context_policy_architecture_task_prioritizes_l2_l3() {
        let policy = ContextPolicy::for_category(TaskCategory::Architecture);
        assert!(policy.l2_repo_pct + policy.l3_project_pct > policy.l0_recent_pct);
        assert!(policy.l3_limit > 0);
    }

    #[test]
    fn token_usage_extracts_processed_cached_and_generated_tokens() {
        let usage = TokenUsage::from_openai_value(&serde_json::json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "prompt_tokens_details": {"cached_tokens": 40}
            }
        }));
        assert_eq!(usage.processed_tokens, 100);
        assert_eq!(usage.cached_tokens, 40);
        assert_eq!(usage.generated_tokens, 25);
    }
}
