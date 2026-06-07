use deadpool_postgres::Pool;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::sync::Arc;

pub use crate::state_context::{
    context_cache_key, CacheStats, CachedContext, ContextCache, ContextPackStats, ContextPolicy,
    EventMemory, MemoryLevel, MemoryScope, PromotionReason, TaskCategory, TaskContextConfig,
    CONTEXT_CACHE_TTL_MS, DEFAULT_CONTEXT_DECAY_RATE, DEFAULT_FAILURE_HISTORY_TOKEN_BUDGET,
    DEFAULT_OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET,
};

/// Default max_tokens for proxied completion requests when the client omits it.
pub const DEFAULT_MAX_TOKENS: u64 = 8192;

/// Maximum max_tokens this backend can safely serve for Claude Code traffic.
pub const MAX_MAX_TOKENS: u64 = 32768;

/// Default max_tokens used by the internal summarizer — summaries are short.
pub const SUMMARIZER_MAX_TOKENS: u64 = 384;

pub fn configured_default_max_tokens() -> u64 {
    std::env::var("DEFAULT_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .max(1)
}

pub fn configured_max_max_tokens() -> u64 {
    std::env::var("MAX_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(MAX_MAX_TOKENS)
        .max(1)
}

/// Default concurrency for best-effort derived background work.
pub const DEFAULT_BACKGROUND_WORK_CONCURRENCY: usize = 4;

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
    /// Optional isolated raw client capture database pool.
    pub capture_pool: Option<Pool>,
    /// Optional CPU sentiment classifier for detecting negative user feedback.
    pub sentiment: Option<Arc<crate::sentiment::SentimentClassifier>>,
    /// Qdrant base URL
    pub qdrant_url: String,
    /// LiteLLM base URL (must include /v1 suffix)
    pub litellm_url: String,
    /// LiteLLM API key
    pub litellm_key: String,
    /// Summarizer OpenAI-compatible base URL (must include /v1 suffix).
    pub summarizer_url: String,
    /// Optional Summarizer API key. Local llama.cpp deployments usually omit this.
    pub summarizer_key: Option<String>,
    /// Model name sent to the summarizer endpoint.
    pub summarizer_model: String,
    /// Optional Total Recall HTTP API base URL.
    pub total_recall_url: Option<String>,
    /// Optional vLLM Prometheus metrics URL for backend prefix-cache observations.
    pub vllm_metrics_url: Option<String>,
    /// Max output tokens for summary generation.
    pub summarizer_max_tokens: u64,
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
    /// Enables deterministic feature extraction and Operational Constraints context.
    pub feature_extraction_enabled: bool,
    /// Token budget for the Operational Constraints context section.
    pub operational_constraints_token_budget: usize,
    /// Bounded gate for best-effort derived background work.
    pub background_work: Arc<tokio::sync::Semaphore>,
    /// Controls sampling parameter audit capture and override hook invocation.
    pub sampling_config: crate::sampling::SamplingConfig,
    /// Request-level sampling policy hook. Defaults to a no-op implementation.
    pub sampling_policy: Arc<dyn crate::sampling::SamplingPolicy>,
    /// Feature-flagged request classification live policy. Disabled by default.
    pub request_live_policy_config: crate::request_classification::LivePolicyConfig,
    /// Enables allowlist-only routing to the prefix-cache canary backend.
    pub prefix_cache_canary_enabled: bool,
    /// Namespaces allowed to route agentic/strong calls to the prefix-cache canary.
    pub prefix_cache_canary_namespace_allowlist: HashSet<String>,
    /// Enables deterministic tool menu shaping and tool-call authorization.
    pub tool_mediation_enabled: bool,
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
pub struct HarnessGuardrailRequest {
    pub event_type: Option<String>,
    pub summary: Option<String>,
    pub evidence: Option<String>,
    pub metadata: Option<Value>,
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

// ── Response types ─────────────────────────────────────────────

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

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;
