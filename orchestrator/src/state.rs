use deadpool_postgres::Pool;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

pub use crate::state_context::{
    ContextPolicy, EventMemory, MemoryLevel, MemoryScope, PromotionReason, TaskCategory,
    TaskContextConfig, CONTEXT_CACHE_TTL_MS, DEFAULT_CONTEXT_DECAY_RATE,
    DEFAULT_FAILURE_HISTORY_TOKEN_BUDGET, DEFAULT_OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET,
};
pub use crate::state_context_cache::{
    context_cache_key, CacheStats, CachedContext, ContextCache, ContextPackStats,
};
pub use crate::state_limits::{
    configured_default_max_tokens, configured_max_max_tokens, DEFAULT_BACKGROUND_WORK_CONCURRENCY,
    DEFAULT_MAX_TOKENS, MAX_MAX_TOKENS,
};
pub use crate::state_summarizer::{MAX_SUMMARIZER_EVENTS, SUMMARIZER_MAX_TOKENS, SUMMARY_PROMPTS};
pub use crate::state_token_usage::TokenUsage;

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
