use deadpool_postgres::Pool;
use std::collections::HashSet;
use std::sync::Arc;

pub mod api_types;
pub mod context;
pub mod context_cache;
pub mod context_policy;
pub mod limits;
pub mod summarizer;
pub mod task_category;
pub mod token_usage;

pub use api_types::{
    AppendEventRequest, CheckpointRequest, ContextPackRequest, ContextPackResponse, ErrorRecord,
    HarnessGuardrailRequest, HarnessGuardrailResponse, SearchHit, StartSessionRequest,
    StartSessionResponse,
};
pub use context::{
    ContextPolicy, EventMemory, MemoryLevel, MemoryScope, PromotionReason, TaskCategory,
    TaskContextConfig, CONTEXT_CACHE_TTL_MS, DEFAULT_CONTEXT_DECAY_RATE,
    DEFAULT_FAILURE_HISTORY_TOKEN_BUDGET, DEFAULT_OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET,
};
pub use context_cache::{
    context_cache_key, CacheStats, CachedContext, ContextCache, ContextPackStats,
};
pub use limits::{
    configured_default_max_tokens, configured_max_max_tokens, DEFAULT_BACKGROUND_WORK_CONCURRENCY,
    DEFAULT_MAX_TOKENS, MAX_MAX_TOKENS,
};
pub use summarizer::{MAX_SUMMARIZER_EVENTS, SUMMARIZER_MAX_TOKENS, SUMMARY_PROMPTS};
pub use token_usage::TokenUsage;

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
    /// Enables classification-driven model routing (recommended_route → model).
    pub classification_routing_enabled: bool,
    /// Model used for RecommendedRoute::SmallLocalModel when routing is on.
    pub route_model_small: Option<String>,
    /// Model used for RecommendedRoute::StrongLocalModel when routing is on.
    pub route_model_strong: Option<String>,
    /// Prometheus scrape handle.
    pub prometheus: metrics_exporter_prometheus::PrometheusHandle,
    /// JSON compatibility snapshot for legacy metrics callers.
    pub metrics: crate::telemetry::MetricsRegistry,
}

#[cfg(test)]
mod tests;
