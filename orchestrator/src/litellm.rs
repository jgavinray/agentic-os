use deadpool_postgres::Pool;
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;
use uuid::Uuid;

pub use crate::litellm_cache::{
    context_pack_hash, context_prefix_hashes, exact_cache_decision, split_context_prefix_tail,
    sse_chunk_has_non_empty_data,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachePolicySnapshot {
    pub cache_backend: Option<String>,
    pub cache_policy_enabled: bool,
    pub cache_bypass_reason: Option<String>,
}

#[derive(Clone, Debug)]
pub struct RouteSelection {
    pub routed_model: String,
    pub selected_route: String,
    pub selection_reason: String,
    pub policy_version: String,
}

#[derive(Clone, Debug)]
pub struct LiteLlmCallAttempt {
    pub attempt_id: Uuid,
    pub request_event_id: Option<Uuid>,
    pub trajectory_id: Option<Uuid>,
    pub context_pack_id: Option<Uuid>,
    pub namespace: String,
    pub repo: String,
    pub task: String,
    pub endpoint: String,
    pub requested_model: String,
    pub routed_model: String,
    pub selected_route: Option<String>,
    pub selection_reason: Option<String>,
    pub policy_version: Option<String>,
    pub reasoning_policy: Option<String>,
    pub reasoning_policy_source: Option<String>,
    pub baseline_arm: Option<String>,
    pub cache_policy: CachePolicySnapshot,
    pub context_pack_hash: Option<String>,
    pub started_at: Instant,
    pub first_token_at: Option<Instant>,
    pub completed_at: Option<Instant>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalStatus {
    Success,
    HttpError,
    NetworkError,
    ParseError,
    StreamError,
    ClientDisconnect,
    Cancelled,
    InternalError,
}

impl TerminalStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::HttpError => "http_error",
            Self::NetworkError => "network_error",
            Self::ParseError => "parse_error",
            Self::StreamError => "stream_error",
            Self::ClientDisconnect => "client_disconnect",
            Self::Cancelled => "cancelled",
            Self::InternalError => "internal_error",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProviderCacheCounters {
    pub provider_cached_tokens: i64,
    pub provider_cache_created_tokens: i64,
    pub provider_cache_read_tokens: i64,
}

impl ProviderCacheCounters {
    pub fn from_value(value: &Value) -> Self {
        let usage = &value["usage"];
        Self {
            provider_cached_tokens: usage["prompt_tokens_details"]["cached_tokens"]
                .as_i64()
                .unwrap_or(0),
            provider_cache_created_tokens: usage["cache_creation_input_tokens"]
                .as_i64()
                .unwrap_or(0),
            provider_cache_read_tokens: usage["cache_read_input_tokens"].as_i64().unwrap_or(0),
        }
    }

    pub fn max_assign(&mut self, other: Self) {
        self.provider_cached_tokens = self
            .provider_cached_tokens
            .max(other.provider_cached_tokens);
        self.provider_cache_created_tokens = self
            .provider_cache_created_tokens
            .max(other.provider_cache_created_tokens);
        self.provider_cache_read_tokens = self
            .provider_cache_read_tokens
            .max(other.provider_cache_read_tokens);
    }
}

pub struct LiteLlmCallFinalizer {
    pool: Pool,
    attempt: LiteLlmCallAttempt,
    finalized: Arc<AtomicBool>,
}

impl LiteLlmCallFinalizer {
    pub async fn begin(pool: Pool, attempt: LiteLlmCallAttempt) -> Self {
        if let Err(e) = insert_litellm_call_ledger_start(&pool, &attempt).await {
            tracing::warn!(
                attempt_id = %attempt.attempt_id,
                "failed to pre-insert litellm call ledger row before dispatch: {e}"
            );
        }
        Self {
            pool,
            attempt,
            finalized: Arc::new(AtomicBool::new(false)),
        }
    }

    pub async fn finalize(
        &self,
        status: TerminalStatus,
        error_kind: Option<&str>,
        error_message: Option<&str>,
        counters: ProviderCacheCounters,
    ) {
        if self.finalized.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Err(e) = update_litellm_call_ledger_terminal(
            &self.pool,
            &self.attempt,
            status,
            error_kind,
            error_message,
            counters,
        )
        .await
        {
            tracing::warn!(
                attempt_id = %self.attempt.attempt_id,
                terminal_status = status.as_str(),
                "failed to write litellm call ledger row: {e}"
            );
        }
    }

    pub fn attempt_mut(&mut self) -> &mut LiteLlmCallAttempt {
        &mut self.attempt
    }

    pub fn attempt(&self) -> &LiteLlmCallAttempt {
        &self.attempt
    }
}

impl Clone for LiteLlmCallFinalizer {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            attempt: self.attempt.clone(),
            finalized: self.finalized.clone(),
        }
    }
}

impl Drop for LiteLlmCallFinalizer {
    fn drop(&mut self) {
        if self.finalized.swap(true, Ordering::SeqCst) {
            return;
        }
        let pool = self.pool.clone();
        let attempt = self.attempt.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Err(e) = update_litellm_call_ledger_terminal(
                    &pool,
                    &attempt,
                    TerminalStatus::ClientDisconnect,
                    Some("client_disconnect"),
                    None,
                    ProviderCacheCounters::default(),
                )
                .await
                {
                    tracing::warn!(attempt_id = %attempt.attempt_id, "failed to write dropped stream ledger row: {e}");
                }
            });
        }
    }
}

pub async fn insert_litellm_call_ledger(
    pool: &Pool,
    attempt: &LiteLlmCallAttempt,
    status: TerminalStatus,
    error_kind: Option<&str>,
    error_message: Option<&str>,
    counters: ProviderCacheCounters,
) -> Result<(), anyhow::Error> {
    insert_litellm_call_ledger_start(pool, attempt).await?;
    update_litellm_call_ledger_terminal(pool, attempt, status, error_kind, error_message, counters)
        .await
}

pub async fn insert_litellm_call_ledger_start(
    pool: &Pool,
    attempt: &LiteLlmCallAttempt,
) -> Result<(), anyhow::Error> {
    let conn = pool.get().await?;
    conn.execute(
        "INSERT INTO litellm_call_ledger
         (attempt_id, request_event_id, trajectory_id, context_pack_id, namespace, repo, task,
          endpoint, requested_model, routed_model, context_pack_hash, cache_backend,
          cache_policy_enabled, cache_bypass_reason, policy_version, selected_route,
          selection_reason, reasoning_policy, reasoning_policy_source, baseline_arm)
         VALUES
         ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
         ON CONFLICT (attempt_id) DO NOTHING",
        &[
            &attempt.attempt_id,
            &attempt.request_event_id,
            &attempt.trajectory_id,
            &attempt.context_pack_id,
            &attempt.namespace,
            &attempt.repo,
            &attempt.task,
            &attempt.endpoint,
            &attempt.requested_model,
            &attempt.routed_model,
            &attempt.context_pack_hash,
            &attempt.cache_policy.cache_backend,
            &attempt.cache_policy.cache_policy_enabled,
            &attempt.cache_policy.cache_bypass_reason,
            &attempt.policy_version,
            &attempt.selected_route,
            &attempt.selection_reason,
            &attempt.reasoning_policy,
            &attempt.reasoning_policy_source,
            &attempt.baseline_arm,
        ],
    )
    .await?;
    Ok(())
}

pub async fn update_litellm_call_ledger_terminal(
    pool: &Pool,
    attempt: &LiteLlmCallAttempt,
    status: TerminalStatus,
    error_kind: Option<&str>,
    error_message: Option<&str>,
    counters: ProviderCacheCounters,
) -> Result<(), anyhow::Error> {
    let conn = pool.get().await?;
    let completed_at = attempt.completed_at.unwrap_or_else(Instant::now);
    let first_token_ms = attempt
        .first_token_at
        .map(|first| first.duration_since(attempt.started_at).as_millis() as i64);
    let total_latency_ms = completed_at.duration_since(attempt.started_at).as_millis() as i64;
    let updated = conn
        .execute(
            "UPDATE litellm_call_ledger
             SET terminal_status = $2,
                 error_kind = $3,
                 error_message = $4,
                 first_token_ms = $5,
                 total_latency_ms = $6,
                 context_pack_hash = $7,
                 cache_backend = $8,
                 cache_policy_enabled = $9,
                 cache_bypass_reason = $10,
                 policy_version = $11,
                 selected_route = $12,
                 selection_reason = $13,
                 reasoning_policy = $14,
                 reasoning_policy_source = $15,
                 baseline_arm = $16,
                 provider_cached_tokens = $17,
                 provider_cache_created_tokens = $18,
                 provider_cache_read_tokens = $19
             WHERE attempt_id = $1",
            &[
                &attempt.attempt_id,
                &status.as_str(),
                &error_kind,
                &error_message,
                &first_token_ms,
                &total_latency_ms,
                &attempt.context_pack_hash,
                &attempt.cache_policy.cache_backend,
                &attempt.cache_policy.cache_policy_enabled,
                &attempt.cache_policy.cache_bypass_reason,
                &attempt.policy_version,
                &attempt.selected_route,
                &attempt.selection_reason,
                &attempt.reasoning_policy,
                &attempt.reasoning_policy_source,
                &attempt.baseline_arm,
                &counters.provider_cached_tokens,
                &counters.provider_cache_created_tokens,
                &counters.provider_cache_read_tokens,
            ],
        )
        .await?;
    if updated > 0 {
        return Ok(());
    }

    conn.execute(
        "INSERT INTO litellm_call_ledger
         (attempt_id, request_event_id, trajectory_id, context_pack_id, namespace, repo, task,
          endpoint, requested_model, routed_model, terminal_status, error_kind, error_message,
          first_token_ms, total_latency_ms, context_pack_hash, cache_backend,
          cache_policy_enabled, cache_bypass_reason, policy_version, selected_route,
          selection_reason, reasoning_policy, reasoning_policy_source, baseline_arm,
          provider_cached_tokens, provider_cache_created_tokens,
          provider_cache_read_tokens)
         VALUES
         ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
          $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28)
         ON CONFLICT (attempt_id) DO UPDATE
         SET terminal_status = EXCLUDED.terminal_status,
             error_kind = EXCLUDED.error_kind,
             error_message = EXCLUDED.error_message,
             first_token_ms = EXCLUDED.first_token_ms,
             total_latency_ms = EXCLUDED.total_latency_ms,
             baseline_arm = EXCLUDED.baseline_arm,
             provider_cached_tokens = EXCLUDED.provider_cached_tokens,
             provider_cache_created_tokens = EXCLUDED.provider_cache_created_tokens,
             provider_cache_read_tokens = EXCLUDED.provider_cache_read_tokens",
        &[
            &attempt.attempt_id,
            &attempt.request_event_id,
            &attempt.trajectory_id,
            &attempt.context_pack_id,
            &attempt.namespace,
            &attempt.repo,
            &attempt.task,
            &attempt.endpoint,
            &attempt.requested_model,
            &attempt.routed_model,
            &status.as_str(),
            &error_kind,
            &error_message,
            &first_token_ms,
            &total_latency_ms,
            &attempt.context_pack_hash,
            &attempt.cache_policy.cache_backend,
            &attempt.cache_policy.cache_policy_enabled,
            &attempt.cache_policy.cache_bypass_reason,
            &attempt.policy_version,
            &attempt.selected_route,
            &attempt.selection_reason,
            &attempt.reasoning_policy,
            &attempt.reasoning_policy_source,
            &attempt.baseline_arm,
            &counters.provider_cached_tokens,
            &counters.provider_cache_created_tokens,
            &counters.provider_cache_read_tokens,
        ],
    )
    .await?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn new_attempt(
    request_event_id: Option<Uuid>,
    trajectory_id: Option<Uuid>,
    context_pack_id: Option<Uuid>,
    namespace: String,
    repo: String,
    task: String,
    endpoint: &str,
    requested_model: String,
    route: &RouteSelection,
    cache_policy: CachePolicySnapshot,
    context_pack_hash: Option<String>,
    baseline_arm: Option<String>,
) -> LiteLlmCallAttempt {
    LiteLlmCallAttempt {
        attempt_id: Uuid::new_v4(),
        request_event_id,
        trajectory_id,
        context_pack_id,
        namespace,
        repo,
        task,
        endpoint: endpoint.to_string(),
        requested_model,
        routed_model: route.routed_model.clone(),
        selected_route: Some(route.selected_route.clone()),
        selection_reason: Some(route.selection_reason.clone()),
        policy_version: Some(route.policy_version.clone()),
        reasoning_policy: None,
        reasoning_policy_source: None,
        baseline_arm,
        cache_policy,
        context_pack_hash,
        started_at: Instant::now(),
        first_token_at: None,
        completed_at: None,
    }
}

pub fn route_for_namespace(
    default_model: &str,
    namespace: &str,
    canary_enabled: bool,
    allowlist: &std::collections::HashSet<String>,
) -> RouteSelection {
    if canary_enabled && default_model == "agentic/strong" && allowlist.contains(namespace) {
        return RouteSelection {
            routed_model: "agentic/strong-prefix-canary".to_string(),
            selected_route: "prefix_cache_canary".to_string(),
            selection_reason: "namespace_allowlist".to_string(),
            policy_version: "prefix-cache-canary-v1".to_string(),
        };
    }
    RouteSelection {
        routed_model: default_model.to_string(),
        selected_route: "default".to_string(),
        selection_reason: "default_model".to_string(),
        policy_version: "default-routing-v1".to_string(),
    }
}

pub fn add_agentic_os_metadata(req: &mut Value, attempt: &LiteLlmCallAttempt) {
    if !req.get("metadata").is_some_and(Value::is_object) {
        req["metadata"] = json!({});
    }
    req["metadata"]["agentic_os"] = json!({
        "attempt_id": attempt.attempt_id,
        "request_event_id": attempt.request_event_id,
        "trajectory_id": attempt.trajectory_id,
        "context_pack_id": attempt.context_pack_id,
        "namespace": attempt.namespace,
        "repo": attempt.repo,
        "task": attempt.task,
        "endpoint": attempt.endpoint,
        "requested_model": attempt.requested_model,
        "routed_model": attempt.routed_model,
        "selected_route": attempt.selected_route,
        "selection_reason": attempt.selection_reason,
        "policy_version": attempt.policy_version,
        "reasoning_policy": attempt.reasoning_policy,
        "reasoning_policy_source": attempt.reasoning_policy_source,
        "baseline_arm": attempt.baseline_arm,
        "context_pack_hash": attempt.context_pack_hash,
        "cache_backend": attempt.cache_policy.cache_backend,
        "cache_policy_enabled": attempt.cache_policy.cache_policy_enabled,
        "cache_bypass_reason": attempt.cache_policy.cache_bypass_reason,
    });
}

pub fn provider_counters_from_sse(raw: &str) -> ProviderCacheCounters {
    let mut counters = ProviderCacheCounters::default();
    for line in raw.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data.trim() == "[DONE]" {
            break;
        }
        if let Ok(value) = serde_json::from_str::<Value>(data) {
            counters.max_assign(ProviderCacheCounters::from_value(&value));
            let usage = if value["type"].as_str() == Some("message_start") {
                &value["message"]["usage"]
            } else if value["type"].as_str() == Some("message_delta") {
                &value["usage"]
            } else {
                &Value::Null
            };
            counters.max_assign(ProviderCacheCounters {
                provider_cached_tokens: usage["prompt_tokens_details"]["cached_tokens"]
                    .as_i64()
                    .unwrap_or(0),
                provider_cache_created_tokens: usage["cache_creation_input_tokens"]
                    .as_i64()
                    .unwrap_or(0),
                provider_cache_read_tokens: usage["cache_read_input_tokens"].as_i64().unwrap_or(0),
            });
        }
    }
    counters
}

#[cfg(test)]
#[path = "litellm_tests.rs"]
mod tests;
