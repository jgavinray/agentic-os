use deadpool_postgres::Pool;
use serde_json::Value;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;
use uuid::Uuid;

use crate::litellm::CachePolicySnapshot;
use crate::litellm_ledger_persistence::{
    insert_litellm_call_ledger_start, update_litellm_call_ledger_terminal,
};

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
