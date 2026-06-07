use serde_json::{json, Value};
use std::time::Instant;
use uuid::Uuid;

pub use crate::litellm_cache::{
    context_pack_hash, context_prefix_hashes, exact_cache_decision, split_context_prefix_tail,
    sse_chunk_has_non_empty_data,
};
pub use crate::litellm_ledger::{
    LiteLlmCallAttempt, LiteLlmCallFinalizer, ProviderCacheCounters, TerminalStatus,
};
pub use crate::litellm_ledger_persistence::{
    insert_litellm_call_ledger, insert_litellm_call_ledger_start,
    update_litellm_call_ledger_terminal,
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
