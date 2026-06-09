use crate::state::{AppState, TokenUsage};

pub mod cache_usage;
pub mod metrics_snapshot;

pub(crate) use cache_usage::{
    anthropic_cache_usage_sse_event, inject_anthropic_cache_usage, merge_provider_cache_from_delta,
};
pub use metrics_snapshot::{
    fetch_cache_snapshot, parse_cache_snapshot, VllmCacheDelta, VllmCacheSnapshot,
};

pub(crate) async fn cache_snapshot(
    state: &AppState,
) -> Option<(String, crate::vllm::VllmCacheSnapshot)> {
    let metrics_url = state.vllm_metrics_url.as_deref()?;
    match fetch_cache_snapshot(&state.http, metrics_url).await {
        Ok(snapshot) => Some((metrics_url.to_string(), snapshot)),
        Err(e) => {
            tracing::warn!(metrics_url, "failed to fetch vLLM cache metrics: {e}");
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn record_cache_observation(
    state: &AppState,
    before: Option<(String, crate::vllm::VllmCacheSnapshot)>,
    session_id: Option<&str>,
    namespace: &str,
    repo: &str,
    task: &str,
    attempt: &crate::litellm::LiteLlmCallAttempt,
    usage: &TokenUsage,
    provider_cache: crate::litellm::ProviderCacheCounters,
) -> Option<crate::vllm::VllmCacheDelta> {
    let Some((metrics_url, before)) = before else {
        return None;
    };
    let after = match fetch_cache_snapshot(&state.http, &metrics_url).await {
        Ok(snapshot) => snapshot,
        Err(e) => {
            tracing::warn!(
                metrics_url,
                "failed to fetch post-request vLLM cache metrics: {e}"
            );
            return None;
        }
    };
    let delta = after.delta_since(before);
    crate::telemetry::record_vllm_cache_delta(&delta, &attempt.routed_model);
    let observation = crate::db::VllmCacheObservationInput {
        session_id: session_id.map(str::to_string),
        namespace: namespace.to_string(),
        repo: repo.to_string(),
        task: task.to_string(),
        endpoint: attempt.endpoint.clone(),
        requested_model: attempt.requested_model.clone(),
        routed_model: attempt.routed_model.clone(),
        request_event_id: attempt.request_event_id,
        context_pack_id: attempt.context_pack_id,
        attempt_id: attempt.attempt_id,
        metrics_url,
        delta,
        request_input_tokens: usage.processed_tokens as i64,
        request_output_tokens: usage.generated_tokens as i64,
        provider_cache,
    };
    if let Err(e) = crate::db::insert_vllm_cache_observation(&state.pool, &observation).await {
        tracing::warn!(
            attempt_id = %attempt.attempt_id,
            "failed to record vLLM cache observation: {e}"
        );
    }
    Some(delta)
}
