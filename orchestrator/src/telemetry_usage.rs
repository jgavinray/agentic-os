use crate::telemetry::MetricsRegistry;
use metrics::counter;

pub fn record_trajectory_result(summary: &crate::trajectory::TrajectoryResultSummary) {
    counter!(
        "trajectory_results_total",
        "status" => summary.final_status.as_str()
    )
    .increment(1);
    counter!("trajectory_attempts_total").increment(summary.final_attempt_index.max(1) as u64);
    counter!("trajectory_tokens_total", "direction" => "input")
        .increment(summary.total_input_tokens.max(0) as u64);
    counter!("trajectory_tokens_total", "direction" => "output")
        .increment(summary.total_output_tokens.max(0) as u64);
}

pub fn record_trajectory_feature_written(registry: &MetricsRegistry) {
    counter!("trajectory_features_total").increment(1);
    registry.with_snapshot_mut(|metrics| {
        metrics.trajectory_features += 1;
    });
}

pub fn record_tokens(registry: &MetricsRegistry, usage: &crate::state::TokenUsage, model: &str) {
    if usage.is_empty() {
        return;
    }
    counter!("inference_tokens_total", "kind" => "processed", "model" => model.to_string())
        .increment(usage.processed_tokens);
    counter!("inference_tokens_total", "kind" => "cached", "model" => model.to_string())
        .increment(usage.cached_tokens);
    counter!("inference_tokens_total", "kind" => "generated", "model" => model.to_string())
        .increment(usage.generated_tokens);

    registry.with_snapshot_mut(|metrics| {
        metrics.processed_tokens += usage.processed_tokens;
        metrics.cached_tokens += usage.cached_tokens;
        metrics.generated_tokens += usage.generated_tokens;
    });
}

pub fn record_vllm_cache_delta(delta: &crate::vllm_metrics::VllmCacheDelta, model: &str) {
    counter!(
        "vllm_prefix_cache_tokens_total",
        "kind" => "queries",
        "model" => model.to_string()
    )
    .increment(delta.prefix_cache_queries_delta.max(0) as u64);
    counter!(
        "vllm_prefix_cache_tokens_total",
        "kind" => "hits",
        "model" => model.to_string()
    )
    .increment(delta.prefix_cache_hits_delta.max(0) as u64);
    for (source, value) in [
        ("total", delta.prompt_tokens_total_delta),
        ("cached", delta.prompt_tokens_cached_delta),
        ("local_compute", delta.prompt_tokens_local_compute_delta),
        ("local_cache_hit", delta.prompt_tokens_local_cache_hit_delta),
        (
            "external_kv_transfer",
            delta.prompt_tokens_external_kv_delta,
        ),
    ] {
        counter!(
            "vllm_prompt_tokens_by_cache_source_total",
            "source" => source,
            "model" => model.to_string()
        )
        .increment(value.max(0) as u64);
    }
}

pub fn record_vllm_cache_token_flow(
    input_tokens: u64,
    output_tokens: u64,
    provider_cache: crate::litellm::ProviderCacheCounters,
    model: &str,
) {
    for (kind, value) in [
        ("request_input", input_tokens),
        ("request_output", output_tokens),
        (
            "provider_cached",
            provider_cache.provider_cached_tokens.max(0) as u64,
        ),
        (
            "provider_cache_created",
            provider_cache.provider_cache_created_tokens.max(0) as u64,
        ),
        (
            "provider_cache_read",
            provider_cache.provider_cache_read_tokens.max(0) as u64,
        ),
    ] {
        counter!(
            "vllm_prompt_tokens_by_cache_source_total",
            "source" => kind,
            "model" => model.to_string()
        )
        .increment(value);
    }
}
