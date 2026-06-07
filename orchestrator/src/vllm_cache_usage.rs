use serde_json::Value;

pub(crate) fn merge_provider_cache_from_delta(
    provider_cache: crate::litellm::ProviderCacheCounters,
    delta: Option<crate::vllm_metrics::VllmCacheDelta>,
) -> crate::litellm::ProviderCacheCounters {
    let Some(delta) = delta else {
        return provider_cache;
    };
    let mut merged = provider_cache;
    merged.max_assign(provider_cache_from_delta(delta));
    merged
}

pub(crate) fn inject_anthropic_cache_usage(
    value: &mut Value,
    counters: crate::litellm::ProviderCacheCounters,
) {
    if let Some(usage) = value.get_mut("usage").and_then(Value::as_object_mut) {
        usage.insert(
            "cache_creation_input_tokens".to_string(),
            serde_json::json!(counters.provider_cache_created_tokens.max(0)),
        );
        usage.insert(
            "cache_read_input_tokens".to_string(),
            serde_json::json!(counters.provider_cache_read_tokens.max(0)),
        );
    }
}

pub(crate) fn anthropic_cache_usage_sse_event(
    counters: crate::litellm::ProviderCacheCounters,
) -> String {
    format!(
        concat!(
            "event: message_delta\n",
            "data: {{\"type\":\"message_delta\",\"delta\":{{}},\"usage\":{{",
            "\"cache_creation_input_tokens\":{},",
            "\"cache_read_input_tokens\":{}",
            "}}}}\n\n"
        ),
        counters.provider_cache_created_tokens.max(0),
        counters.provider_cache_read_tokens.max(0)
    )
}

fn provider_cache_from_delta(
    delta: crate::vllm_metrics::VllmCacheDelta,
) -> crate::litellm::ProviderCacheCounters {
    let cache_read = delta
        .prompt_tokens_local_cache_hit_delta
        .saturating_add(delta.prompt_tokens_external_kv_delta)
        .max(delta.prompt_tokens_cached_delta);
    crate::litellm::ProviderCacheCounters {
        provider_cached_tokens: delta.prompt_tokens_cached_delta.max(0),
        provider_cache_created_tokens: 0,
        provider_cache_read_tokens: cache_read.max(0),
    }
}
