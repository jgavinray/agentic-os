use serde_json::Value;

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
