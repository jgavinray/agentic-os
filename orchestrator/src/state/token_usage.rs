use serde::Serialize;
use serde_json::Value;

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
