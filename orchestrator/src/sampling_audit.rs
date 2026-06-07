use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::sampling::{
    PARAM_MAX_TOKENS, PARAM_SEED, PARAM_TEMPERATURE, PARAM_TOP_P, REQUEST_EVENT_SCHEMA_VERSION,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: Value,
    pub top_p: Value,
    pub max_tokens: Value,
    pub seed: Value,
}

impl SamplingParams {
    pub fn from_request(request: &Value) -> Self {
        Self {
            temperature: request
                .get(PARAM_TEMPERATURE)
                .cloned()
                .unwrap_or(Value::Null),
            top_p: request.get(PARAM_TOP_P).cloned().unwrap_or(Value::Null),
            max_tokens: request
                .get(PARAM_MAX_TOKENS)
                .cloned()
                .unwrap_or(Value::Null),
            seed: request.get(PARAM_SEED).cloned().unwrap_or(Value::Null),
        }
    }

    pub(crate) fn get(&self, parameter: &str) -> &Value {
        match parameter {
            PARAM_TEMPERATURE => &self.temperature,
            PARAM_TOP_P => &self.top_p,
            PARAM_MAX_TOKENS => &self.max_tokens,
            PARAM_SEED => &self.seed,
            _ => &Value::Null,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingAudit {
    pub sampling_params: SamplingParams,
    pub forwarded_sampling_params: SamplingParams,
}

impl SamplingAudit {
    pub fn metadata(&self, endpoint: &str, requested_model: &str, routed_model: &str) -> Value {
        json!({
            "event_schema_version": REQUEST_EVENT_SCHEMA_VERSION,
            "endpoint": endpoint,
            "requested_model": requested_model,
            "routed_model": routed_model,
            "sampling_params": self.sampling_params,
            "forwarded_sampling_params": self.forwarded_sampling_params,
        })
    }
}

pub fn capture_without_override(requested: SamplingParams) -> SamplingAudit {
    SamplingAudit {
        sampling_params: requested.clone(),
        forwarded_sampling_params: requested,
    }
}

pub fn capture_with_override(original_request: &Value, forwarded_request: &Value) -> SamplingAudit {
    SamplingAudit {
        sampling_params: SamplingParams::from_request(original_request),
        forwarded_sampling_params: SamplingParams::from_request(forwarded_request),
    }
}
