use serde_json::Value;

use crate::anthropic;
use crate::local_reasoning::{
    apply_local_reasoning_defaults, inject_contract_anthropic, inject_contract_openai,
    LocalReasoningSelection,
};
use crate::token_limits::enforce_min_max_tokens;

pub(crate) fn prepare_openai_litellm_request(
    payload: &Value,
    routed_model: &str,
    reasoning_selection: LocalReasoningSelection,
) -> Value {
    let mut req = payload.clone();
    req["model"] = Value::String(routed_model.to_string());
    apply_local_reasoning_defaults(&mut req, reasoning_selection);
    enforce_min_max_tokens(&mut req);
    inject_contract_openai(&mut req, reasoning_selection);
    req
}

pub(crate) fn prepare_anthropic_litellm_request(
    payload: Value,
    routed_model: &str,
    reasoning_selection: LocalReasoningSelection,
) -> Value {
    let mut req = payload;
    req["model"] = Value::String(routed_model.to_string());
    apply_local_reasoning_defaults(&mut req, reasoning_selection);
    enforce_min_max_tokens(&mut req);
    anthropic::normalize_response_content_types(&mut req);
    anthropic::sanitize_litellm_request(&mut req);
    inject_contract_anthropic(&mut req, reasoning_selection);
    req
}
