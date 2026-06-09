use crate::handlers_request_preparation::{
    prepare_anthropic_litellm_request, prepare_openai_litellm_request,
};
use crate::local_reasoning::{LocalReasoningPolicy, LocalReasoningSelection};
use crate::state::{DEFAULT_MAX_TOKENS, MAX_MAX_TOKENS};
use crate::token_limits::enforce_min_max_tokens;
use serde_json::json;

#[test]
fn model_substituted_with_default_in_chat_completions() {
    let payload = json!({
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "hi"}]
    });
    let default_model = "qwen36-35b-heretic";
    let req = prepare_openai_litellm_request(
        &payload,
        default_model,
        LocalReasoningSelection {
            policy: LocalReasoningPolicy::Medium,
            source: "test",
        },
    );

    assert_eq!(req["model"].as_str().unwrap(), default_model);
    assert!(req["messages"][0]["content"]
        .as_str()
        .unwrap()
        .contains("Local harness reasoning policy: medium"));
}

#[test]
fn model_substituted_with_default_for_any_client_model_name() {
    let client_models = [
        "gpt-4o",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "gpt-4-turbo",
    ];
    let default_model = "qwen36-35b-heretic";
    for client_model in client_models {
        let payload = json!({
            "model": client_model,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = prepare_openai_litellm_request(
            &payload,
            default_model,
            LocalReasoningSelection {
                policy: LocalReasoningPolicy::Medium,
                source: "test",
            },
        );
        assert_eq!(
            req["model"].as_str().unwrap(),
            default_model,
            "client model '{client_model}' was not replaced"
        );
    }
}

#[test]
fn anthropic_sanitize_removes_local_backend_incompatible_params() {
    let payload = json!({
        "model": "qwen36-27b",
        "max_tokens": 1024,
        "max_output_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "reasoning_effort": "high",
        "context_management": {"edits": []},
        "messages": [{"role": "user", "content": "hi"}]
    });

    let req = prepare_anthropic_litellm_request(
        payload,
        "qwen36-35b-heretic",
        LocalReasoningSelection {
            policy: LocalReasoningPolicy::Medium,
            source: "test",
        },
    );

    assert!(req.get("thinking").is_none());
    assert!(req.get("max_output_tokens").is_none());
    assert!(req.get("reasoning_effort").is_none());
    assert!(req.get("context_management").is_none());
    assert_eq!(req["max_tokens"], json!(1024));
    assert_eq!(req["messages"][0]["content"], "hi");
    assert_eq!(req["model"], json!("qwen36-35b-heretic"));
    assert!(req["system"]
        .as_array()
        .unwrap()
        .iter()
        .any(|block| block["text"]
            .as_str()
            .unwrap_or_default()
            .contains("Local harness reasoning policy: medium")));
}

#[test]
fn max_tokens_clamp_respects_explicit_small_values() {
    let mut req = json!({
        "model": "qwen36-27b",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}]
    });

    enforce_min_max_tokens(&mut req);

    assert_eq!(req["max_tokens"], json!(1));
}

#[test]
fn max_tokens_clamp_defaults_missing_values() {
    let mut req = json!({
        "model": "qwen36-27b",
        "messages": [{"role": "user", "content": "hi"}]
    });

    enforce_min_max_tokens(&mut req);

    assert_eq!(req["max_tokens"], json!(DEFAULT_MAX_TOKENS));
}

#[test]
fn max_tokens_clamp_caps_oversized_values() {
    let mut req = json!({
        "model": "qwen36-27b",
        "max_tokens": MAX_MAX_TOKENS + 1,
        "messages": [{"role": "user", "content": "hi"}]
    });

    enforce_min_max_tokens(&mut req);

    assert_eq!(req["max_tokens"], json!(MAX_MAX_TOKENS));
}
