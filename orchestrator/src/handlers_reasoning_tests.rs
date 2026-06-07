use super::*;
use axum::http::HeaderMap;

#[test]
fn local_reasoning_policy_prefers_header_over_client_fields() {
    let mut headers = HeaderMap::new();
    headers.insert("x-agent-reasoning-policy", "high".parse().unwrap());
    let payload = serde_json::json!({
        "reasoning_effort": "low",
        "thinking": {"type": "enabled", "budget_tokens": 1024}
    });

    let selection = local_reasoning_selection(&headers, &payload);

    assert_eq!(selection.policy, LocalReasoningPolicy::High);
    assert_eq!(selection.source, "x-agent-reasoning-policy");
}

#[test]
fn local_reasoning_policy_maps_anthropic_thinking_budget() {
    let headers = HeaderMap::new();

    let low = local_reasoning_selection(
        &headers,
        &serde_json::json!({"thinking": {"budget_tokens": 1024}}),
    );
    let medium = local_reasoning_selection(
        &headers,
        &serde_json::json!({"thinking": {"budget_tokens": 4096}}),
    );
    let high = local_reasoning_selection(
        &headers,
        &serde_json::json!({"thinking": {"budget_tokens": 12000}}),
    );

    assert_eq!(low.policy, LocalReasoningPolicy::Low);
    assert_eq!(medium.policy, LocalReasoningPolicy::Medium);
    assert_eq!(high.policy, LocalReasoningPolicy::High);
    assert_eq!(high.source, "client_thinking_budget");
}

#[test]
fn local_reasoning_defaults_preserve_explicit_sampling() {
    let mut req = serde_json::json!({
        "max_tokens": 333,
        "temperature": 0.7
    });

    apply_local_reasoning_defaults(
        &mut req,
        LocalReasoningSelection {
            policy: LocalReasoningPolicy::Low,
            source: "test",
        },
    );

    assert_eq!(req["max_tokens"], serde_json::json!(333));
    assert_eq!(req["temperature"], serde_json::json!(0.7));
}

#[test]
fn local_reasoning_defaults_shape_missing_sampling() {
    let mut req = serde_json::json!({});

    apply_local_reasoning_defaults(
        &mut req,
        LocalReasoningSelection {
            policy: LocalReasoningPolicy::Low,
            source: "test",
        },
    );

    assert_eq!(req["max_tokens"], serde_json::json!(2048));
    assert_eq!(req["temperature"], serde_json::json!(0.1));
}
