use super::*;

#[test]
fn exact_cache_bypass_reasons_are_explicit() {
    assert_eq!(
        exact_cache_decision("chat_completions", &json!({"stream": true}), false)
            .cache_bypass_reason
            .as_deref(),
        Some("streaming")
    );
    assert_eq!(
        exact_cache_decision("chat_completions", &json!({"tools": []}), false)
            .cache_bypass_reason
            .as_deref(),
        Some("tools_present")
    );
    assert_eq!(
        exact_cache_decision("chat_completions", &json!({"temperature": 0}), true)
            .cache_bypass_reason
            .as_deref(),
        Some("disabled_by_config")
    );
}

#[test]
fn context_pack_hash_is_stable_and_content_sensitive() {
    let a = context_pack_hash("alpha\nbeta");
    let b = context_pack_hash("alpha\nbeta");
    let c = context_pack_hash("alpha\nbeta!");
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn context_prefix_hashes_split_stable_artifacts_from_dynamic_tail() {
    let context =
        "== Stable Context Artifacts ==\n[repo:service_topology:active]\nmodel\n\nRepository: r\nTask: t\n";
    let (stable, dynamic) = context_prefix_hashes(context);
    assert!(stable.is_some());
    assert!(dynamic.is_some());
    assert_ne!(stable, dynamic);
}

#[test]
fn canary_routing_requires_enabled_allowlisted_strong() {
    let mut allow = std::collections::HashSet::new();
    allow.insert("ns1".to_string());
    assert_eq!(
        route_for_namespace("agentic/strong", "ns1", true, &allow).routed_model,
        "agentic/strong-prefix-canary"
    );
    assert_eq!(
        route_for_namespace("agentic/strong", "ns2", true, &allow).selected_route,
        "default"
    );
    assert_eq!(
        route_for_namespace("agentic/small", "ns1", true, &allow).routed_model,
        "agentic/small"
    );
    assert_eq!(
        route_for_namespace("agentic/summarizer", "ns1", true, &allow).routed_model,
        "agentic/summarizer"
    );
}

#[test]
fn streaming_first_token_detection_ignores_done_only() {
    assert!(sse_chunk_has_non_empty_data(
        br#"data: {"choices":[{"delta":{"content":"x"}}]}"#
    ));
    assert!(!sse_chunk_has_non_empty_data(b"data: [DONE]\n\n"));
}

#[test]
fn terminal_status_values_match_ledger_contract() {
    assert_eq!(TerminalStatus::Success.as_str(), "success");
    assert_eq!(TerminalStatus::NetworkError.as_str(), "network_error");
    assert_eq!(TerminalStatus::ParseError.as_str(), "parse_error");
    assert_eq!(TerminalStatus::StreamError.as_str(), "stream_error");
    assert_eq!(
        TerminalStatus::ClientDisconnect.as_str(),
        "client_disconnect"
    );
}

#[test]
fn provider_cache_counters_parse_openai_and_anthropic_usage() {
    let value = json!({
        "usage": {
            "prompt_tokens_details": {"cached_tokens": 11},
            "cache_creation_input_tokens": 7,
            "cache_read_input_tokens": 5
        }
    });
    assert_eq!(
        ProviderCacheCounters::from_value(&value),
        ProviderCacheCounters {
            provider_cached_tokens: 11,
            provider_cache_created_tokens: 7,
            provider_cache_read_tokens: 5,
        }
    );

    let sse = r#"data: {"type":"message_start","message":{"usage":{"cache_creation_input_tokens":3,"cache_read_input_tokens":2}}}"#;
    assert_eq!(
        provider_counters_from_sse(sse),
        ProviderCacheCounters {
            provider_cached_tokens: 0,
            provider_cache_created_tokens: 3,
            provider_cache_read_tokens: 2,
        }
    );
}
