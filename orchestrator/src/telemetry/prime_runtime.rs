use metrics::{counter, gauge, histogram};

pub(crate) fn prime_request_entrypoint_metrics() {
    for result in ["accepted", "rejected"] {
        counter!("auth_attempts_total", "result" => result).increment(0);
        counter!("memory_promotions_total", "result" => result).increment(0);
    }
    for endpoint in [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/models",
        "/sessions/start",
        "/events/append",
        "/harness/guardrail",
        "/tools/authorize",
        "/v1/validations",
        "/context/pack",
        "/search",
        "/metrics",
    ] {
        gauge!("http_requests_in_flight", "endpoint" => endpoint).set(0.0);
    }
}

pub(crate) fn prime_upstream_metrics() {
    for path in ["chat_completions", "messages", "models"] {
        counter!("upstream_litellm_requests_total", "path" => path, "status" => "none")
            .increment(0);
        histogram!("upstream_litellm_duration_seconds", "path" => path).record(0.0);
        for kind in ["timeout", "connection", "4xx", "5xx", "parse"] {
            counter!("upstream_litellm_errors_total", "path" => path, "kind" => kind).increment(0);
        }
    }
    for path in ["chat_completions"] {
        counter!("summarizer_upstream_requests_total", "path" => path, "status" => "none")
            .increment(0);
        histogram!("summarizer_upstream_duration_seconds", "path" => path).record(0.0);
        for kind in ["timeout", "connection", "4xx", "5xx", "parse"] {
            counter!("summarizer_upstream_errors_total", "path" => path, "kind" => kind)
                .increment(0);
        }
    }
    for path in ["chat_completions", "messages"] {
        histogram!("stream_first_token_seconds", "path" => path).record(0.0);
        histogram!("stream_duration_seconds", "path" => path).record(0.0);
        counter!("stream_disconnects_total", "path" => path, "reason" => "client_disconnect")
            .increment(0);
    }
}

pub(crate) fn prime_dependency_metrics(sentiment_loaded: bool) {
    for op in [
        "create_session",
        "find_or_create_session",
        "insert_event",
        "count_events_for_repo",
        "get_context_evidence",
        "search_events_fts",
        "hydrate_active_search_hits",
        "get_event_chain",
        "get_trajectory",
        "get_trajectory_attempts",
        "get_trajectory_result",
        "latest_trajectory_event_for_session",
        "idle_trajectory_ids",
        "insert_trajectory_result_event",
        "get_failure_history",
        "insert_error_record",
        "get_active_errors",
        "record_token_usage",
        "check_ready",
    ] {
        histogram!("db_query_duration_seconds", "op" => op).record(0.0);
        counter!("db_query_errors_total", "op" => op).increment(0);
    }
    for op in ["upsert", "search", "health", "create_collection"] {
        counter!("qdrant_requests_total", "op" => op, "status" => "none").increment(0);
        histogram!("qdrant_request_duration_seconds", "op" => op).record(0.0);
    }
    counter!("embedder_inferences_total").increment(0);
    histogram!("embedder_inference_duration_seconds").record(0.0);
    histogram!("embedder_input_tokens").record(0.0);
    if sentiment_loaded {
        for verdict in ["negative", "non_negative"] {
            counter!("sentiment_inferences_total", "verdict" => verdict).increment(0);
        }
        histogram!("sentiment_inference_duration_seconds").record(0.0);
    }
}
