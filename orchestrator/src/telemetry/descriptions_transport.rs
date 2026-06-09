use metrics::{describe_counter, describe_gauge, describe_histogram};

pub(crate) fn describe_transport_metrics() {
    describe_counter!(
        "http_requests_total",
        "HTTP requests served by the orchestrator."
    );
    describe_histogram!(
        "http_request_duration_seconds",
        "HTTP request latency in seconds."
    );
    describe_gauge!(
        "http_requests_in_flight",
        "HTTP requests currently in flight."
    );
    describe_counter!("auth_attempts_total", "API authentication attempts.");
    describe_counter!(
        "upstream_litellm_requests_total",
        "Requests sent from the orchestrator to LiteLLM."
    );
    describe_histogram!(
        "upstream_litellm_duration_seconds",
        "LiteLLM upstream request latency in seconds."
    );
    describe_counter!(
        "upstream_litellm_errors_total",
        "LiteLLM upstream request errors by path and kind."
    );
    describe_counter!(
        "summarizer_upstream_requests_total",
        "Requests sent from the orchestrator to the summarizer endpoint."
    );
    describe_histogram!(
        "summarizer_upstream_duration_seconds",
        "Summarizer upstream request latency in seconds."
    );
    describe_counter!(
        "summarizer_upstream_errors_total",
        "Summarizer upstream request errors by path and kind."
    );
    describe_histogram!(
        "stream_first_token_seconds",
        "Seconds from request handling to first streamed byte."
    );
    describe_histogram!(
        "stream_duration_seconds",
        "Total stream duration in seconds."
    );
    describe_counter!(
        "stream_disconnects_total",
        "Streams interrupted before completion."
    );
}
