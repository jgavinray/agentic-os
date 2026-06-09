use metrics::{counter, histogram};
use std::time::Duration;

pub fn record_upstream_litellm(path: &'static str, elapsed: Duration, status: &str) {
    histogram!("upstream_litellm_duration_seconds", "path" => path).record(elapsed.as_secs_f64());
    counter!(
        "upstream_litellm_requests_total",
        "path" => path,
        "status" => status.to_string()
    )
    .increment(1);
}

pub fn record_upstream_litellm_error(path: &'static str, kind: &'static str) {
    counter!("upstream_litellm_errors_total", "path" => path, "kind" => kind).increment(1);
}

pub fn record_upstream_summarizer(path: &'static str, elapsed: Duration, status: &str) {
    histogram!("summarizer_upstream_duration_seconds", "path" => path)
        .record(elapsed.as_secs_f64());
    counter!(
        "summarizer_upstream_requests_total",
        "path" => path,
        "status" => status.to_string()
    )
    .increment(1);
}

pub fn record_upstream_summarizer_error(path: &'static str, kind: &'static str) {
    counter!("summarizer_upstream_errors_total", "path" => path, "kind" => kind).increment(1);
}

pub fn upstream_error_kind(status: reqwest::StatusCode) -> &'static str {
    if status.is_client_error() {
        "4xx"
    } else if status.is_server_error() {
        "5xx"
    } else {
        "parse"
    }
}

pub fn reqwest_error_kind(err: &reqwest::Error) -> &'static str {
    if err.is_timeout() {
        "timeout"
    } else if err.is_connect() {
        "connection"
    } else if err.is_status() {
        "5xx"
    } else {
        "parse"
    }
}
