use metrics::{counter, histogram};
use std::time::Duration;

pub fn record_rate_limited(key_hash: &str) {
    counter!("rate_limited_total", "key_hash" => key_hash.to_string()).increment(1);
}

pub fn record_sampling_param_override(parameter: &'static str, reason: &'static str) {
    counter!("sampling_param_overrides_total", "parameter" => parameter, "reason" => reason)
        .increment(1);
}

pub fn record_db_query(op: &'static str, elapsed: Duration, success: bool) {
    histogram!("db_query_duration_seconds", "op" => op).record(elapsed.as_secs_f64());
    if !success {
        counter!("db_query_errors_total", "op" => op).increment(1);
    }
}

pub fn record_qdrant_request(op: &'static str, elapsed: Duration, status: &str) {
    histogram!("qdrant_request_duration_seconds", "op" => op).record(elapsed.as_secs_f64());
    counter!("qdrant_requests_total", "op" => op, "status" => status.to_string()).increment(1);
}
