use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use subtle::ConstantTimeEq;

use crate::rate_limit;
use crate::state::AppState;
use crate::telemetry;

fn bearer_token(headers: &HeaderMap) -> &str {
    headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .unwrap_or("")
}

pub(crate) fn provided_api_token(headers: &HeaderMap) -> &str {
    let bearer = bearer_token(headers);
    if !bearer.is_empty() {
        return bearer;
    }
    headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
}

// Returns the (token, namespace) pair if auth passes; None otherwise.
// Constant-time comparison prevents timing-based key recovery.
pub(crate) fn authenticate(state: &AppState, headers: &HeaderMap) -> Option<(String, String)> {
    let provided = provided_api_token(headers).as_bytes();
    for (token, namespace) in &state.api_keys {
        let expected = token.as_bytes();
        if expected.len() == provided.len() && expected.ct_eq(provided).into() {
            telemetry::record_auth_attempt(true);
            return Some((token.clone(), namespace.clone()));
        }
    }
    telemetry::record_auth_attempt(false);
    None
}

pub(crate) fn check_auth(state: &AppState, headers: &HeaderMap) -> bool {
    authenticate(state, headers).is_some()
}

pub(crate) fn check_rate_limit(state: &AppState, token: &str) -> Option<Response> {
    match state.rate_limiter.check(token) {
        Ok(()) => None,
        Err(retry_after) => Some(rate_limited_response(token, retry_after)),
    }
}

pub(crate) fn rate_limited_response(token: &str, retry_after: u64) -> Response {
    let key_hash = rate_limit::key_hash(token);
    telemetry::record_rate_limited(&key_hash);
    (
        StatusCode::TOO_MANY_REQUESTS,
        [(header::RETRY_AFTER, retry_after.to_string())],
        axum::Json(serde_json::json!({
            "error": "rate_limited",
            "retry_after": retry_after
        })),
    )
        .into_response()
}
