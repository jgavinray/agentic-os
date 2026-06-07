use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::Value;

use crate::state::AppState;
use crate::telemetry;

pub(crate) fn merge_request_metadata(
    parts: impl IntoIterator<Item = Option<Value>>,
) -> Option<Value> {
    let mut merged = serde_json::Map::new();
    for part in parts.into_iter().flatten() {
        if let Some(object) = part.as_object() {
            for (key, value) in object {
                merged.insert(key.clone(), value.clone());
            }
        }
    }
    if merged.is_empty() {
        None
    } else {
        Some(Value::Object(merged))
    }
}

pub(crate) fn baseline_arm_selection(
    headers: &HeaderMap,
) -> Result<crate::adversarial_harness::BaselineArm, String> {
    crate::adversarial_harness::baseline_arm_from_header(
        headers
            .get(crate::adversarial_harness::BASELINE_ARM_HEADER)
            .and_then(|value| value.to_str().ok()),
    )
}

pub(crate) fn litellm_route(state: &AppState, namespace: &str) -> crate::litellm::RouteSelection {
    crate::litellm::route_for_namespace(
        &state.default_model,
        namespace,
        state.prefix_cache_canary_enabled,
        &state.prefix_cache_canary_namespace_allowlist,
    )
}

pub(crate) fn extract_user_content_openai(payload: &Value) -> String {
    payload["messages"]
        .as_array()
        .and_then(|msgs| msgs.iter().rfind(|m| m["role"].as_str() == Some("user")))
        .map(|m| {
            if let Some(text) = m["content"].as_str() {
                text.to_string()
            } else {
                m["content"].to_string()
            }
        })
        .unwrap_or_default()
        .chars()
        .take(500)
        .collect()
}

pub(crate) async fn dispatch_non_streaming_raw(
    state: &AppState,
    openai_req: &Value,
    finalizer: &crate::litellm::LiteLlmCallFinalizer,
) -> Result<(Value, u64), Response> {
    let url = format!("{}/chat/completions", state.litellm_url);
    let started = std::time::Instant::now();
    let response = match state
        .http
        .post(&url)
        .bearer_auth(&state.litellm_key)
        .json(openai_req)
        .send()
        .await
    {
        Ok(r) => Ok(r),
        Err(e) => {
            telemetry::record_upstream_litellm("chat_completions", started.elapsed(), "error");
            telemetry::record_upstream_litellm_error(
                "chat_completions",
                telemetry::reqwest_error_kind(&e),
            );
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::NetworkError,
                    Some(telemetry::reqwest_error_kind(&e)),
                    Some(&e.to_string()),
                    crate::litellm::ProviderCacheCounters::default(),
                )
                .await;
            Err((
                StatusCode::BAD_GATEWAY,
                axum::Json(
                    serde_json::json!({"error": "litellm_unreachable", "detail": e.to_string()}),
                ),
            )
                .into_response())
        }
    }?;

    let status = response.status();
    telemetry::record_upstream_litellm(
        "chat_completions",
        started.elapsed(),
        &status.as_u16().to_string(),
    );
    if !status.is_success() {
        telemetry::record_upstream_litellm_error(
            "chat_completions",
            telemetry::upstream_error_kind(status),
        );
    }
    let latency_ms = started.elapsed().as_millis() as u64;
    let value = match response.json::<Value>().await {
        Ok(value) => value,
        Err(e) => {
            telemetry::record_upstream_litellm_error("chat_completions", "parse");
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::ParseError,
                    Some("parse"),
                    Some(&e.to_string()),
                    crate::litellm::ProviderCacheCounters::default(),
                )
                .await;
            return Err((
                StatusCode::BAD_GATEWAY,
                axum::Json(serde_json::json!({"error": "litellm_parse_error"})),
            )
                .into_response());
        }
    };
    if !status.is_success() {
        finalizer
            .finalize(
                crate::litellm::TerminalStatus::HttpError,
                Some(telemetry::upstream_error_kind(status)),
                Some("upstream returned non-success status"),
                crate::litellm::ProviderCacheCounters::from_value(&value),
            )
            .await;
        return Err((status, axum::Json(value)).into_response());
    }
    finalizer
        .finalize(
            crate::litellm::TerminalStatus::Success,
            None,
            None,
            crate::litellm::ProviderCacheCounters::from_value(&value),
        )
        .await;
    Ok((value, latency_ms))
}
