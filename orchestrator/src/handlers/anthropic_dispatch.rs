use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use std::sync::Arc;
use uuid::Uuid;

use crate::anthropic;
use crate::handlers_anthropic_json_upstream::{finalize_network_error, send_json_messages_request};
use crate::litellm::{LiteLlmCallFinalizer, ProviderCacheCounters, RouteSelection, TerminalStatus};
use crate::state::AppState;
use crate::telemetry;
use crate::token_limits::{context_window_retry_max_tokens, set_max_tokens};
use crate::trajectory::TrajectoryContext;

pub(crate) struct AnthropicMessagesDispatch {
    pub value: Value,
    pub finalizer: LiteLlmCallFinalizer,
    pub latency_ms: i64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn dispatch_anthropic_messages_with_retry(
    state: &Arc<AppState>,
    req: &mut Value,
    mut finalizer: LiteLlmCallFinalizer,
    request_event_id: Option<Uuid>,
    trajectory: Option<TrajectoryContext>,
    context_pack_id: Option<Uuid>,
    namespace: &str,
    repo: &str,
    task: &str,
    model: &str,
    route: &RouteSelection,
    context_pack_hash: Option<String>,
    baseline_arm: Option<String>,
) -> Result<AnthropicMessagesDispatch, Response> {
    let url = format!("{}/messages", state.litellm_url);
    let started = std::time::Instant::now();
    let upstream_resp = match send_json_messages_request(state, &url, req).await {
        Ok(r) => r,
        Err(e) => {
            finalize_network_error(&mut finalizer, &e).await;
            return Err(anthropic::error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                "upstream LiteLLM request failed",
            ));
        }
    };

    let mut status = upstream_resp.status();
    let mut val: Value = match upstream_resp.json().await {
        Ok(v) => v,
        Err(_) => {
            telemetry::record_upstream_litellm_error("messages", "parse");
            finalizer
                .finalize(
                    TerminalStatus::ParseError,
                    Some("parse"),
                    Some("invalid upstream response"),
                    ProviderCacheCounters::default(),
                )
                .await;
            return Err(anthropic::error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                "invalid upstream response",
            ));
        }
    };

    if !status.is_success() {
        let error_body = val.to_string();
        if let Some(retry_max_tokens) = context_window_retry_max_tokens(&error_body) {
            finalizer
                .finalize(
                    TerminalStatus::HttpError,
                    Some(telemetry::upstream_error_kind(status)),
                    Some("upstream returned non-success status"),
                    ProviderCacheCounters::from_value(&val),
                )
                .await;
            tracing::warn!(
                retry_max_tokens,
                upstream_status = status.as_u16(),
                "retrying messages request with reduced max_tokens after context window error"
            );
            set_max_tokens(req, retry_max_tokens);
            let retry_attempt = crate::litellm::new_attempt(
                request_event_id,
                trajectory.map(|trajectory| trajectory.trajectory_id),
                context_pack_id,
                namespace.to_string(),
                repo.to_string(),
                task.to_string(),
                "messages",
                model.to_string(),
                route,
                crate::litellm::exact_cache_decision("messages", req, false),
                context_pack_hash,
                baseline_arm,
            );
            crate::litellm::add_agentic_os_metadata(req, &retry_attempt);
            finalizer =
                crate::litellm::LiteLlmCallFinalizer::begin(state.pool.clone(), retry_attempt)
                    .await;
            let retry_resp = match send_json_messages_request(state, &url, req).await {
                Ok(r) => r,
                Err(e) => {
                    finalize_network_error(&mut finalizer, &e).await;
                    return Err(anthropic::error(
                        StatusCode::BAD_GATEWAY,
                        "api_error",
                        "upstream LiteLLM retry failed",
                    ));
                }
            };
            status = retry_resp.status();
            val = match retry_resp.json().await {
                Ok(v) => v,
                Err(_) => {
                    telemetry::record_upstream_litellm_error("messages", "parse");
                    finalizer
                        .finalize(
                            TerminalStatus::ParseError,
                            Some("parse"),
                            Some("invalid upstream retry response"),
                            ProviderCacheCounters::default(),
                        )
                        .await;
                    return Err(anthropic::error(
                        StatusCode::BAD_GATEWAY,
                        "api_error",
                        "invalid upstream retry response",
                    ));
                }
            };
        }
    }

    if !status.is_success() {
        finalizer
            .finalize(
                TerminalStatus::HttpError,
                Some(telemetry::upstream_error_kind(status)),
                Some("upstream returned non-success status"),
                ProviderCacheCounters::from_value(&val),
            )
            .await;
        return Err((status, axum::Json(val)).into_response());
    }
    finalizer
        .finalize(
            TerminalStatus::Success,
            None,
            None,
            ProviderCacheCounters::from_value(&val),
        )
        .await;

    Ok(AnthropicMessagesDispatch {
        value: val,
        finalizer,
        latency_ms: started.elapsed().as_millis() as i64,
    })
}
