use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use uuid::Uuid;

use crate::anthropic;
use crate::handlers_anthropic_stream_upstream::{
    finalize_stream_network_error, send_streaming_messages_request,
};
use crate::litellm::{LiteLlmCallFinalizer, ProviderCacheCounters, RouteSelection, TerminalStatus};
use crate::state::AppState;
use crate::telemetry;
use crate::token_limits::{context_window_retry_max_tokens, set_max_tokens};
use crate::trajectory::TrajectoryContext;

pub(crate) struct AnthropicStreamingUpstream {
    pub response: reqwest::Response,
    pub finalizer: LiteLlmCallFinalizer,
}

#[derive(Clone, Copy)]
struct AnthropicRetryAttemptContext<'a> {
    request_event_id: Option<Uuid>,
    trajectory: Option<TrajectoryContext>,
    context_pack_id: Option<Uuid>,
    namespace: &'a str,
    repo: &'a str,
    task: &'a str,
    model: &'a str,
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn dispatch_anthropic_streaming_with_retry(
    state: &AppState,
    mut req: Value,
    mut finalizer: LiteLlmCallFinalizer,
    request_event_id: Option<Uuid>,
    trajectory: Option<TrajectoryContext>,
    context_pack_id: Option<Uuid>,
    namespace: &str,
    repo: &str,
    task: &str,
    model: &str,
) -> Result<AnthropicStreamingUpstream, Response> {
    let url = format!("{}/messages", state.litellm_url);
    let retry_context = AnthropicRetryAttemptContext {
        request_event_id,
        trajectory,
        context_pack_id,
        namespace,
        repo,
        task,
        model,
    };
    let started = finalizer.attempt_mut().started_at;
    let mut upstream = match send_streaming_messages_request(state, &url, &req, started).await {
        Ok(r) => r,
        Err(e) => {
            finalize_stream_network_error(&mut finalizer, &e).await;
            return Err(anthropic::error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("upstream unreachable: {e}"),
            ));
        }
    };

    let upstream_status = upstream.status();
    if upstream_status.is_success() {
        return Ok(AnthropicStreamingUpstream {
            response: upstream,
            finalizer,
        });
    }

    let body = upstream.text().await.unwrap_or_else(|e| {
        serde_json::json!({
            "type": "error",
            "error": {
                "type": "api_error",
                "message": format!("failed to read upstream error body: {e}")
            }
        })
        .to_string()
    });
    if let Some(retry_max_tokens) = context_window_retry_max_tokens(&body) {
        finalizer
            .finalize(
                TerminalStatus::HttpError,
                Some(telemetry::upstream_error_kind(upstream_status)),
                Some("upstream returned non-success status"),
                ProviderCacheCounters::default(),
            )
            .await;
        tracing::warn!(
            retry_max_tokens,
            upstream_status = upstream_status.as_u16(),
            "retrying streaming messages request with reduced max_tokens after context window error"
        );
        set_max_tokens(&mut req, retry_max_tokens);
        finalizer = begin_retry_attempt(state, &mut req, &finalizer, retry_context).await;
        let retry_started = std::time::Instant::now();
        upstream = match send_streaming_messages_request(state, &url, &req, retry_started).await {
            Ok(r) => r,
            Err(e) => {
                finalize_stream_network_error(&mut finalizer, &e).await;
                return Err(anthropic::error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    format!("upstream retry unreachable: {e}"),
                ));
            }
        };
        if upstream.status().is_success() {
            return Ok(AnthropicStreamingUpstream {
                response: upstream,
                finalizer,
            });
        }
        let retry_status = upstream.status();
        let retry_body = upstream.text().await.unwrap_or_else(|e| {
            serde_json::json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": format!("failed to read upstream retry error body: {e}")
                }
            })
            .to_string()
        });
        finalizer
            .finalize(
                TerminalStatus::HttpError,
                Some(telemetry::upstream_error_kind(retry_status)),
                Some("upstream returned non-success status"),
                ProviderCacheCounters::default(),
            )
            .await;
        return Err((
            retry_status,
            [(header::CONTENT_TYPE, "application/json")],
            retry_body,
        )
            .into_response());
    }

    finalizer
        .finalize(
            TerminalStatus::HttpError,
            Some(telemetry::upstream_error_kind(upstream_status)),
            Some("upstream returned non-success status"),
            ProviderCacheCounters::default(),
        )
        .await;
    Err((
        upstream_status,
        [(header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response())
}

async fn begin_retry_attempt(
    state: &AppState,
    req: &mut Value,
    finalizer: &LiteLlmCallFinalizer,
    retry_context: AnthropicRetryAttemptContext<'_>,
) -> LiteLlmCallFinalizer {
    let prior_attempt = finalizer.attempt().clone();
    let route = RouteSelection {
        routed_model: prior_attempt.routed_model.clone(),
        selected_route: prior_attempt
            .selected_route
            .clone()
            .unwrap_or_else(|| "default".to_string()),
        selection_reason: prior_attempt
            .selection_reason
            .clone()
            .unwrap_or_else(|| "default_model".to_string()),
        policy_version: prior_attempt
            .policy_version
            .clone()
            .unwrap_or_else(|| "default-routing-v1".to_string()),
    };
    let retry_attempt = crate::litellm::new_attempt(
        retry_context.request_event_id,
        retry_context
            .trajectory
            .map(|trajectory| trajectory.trajectory_id),
        retry_context.context_pack_id,
        retry_context.namespace.to_string(),
        retry_context.repo.to_string(),
        retry_context.task.to_string(),
        "messages",
        retry_context.model.to_string(),
        &route,
        crate::litellm::exact_cache_decision("messages", req, false),
        prior_attempt.context_pack_hash.clone(),
        prior_attempt.baseline_arm.clone(),
    );
    crate::litellm::add_agentic_os_metadata(req, &retry_attempt);
    LiteLlmCallFinalizer::begin(state.pool.clone(), retry_attempt).await
}
