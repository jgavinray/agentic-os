use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::Value;

use crate::state::AppState;
use crate::telemetry;

pub(crate) struct OpenAiStreamingUpstream {
    pub response: reqwest::Response,
    pub finalizer: crate::litellm::LiteLlmCallFinalizer,
    pub started: std::time::Instant,
}

pub(crate) async fn dispatch_openai_streaming(
    state: &AppState,
    req: Value,
    mut finalizer: crate::litellm::LiteLlmCallFinalizer,
) -> Result<OpenAiStreamingUpstream, Response> {
    let url = format!("{}/chat/completions", state.litellm_url);
    let started = finalizer.attempt_mut().started_at;
    let upstream = match state
        .http_stream
        .post(&url)
        .bearer_auth(&state.litellm_key)
        .json(&req)
        .send()
        .await
    {
        Ok(r) => {
            let status = r.status();
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
            r
        }
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
            return Err((
                StatusCode::BAD_GATEWAY,
                axum::Json(
                    serde_json::json!({"error": "litellm_unreachable", "detail": e.to_string()}),
                ),
            )
                .into_response());
        }
    };

    if !upstream.status().is_success() {
        let status = upstream.status();
        telemetry::record_upstream_litellm_error(
            "chat_completions",
            telemetry::upstream_error_kind(status),
        );
        let body = upstream.text().await.unwrap_or_default();
        finalizer
            .finalize(
                crate::litellm::TerminalStatus::HttpError,
                Some(telemetry::upstream_error_kind(status)),
                Some("upstream returned non-success status"),
                crate::litellm::ProviderCacheCounters::default(),
            )
            .await;
        return Err((status, [(header::CONTENT_TYPE, "application/json")], body).into_response());
    }

    Ok(OpenAiStreamingUpstream {
        response: upstream,
        finalizer,
        started,
    })
}
