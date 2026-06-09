use serde_json::Value;

use crate::litellm::{LiteLlmCallFinalizer, ProviderCacheCounters, TerminalStatus};
use crate::state::AppState;
use crate::telemetry;

pub(crate) async fn send_json_messages_request(
    state: &AppState,
    url: &str,
    req: &Value,
) -> Result<reqwest::Response, reqwest::Error> {
    let started = std::time::Instant::now();
    match state
        .http
        .post(url)
        .bearer_auth(&state.litellm_key)
        .json(req)
        .send()
        .await
    {
        Ok(r) => {
            let status = r.status();
            telemetry::record_upstream_litellm(
                "messages",
                started.elapsed(),
                &status.as_u16().to_string(),
            );
            if !status.is_success() {
                telemetry::record_upstream_litellm_error(
                    "messages",
                    telemetry::upstream_error_kind(status),
                );
            }
            Ok(r)
        }
        Err(e) => {
            telemetry::record_upstream_litellm("messages", started.elapsed(), "error");
            telemetry::record_upstream_litellm_error("messages", telemetry::reqwest_error_kind(&e));
            Err(e)
        }
    }
}

pub(crate) async fn finalize_network_error(
    finalizer: &mut LiteLlmCallFinalizer,
    e: &reqwest::Error,
) {
    finalizer
        .finalize(
            TerminalStatus::NetworkError,
            Some(telemetry::reqwest_error_kind(e)),
            Some(&e.to_string()),
            ProviderCacheCounters::default(),
        )
        .await;
}
