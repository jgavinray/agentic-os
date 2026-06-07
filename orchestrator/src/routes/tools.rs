use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use std::sync::Arc;

use crate::auth::authenticate;
use crate::orchestration_policy;
use crate::routes::tools_events::persist_tool_authorization_event;
use crate::state::AppState;
use crate::telemetry;

#[tracing::instrument(name = "handler.authorize_tool", skip(state, headers, req), fields(tool = %req.tool_name))]
pub async fn authorize_tool(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<crate::tool_mediation::ToolAuthorizeRequest>,
) -> Response {
    let Some((_caller_token, namespace)) = authenticate(&state, &headers) else {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    };

    // `/tools/authorize` is the runtime enforcement path. A model request can
    // have its tool menu shaped before generation, but a client may still ask
    // to run a specific tool later. This endpoint re-derives policy from the
    // attempted tool call and decides whether that concrete call is allowed.
    let (classification, policy) =
        derive_tool_authorization_policy(&req, &namespace, state.capture_pool.is_some());
    let response = crate::tool_mediation::authorize_tool_call_with_policy(
        &req,
        state.tool_mediation_enabled,
        Some(&policy),
    );
    let policy_metadata = orchestration_policy::compact_policy_metadata(&classification, &policy);
    telemetry::record_tool_authorization(&state.metrics, &response);
    persist_tool_authorization_event(
        &state,
        &namespace,
        &req,
        &response,
        Some(classification),
        Some(policy),
        Some(policy_metadata),
    );
    axum::Json(response).into_response()
}

/// Build compact classification text from the tool-authorization request.
///
/// Tool authorization is the one place where the orchestrator sees a concrete
/// attempted tool call before the client executes it. We need enough text to let
/// request classification detect destructive shell commands, current-info
/// queries, paths, and similar operational signals. We do not want to serialize
/// the entire argument object into classifier input, because tool payloads may
/// contain arbitrary user-controlled text or sensitive values.
///
/// The bounded key list below is therefore part of the policy contract. Unknown
/// keys are ignored for classification even though the full argument payload may
/// still be preserved in the tool authorization event metadata.
pub(crate) fn tool_authorization_classification_text(
    req: &crate::tool_mediation::ToolAuthorizeRequest,
) -> String {
    let mut parts = Vec::new();

    if let Some(intent) = req
        .user_intent
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        parts.push(intent.to_string());
    }

    parts.push(format!("tool:{}", req.tool_name));

    let argument_keys = ["command", "cmd", "script", "query", "path", "file_path"];
    for key in argument_keys {
        if let Some(Value::String(s)) = req.arguments.get(key) {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                parts.push(format!("{}:{}", key, trimmed));
            }
        }
    }

    parts.join("\n")
}

pub(crate) fn derive_tool_authorization_policy(
    req: &crate::tool_mediation::ToolAuthorizeRequest,
    namespace: &str,
    raw_capture_enabled: bool,
) -> (
    crate::request_classification::RequestClassification,
    orchestration_policy::OrchestrationPolicy,
) {
    // Tool authorization derives policy from the attempted tool call, not just
    // the original user prompt. That matters for generic intents like "run a
    // command" where the destructive signal exists only inside the command
    // argument.
    let repo = req.repo.as_deref().unwrap_or(namespace);
    let session_id = req.session_id.as_deref().unwrap_or("unknown");
    let request_text = tool_authorization_classification_text(req);
    let classification = crate::request_classification::classify_request_text(
        repo,
        session_id,
        &request_text,
        None,
        "tool_authorization_request",
    );
    let policy = orchestration_policy::derive_orchestration_policy(
        &classification,
        &request_text,
        raw_capture_enabled,
    );
    (classification, policy)
}
