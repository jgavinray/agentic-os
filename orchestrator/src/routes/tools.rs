use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use std::sync::Arc;

use crate::auth::authenticate;
use crate::background::spawn_bounded_background;
use crate::event_capture::spawn_feature_extraction;
use crate::orchestration_policy;
use crate::state::{AppState, AppendEventRequest};
use crate::{db, telemetry};

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

fn persist_tool_authorization_event(
    state: &AppState,
    namespace: &str,
    req: &crate::tool_mediation::ToolAuthorizeRequest,
    response: &crate::tool_mediation::ToolAuthorizeResponse,
    policy_classification: Option<crate::request_classification::RequestClassification>,
    policy: Option<orchestration_policy::OrchestrationPolicy>,
    policy_metadata: Option<Value>,
) {
    if !state.tool_mediation_enabled {
        return;
    }
    // Tool authorization can be called without a session when a client only
    // wants an immediate allow/deny answer. Without a session, there is nowhere
    // durable to attach the decision, so persistence is skipped.
    let Some(session_id) = req.session_id.clone() else {
        return;
    };
    let repo = req.repo.clone().unwrap_or_else(|| namespace.to_string());
    let task = req
        .task
        .clone()
        .unwrap_or_else(|| state.default_task.clone());
    let mut metadata = response.metadata(req);
    if let Some(object) = metadata.as_object_mut() {
        // The full tool arguments are stored in event metadata for audit and
        // operator inspection. This is separate from classifier input, which is
        // bounded by `tool_authorization_classification_text`.
        object.insert("tool_arguments".to_string(), req.arguments.clone());
        object.insert(
            "available_tools".to_string(),
            serde_json::json!(req.available_tools),
        );
        if let Some(policy_metadata) = policy_metadata {
            object.insert("orchestration_policy".to_string(), policy_metadata);
        }
    }
    let trajectory_id = state
        .trajectory_capture_enabled
        .then_some(req.trajectory_id)
        .flatten();
    let attempt_index = trajectory_id.map(|_| req.attempt_index.unwrap_or(1));
    let event_role =
        trajectory_id.map(|_| crate::trajectory::EventRole::ToolCall.as_str().to_string());
    let event = AppendEventRequest {
        session_id: session_id.clone(),
        repo: repo.clone(),
        actor: Some("tool_mediator".to_string()),
        event_type: "tool_authorization_decision".to_string(),
        summary: format!(
            "tool authorization {} reason={} tool={}",
            response.decision, response.reason, response.attempted_tool
        ),
        evidence: None,
        metadata: Some(metadata),
        correlation_id: trajectory_id,
        parent_event_id: req.parent_event_id,
        trajectory_id,
        attempt_index,
        event_role,
        task: Some(task.clone()),
        error_type: None,
        error_description: None,
    };
    let state_bg = state.clone();
    let policy_classification_bg = policy_classification.clone();
    let policy_bg = policy.clone();
    spawn_bounded_background(state, "tool_authorization_event", async move {
        match db::append_event_from_request(
            &state_bg.pool,
            &state_bg.embedder,
            &state_bg.qdrant_url,
            &event,
        )
        .await
        {
            Ok((event_id, _indexed)) => {
                spawn_feature_extraction(&state_bg, &repo, &session_id, event.trajectory_id);
                if let (Some(classification), Some(policy)) = (policy_classification_bg, policy_bg)
                {
                    let mut classification = classification;
                    classification.event_id = event_id.clone();
                    // The policy row is written after the authorization event
                    // so the append-only ledger can point at a durable event id
                    // instead of a synthetic pre-insert placeholder.
                    if let Err(e) = orchestration_policy::persist_orchestration_policy(
                        &state_bg.pool,
                        &classification,
                        &policy,
                    )
                    .await
                    {
                        tracing::warn!(
                            target = "orchestration_policy",
                            repo = %repo,
                            task = %task,
                            event_id = %event_id,
                            "failed to persist tool authorization orchestration policy: {e}"
                        );
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    repo = %repo,
                    task = %task,
                    "failed to persist tool authorization event: {e}"
                );
            }
        }
    });
}
