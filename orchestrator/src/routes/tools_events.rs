use serde_json::Value;

use crate::background::spawn_bounded_background;
use crate::event_capture::spawn_feature_extraction;
use crate::orchestration_policy;
use crate::state::{AppState, AppendEventRequest};
use crate::{db, tool_mediation};

pub(crate) fn persist_tool_authorization_event(
    state: &AppState,
    namespace: &str,
    req: &tool_mediation::ToolAuthorizeRequest,
    response: &tool_mediation::ToolAuthorizeResponse,
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
