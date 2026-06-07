use serde_json::Value;

use crate::context_packing::{cached_context_for_request, context_task_category};
use crate::event_capture::maybe_write_context_pack_event;
use crate::state::AppState;
use crate::system_context::{inject_system_context, inject_system_context_anthropic};

/// Pack orchestrator context into an Anthropic-format request's system field.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn pack_context_into_anthropic_req(
    state: &AppState,
    req: &mut Value,
    session_id: Option<&str>,
    repo: &str,
    task: &str,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    parent_event_id: Option<uuid::Uuid>,
    classification: Option<&crate::request_classification::RequestClassification>,
    policy: Option<&crate::orchestration_policy::OrchestrationPolicy>,
) -> (Option<uuid::Uuid>, Option<String>) {
    let task_category = context_task_category(task, classification);
    let task_config = crate::state::TaskContextConfig::for_category(task_category);
    let cached = cached_context_for_request(
        state,
        repo,
        task,
        session_id,
        trajectory,
        &task_config,
        classification,
        policy,
    );
    let context_pack_id = maybe_write_context_pack_event(
        state,
        session_id,
        repo,
        task,
        trajectory,
        parent_event_id,
        &cached.stats,
        task_config.char_budget / 4,
    );
    let context_pack_hash = Some(crate::litellm::context_pack_hash(&cached.context));
    inject_system_context_anthropic(req, &cached.context);
    (context_pack_id, context_pack_hash)
}

/// Pack orchestrator context into an OpenAI-shaped request.
/// Sets a default model if absent, fetches memory events, builds context string,
/// and injects it as a system message.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn pack_context_into_req(
    state: &AppState,
    req: &mut Value,
    session_id: Option<&str>,
    repo: &str,
    task: &str,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    parent_event_id: Option<uuid::Uuid>,
    classification: Option<&crate::request_classification::RequestClassification>,
    policy: Option<&crate::orchestration_policy::OrchestrationPolicy>,
) -> (Option<uuid::Uuid>, Option<String>) {
    if req.get("model").is_none() {
        req["model"] = Value::String(state.default_model.clone());
    }
    let task_category = context_task_category(task, classification);
    let task_config = crate::state::TaskContextConfig::for_category(task_category);
    let cached = cached_context_for_request(
        state,
        repo,
        task,
        session_id,
        trajectory,
        &task_config,
        classification,
        policy,
    );
    let context_pack_id = maybe_write_context_pack_event(
        state,
        session_id,
        repo,
        task,
        trajectory,
        parent_event_id,
        &cached.stats,
        task_config.char_budget / 4,
    );
    let context_pack_hash = Some(crate::litellm::context_pack_hash(&cached.context));
    inject_system_context(req, &cached.context);
    (context_pack_id, context_pack_hash)
}
