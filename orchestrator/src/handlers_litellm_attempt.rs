use std::sync::Arc;
use uuid::Uuid;

use crate::client_capture::RawHttpCapture;
use crate::litellm::{LiteLlmCallFinalizer, RouteSelection};
use crate::local_reasoning::{add_local_reasoning_metadata, LocalReasoningSelection};
use crate::state::AppState;

pub(crate) struct PreparedLiteLlmAttempt {
    pub(crate) finalizer: LiteLlmCallFinalizer,
    pub(crate) vllm_cache_before: Option<(String, crate::vllm_metrics::VllmCacheSnapshot)>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn prepare_litellm_attempt(
    state: &Arc<AppState>,
    req: &mut serde_json::Value,
    capture: &mut RawHttpCapture,
    endpoint: &'static str,
    requested_model: String,
    namespace: String,
    repo: String,
    task: String,
    request_event_id: Option<Uuid>,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    context_pack_id: Option<Uuid>,
    route: &RouteSelection,
    context_pack_hash: Option<String>,
    baseline_arm: Option<String>,
    reasoning_selection: LocalReasoningSelection,
) -> PreparedLiteLlmAttempt {
    let cache_policy = crate::litellm::exact_cache_decision(endpoint, req, false);
    let mut attempt = crate::litellm::new_attempt(
        request_event_id,
        trajectory.map(|trajectory| trajectory.trajectory_id),
        context_pack_id,
        namespace,
        repo,
        task,
        endpoint,
        requested_model,
        route,
        cache_policy,
        context_pack_hash,
        baseline_arm,
    );
    add_local_reasoning_metadata(&mut attempt, reasoning_selection);
    capture.attempt_id = Some(attempt.attempt_id);
    crate::litellm::add_agentic_os_metadata(req, &attempt);
    capture.forwarded_request_body = Some(crate::client_capture::to_json_bytes(req));
    let vllm_cache_before = crate::vllm_metrics::cache_snapshot(state).await;
    let finalizer = crate::litellm::LiteLlmCallFinalizer::begin(state.pool.clone(), attempt).await;

    PreparedLiteLlmAttempt {
        finalizer,
        vllm_cache_before,
    }
}
