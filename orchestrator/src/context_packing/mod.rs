pub mod build;
pub mod cache;
pub mod endpoint;
pub mod policy;

use crate::db;
use crate::state::*;
use crate::telemetry;
use build::build_cached_context;
pub(crate) use cache::cached_context_for_request;
use cache::context_cache_task;
pub use endpoint::context_pack;
pub(crate) use policy::context_task_category;
use policy::policy_context_cache_key;

pub(crate) async fn get_or_build_cached_context(
    state: &AppState,
    repo: &str,
    task: &str,
    session_id: Option<&str>,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    limit_override: Option<i64>,
    task_config: &TaskContextConfig,
    classification: Option<&crate::request_classification::RequestClassification>,
    policy_override: Option<&crate::orchestration_policy::OrchestrationPolicy>,
) -> Result<CachedContext, anyhow::Error> {
    get_or_build_cached_context_inner(
        state,
        repo,
        task,
        session_id,
        trajectory,
        limit_override,
        task_config,
        classification,
        policy_override,
        true,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn get_or_build_cached_context_inner(
    state: &AppState,
    repo: &str,
    task: &str,
    session_id: Option<&str>,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    limit_override: Option<i64>,
    task_config: &TaskContextConfig,
    classification: Option<&crate::request_classification::RequestClassification>,
    policy_override: Option<&crate::orchestration_policy::OrchestrationPolicy>,
    record_metrics: bool,
) -> Result<CachedContext, anyhow::Error> {
    let event_count = db::count_events_for_repo(&state.pool, repo).await?;
    let policy_cache_key = policy_override.map(policy_context_cache_key);
    let cache_task = context_cache_task(
        task,
        session_id,
        trajectory,
        limit_override,
        state.feature_extraction_enabled,
        policy_cache_key.as_deref(),
    );
    let cache_key = context_cache_key(repo, &cache_task, event_count);
    if let Some(cached) = state.cache.get(&cache_key) {
        let mut cached = cached;
        cached.stats.cache_hit = true;
        cached.stats.build_ms = 0;
        if record_metrics {
            telemetry::record_context_pack(&state.metrics, &cached.stats);
        }
        return Ok(cached);
    }

    let cached = build_cached_context(
        state,
        repo,
        task,
        session_id,
        trajectory,
        limit_override,
        task_config,
        classification,
        policy_override,
    )
    .await?;

    let replaced = state.cache.put(cache_key, cached.clone());
    if replaced > 0 {
        telemetry::record_context_cache_replacement(&state.metrics, replaced);
    }
    if record_metrics {
        telemetry::record_context_pack(&state.metrics, &cached.stats);
    }
    Ok(cached)
}

#[cfg(test)]
mod tests;
