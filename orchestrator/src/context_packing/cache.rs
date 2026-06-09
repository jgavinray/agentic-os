use crate::background::spawn_bounded_background;
use crate::context_packing_policy::policy_context_cache_key;
use crate::db;
use crate::state::{AppState, CachedContext, ContextPackStats, TaskContextConfig};
use crate::telemetry;

pub(crate) fn context_cache_task(
    task: &str,
    session_id: Option<&str>,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    limit_override: Option<i64>,
    feature_extraction_enabled: bool,
    policy_cache_key: Option<&str>,
) -> String {
    let mut cache_task = limit_override
        .map(|limit| format!("{task}:limit={limit}"))
        .unwrap_or_else(|| task.to_string());
    if let Some(policy_cache_key) = policy_cache_key {
        cache_task = format!("{cache_task}:policy={policy_cache_key}");
    }
    if feature_extraction_enabled {
        if let Some(trajectory) = trajectory {
            format!("{cache_task}:trajectory={}", trajectory.trajectory_id)
        } else if let Some(session_id) = session_id {
            format!("{cache_task}:session={session_id}")
        } else {
            cache_task
        }
    } else {
        cache_task
    }
}

pub(crate) fn context_cache_prefix(repo: &str, cache_task: &str) -> String {
    format!("{repo}:{cache_task}:")
}

pub(crate) fn cached_context_for_request(
    state: &AppState,
    repo: &str,
    task: &str,
    session_id: Option<&str>,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    task_config: &TaskContextConfig,
    classification: Option<&crate::request_classification::RequestClassification>,
    policy_override: Option<&crate::orchestration_policy::OrchestrationPolicy>,
) -> CachedContext {
    let policy_cache_key = policy_override.map(policy_context_cache_key);
    let cache_task = context_cache_task(
        task,
        session_id,
        trajectory,
        None,
        state.feature_extraction_enabled,
        policy_cache_key.as_deref(),
    );
    let cache_prefix = context_cache_prefix(repo, &cache_task);
    spawn_context_cache_refresh(
        state,
        repo,
        task,
        session_id,
        trajectory,
        None,
        task_config,
        classification.cloned(),
        policy_override.cloned(),
        cache_prefix.clone(),
    );

    if let Some(cached) = state.cache.latest_by_prefix(&cache_prefix) {
        let mut cached = cached;
        cached.stats.cache_hit = true;
        cached.stats.build_ms = 0;
        telemetry::record_context_pack(&state.metrics, &cached.stats);
        return cached;
    }

    let context = db::build_context(repo, task, &[], &[], &[], task_config.char_budget);
    let stats = ContextPackStats {
        build_ms: 0,
        context_chars: context.len(),
        context_tokens_estimate: db::estimate_tokens(&context),
        token_budget: task_config.char_budget / 4,
        cache_hit: false,
        ..Default::default()
    };
    telemetry::record_context_pack(&state.metrics, &stats);
    CachedContext {
        context,
        memories: vec![],
        cached_at: std::time::Instant::now(),
        stats,
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_context_cache_refresh(
    state: &AppState,
    repo: &str,
    task: &str,
    session_id: Option<&str>,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    limit_override: Option<i64>,
    task_config: &TaskContextConfig,
    classification: Option<crate::request_classification::RequestClassification>,
    policy_override: Option<crate::orchestration_policy::OrchestrationPolicy>,
    refresh_key: String,
) {
    if !state.cache.try_begin_refresh(refresh_key.clone()) {
        return;
    }

    let state_bg = state.clone();
    let repo = repo.to_string();
    let task = task.to_string();
    let session_id = session_id.map(str::to_string);
    let task_config = task_config.clone();
    spawn_bounded_background(state, "context_cache_refresh", async move {
        if let Err(e) = crate::context_packing::get_or_build_cached_context_inner(
            &state_bg,
            &repo,
            &task,
            session_id.as_deref(),
            trajectory,
            limit_override,
            &task_config,
            classification.as_ref(),
            policy_override.as_ref(),
            false,
        )
        .await
        {
            tracing::warn!(
                repo = %repo,
                task = %task,
                "background context cache refresh failed: {e}"
            );
        }
        state_bg.cache.finish_refresh(&refresh_key);
    });
}
