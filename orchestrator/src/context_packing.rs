use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::auth::check_auth;
use crate::background::spawn_bounded_background;
use crate::context_search::{hybrid_search, HybridSearchResult};
use crate::db;
use crate::state::*;
use crate::telemetry;

fn context_cache_task(
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

fn context_cache_prefix(repo: &str, cache_task: &str) -> String {
    format!("{repo}:{cache_task}:")
}

async fn get_or_build_cached_context(
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
async fn get_or_build_cached_context_inner(
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

    let build_started = std::time::Instant::now();
    let mut policy = ContextPolicy::for_category(context_task_category(task, classification));
    if let Some(orchestration_policy) = policy_override {
        apply_orchestration_context_limits(&mut policy, orchestration_policy);
    }
    if let Some(limit) = limit_override {
        policy.l0_recent_limit = limit.max(0);
    } else {
        policy.l0_recent_limit = policy.l0_recent_limit.min(task_config.max_events);
    }
    let failure_signatures = if state.execution_feedback_enabled {
        crate::execution_feedback::extract_failure_signatures(task)
    } else {
        vec![]
    };
    let failure_history_limit = failure_signatures.len().max(1) as i64;

    let operational_constraints = async {
        if !state.feature_extraction_enabled {
            return Ok((vec![], vec![]));
        }
        crate::feature_extraction::operational_constraints_for_context(
            &state.pool,
            repo,
            session_id,
            trajectory.map(|trajectory| trajectory.trajectory_id),
            state.operational_constraints_token_budget,
        )
        .await
    };

    let (
        evidence_result,
        hybrid_result,
        errors_result,
        failure_history_result,
        operational_constraints_result,
    ) = tokio::join!(
        db::get_context_evidence_for_policy(&state.pool, repo, &policy),
        async {
            if orchestration_context_source_allowed(
                policy_override,
                crate::orchestration_policy::ContextSource::QdrantSemantic,
            ) {
                hybrid_search(state, repo, task, task_config.semantic_limit).await
            } else {
                HybridSearchResult {
                    hits: vec![],
                    semantic_hits: 0,
                    fts_hits: 0,
                    deduped_hits: 0,
                }
            }
        },
        db::get_active_errors(&state.pool, repo, 5),
        async {
            if state.execution_feedback_enabled && !failure_signatures.is_empty() {
                db::get_failure_history_for_signatures(
                    &state.pool,
                    repo,
                    &failure_signatures,
                    failure_history_limit,
                )
                .await
            } else {
                Ok(vec![])
            }
        },
        operational_constraints,
    );

    let mut evidence = evidence_result?;
    evidence.failure_history = failure_history_result.unwrap_or_else(|e| {
        tracing::warn!(repo, task, "failure history lookup failed: {e}");
        vec![]
    });
    let (constraints, suppressed_constraints) =
        operational_constraints_result.unwrap_or_else(|e| {
            tracing::warn!(repo, task, "operational constraint lookup failed: {e}");
            telemetry::record_feature_extraction_failure("constraint_build");
            (vec![], vec![])
        });
    for constraint in &constraints {
        telemetry::record_operational_constraint_injected(&constraint.constraint_type);
    }
    for suppressed in &suppressed_constraints {
        telemetry::record_operational_constraint_suppressed(
            &suppressed.constraint_type,
            &suppressed.reason,
        );
    }
    evidence.operational_constraints = constraints;
    let errors = errors_result.unwrap_or_default();
    let memories = evidence.memories();
    let compiler_output = crate::context_compiler::ContextCompiler::new(&state.pool, &state.http)
        .compile(crate::context_compiler::CompilerRequest {
            repo: repo.to_string(),
            task: task.to_string(),
            session_id: session_id.map(str::to_string),
            policy: policy_override.cloned(),
            runtime: crate::context_compiler::RuntimeContext {
                default_model: state.default_model.clone(),
                litellm_url: state.litellm_url.clone(),
                qdrant_url: state.qdrant_url.clone(),
                summarizer_url: state.summarizer_url.clone(),
                summarizer_model: state.summarizer_model.clone(),
                total_recall_url: state.total_recall_url.clone(),
            },
        })
        .await;
    let (context, mut stats) = db::build_layered_context(
        repo,
        task,
        &evidence,
        &hybrid_result.hits,
        &errors,
        &policy,
        task_config.char_budget,
        state.failure_history_token_budget * 4,
    );
    let artifact_context = crate::context_compiler::render(&compiler_output);
    let context = if artifact_context.is_empty() {
        context
    } else {
        format!("{artifact_context}{context}")
    };
    stats.build_ms = build_started.elapsed().as_millis() as u64;
    stats.retrieval_semantic_hits = hybrid_result.semantic_hits;
    stats.retrieval_fts_hits = hybrid_result.fts_hits;
    stats.retrieval_deduped_hits = hybrid_result.deduped_hits;
    stats.retrieved_event_ids = retrieved_event_ids(&evidence, &hybrid_result.hits);
    stats.memory_levels_used = memory_levels_used(&evidence);
    stats.injected_failure_signatures = evidence
        .failure_history
        .iter()
        .map(|item| item.signature.clone())
        .chain(failure_signatures)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    stats.token_budget = task_config.char_budget / 4;
    stats.truncated = context.contains("[truncated:");
    stats.context_chars = context.len();
    stats.context_tokens_estimate = db::estimate_tokens(&context);
    let (stable_prefix_hash, dynamic_tail_hash) = crate::litellm::context_prefix_hashes(&context);
    stats.stable_prefix_hash = stable_prefix_hash;
    stats.dynamic_tail_hash = dynamic_tail_hash;
    stats.cache_hit = false;

    let cached = CachedContext {
        context,
        memories,
        cached_at: std::time::Instant::now(),
        stats,
    };

    let replaced = state.cache.put(cache_key, cached.clone());
    if replaced > 0 {
        telemetry::record_context_cache_replacement(&state.metrics, replaced);
    }
    if record_metrics {
        telemetry::record_context_pack(&state.metrics, &cached.stats);
    }
    Ok(cached)
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
        if let Err(e) = get_or_build_cached_context_inner(
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

pub(crate) fn context_task_category(
    task: &str,
    classification: Option<&crate::request_classification::RequestClassification>,
) -> TaskCategory {
    use crate::request_classification::{RequestComplexity, RequestIntent};
    let Some(classification) = classification else {
        return TaskCategory::from_task(task);
    };
    match classification.complexity {
        RequestComplexity::L0Trivial | RequestComplexity::L1Simple => TaskCategory::Narrow,
        RequestComplexity::L2Moderate => {
            if matches!(
                classification.intent,
                RequestIntent::Explain | RequestIntent::Search | RequestIntent::Classify
            ) {
                TaskCategory::Narrow
            } else {
                TaskCategory::Moderate
            }
        }
        RequestComplexity::L3Complex | RequestComplexity::L4ToolRequired => TaskCategory::Moderate,
        RequestComplexity::L5HighRisk | RequestComplexity::Unknown => TaskCategory::Narrow,
    }
}

fn policy_context_cache_key(policy: &crate::orchestration_policy::OrchestrationPolicy) -> String {
    let context = policy
        .context_sources
        .iter()
        .map(|source| source.as_str())
        .collect::<Vec<_>>()
        .join(".");
    format!(
        "{}.{}.{}",
        policy.edit_policy.as_str(),
        policy.validation_policy.as_str(),
        context
    )
}

fn orchestration_context_source_allowed(
    policy: Option<&crate::orchestration_policy::OrchestrationPolicy>,
    source: crate::orchestration_policy::ContextSource,
) -> bool {
    policy
        .map(|policy| policy.context_sources.contains(&source))
        .unwrap_or(true)
}

pub(crate) fn apply_orchestration_context_limits(
    policy: &mut ContextPolicy,
    orchestration_policy: &crate::orchestration_policy::OrchestrationPolicy,
) {
    if !orchestration_policy
        .context_sources
        .contains(&crate::orchestration_policy::ContextSource::PostgresEvents)
    {
        policy.l0_recent_limit = 0;
        policy.l1_limit = 0;
        policy.l2_limit = 0;
        policy.l3_limit = 0;
        policy.failure_limit = 0;
    }
    if !orchestration_policy
        .context_sources
        .contains(&crate::orchestration_policy::ContextSource::CompiledSummaries)
    {
        policy.l1_limit = policy.l1_limit.min(1);
        policy.l2_limit = 0;
        policy.l3_limit = 0;
    }
    if !orchestration_policy
        .context_sources
        .contains(&crate::orchestration_policy::ContextSource::ContextLedger)
    {
        policy.failure_limit = 0;
    }
}

fn retrieved_event_ids(evidence: &db::ContextEvidence, hits: &[SearchHit]) -> Vec<String> {
    evidence
        .l0_recent
        .iter()
        .chain(evidence.l1_matching.iter())
        .chain(evidence.l2_repo.iter())
        .chain(evidence.l3_project.iter())
        .chain(evidence.failures.iter())
        .chain(
            evidence
                .failure_history
                .iter()
                .flat_map(|item| std::iter::once(&item.failure).chain(item.remediation.iter())),
        )
        .map(|event| event.id.clone())
        .chain(hits.iter().map(|hit| hit.event_id.clone()))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn memory_levels_used(evidence: &db::ContextEvidence) -> Vec<String> {
    let mut levels = Vec::new();
    if !evidence.l0_recent.is_empty() {
        levels.push("l0".to_string());
    }
    if !evidence.l1_matching.is_empty() {
        levels.push("l1".to_string());
    }
    if !evidence.l2_repo.is_empty() {
        levels.push("l2".to_string());
    }
    if !evidence.l3_project.is_empty() {
        levels.push("l3".to_string());
    }
    if !evidence.failures.is_empty() {
        levels.push("failures".to_string());
    }
    if !evidence.failure_history.is_empty() {
        levels.push("failure_history".to_string());
    }
    if !evidence.operational_constraints.is_empty() {
        levels.push("operational_constraints".to_string());
    }
    levels
}

#[tracing::instrument(name = "handler.context_pack", skip(state, headers, req), fields(repo = %req.repo, task = %req.task))]
pub async fn context_pack(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<ContextPackRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let task_category = TaskCategory::from_task(&req.task);
    let task_config = TaskContextConfig::for_category(task_category);

    let cached = match get_or_build_cached_context(
        &state,
        &req.repo,
        &req.task,
        None,
        None,
        req.limit,
        &task_config,
        None,
        None,
    )
    .await
    {
        Ok(cached) => cached,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": "failed_to_fetch_context", "detail": e.to_string()})),
            ).into_response();
        }
    };

    axum::Json(ContextPackResponse {
        repo: req.repo,
        task: req.task,
        context: cached.context,
        memories: cached.memories,
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    #[test]
    fn context_pack_parallelizes_context_io_calls() {
        let src = include_str!("context_packing.rs");
        let ctx_start = src
            .find("async fn get_or_build_cached_context")
            .expect("get_or_build_cached_context not found in source");
        let ctx_body: String = src[ctx_start..].chars().take(6500).collect();

        assert!(ctx_body.contains("tokio::join!"));
        assert!(ctx_body.contains("db::get_context_evidence_for_policy"));
        assert!(ctx_body.contains("hybrid_search"));
        assert!(ctx_body.contains("db::get_active_errors"));
        assert!(ctx_body.contains("db::get_failure_history_for_signatures"));

        let join_block_start = ctx_body
            .find("tokio::join!")
            .expect("tokio::join! not found");
        let join_block: String = ctx_body[join_block_start..].chars().take(1500).collect();
        assert!(
            join_block.contains("get_context_evidence_for_policy")
                && join_block.contains("hybrid_search")
                && join_block.contains("get_active_errors")
                && join_block.contains("get_failure_history_for_signatures")
        );
    }

    #[test]
    fn context_pack_preserves_error_propagation_for_events() {
        let src = include_str!("context_packing.rs");
        let ctx_start = src
            .find("pub async fn context_pack")
            .expect("context_pack not found");
        let ctx_body = &src[ctx_start..ctx_start + 2000];
        assert!(ctx_body.contains("INTERNAL_SERVER_ERROR"));
    }
}
