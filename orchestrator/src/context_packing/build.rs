use crate::context_packing_policy::{
    apply_orchestration_context_limits, context_task_category, memory_levels_used,
    orchestration_context_source_allowed, retrieved_event_ids,
};
use crate::context_search::{hybrid_search, HybridSearchResult};
use crate::db;
use crate::state::{AppState, CachedContext, ContextPolicy, TaskContextConfig};
use crate::telemetry;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_cached_context(
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

    Ok(CachedContext {
        context,
        memories,
        cached_at: std::time::Instant::now(),
        stats,
    })
}
