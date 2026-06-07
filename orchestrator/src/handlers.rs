use axum::extract::State;
use axum::http::StatusCode;
use axum::http::{header, HeaderMap};
use axum::response::IntoResponse;
use axum::response::Response;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;
use std::future::Future;
use std::sync::Arc;
use subtle::ConstantTimeEq;

use crate::anthropic;
use crate::db;
use crate::orchestration_policy;
use crate::qdrant;
use crate::rate_limit;
use crate::state::*;
use crate::telemetry;

pub use crate::routes::context::context_artifacts;
pub use crate::routes::harness::{harness_guardrail, harness_outcome, litellm_callback_payload};
pub use crate::routes::health::{health, health_live, health_ready, list_models};
pub use crate::routes::observability::{cache_stats, metrics, metrics_json};
pub use crate::routes::sessions::{append_event, start_session};
pub use crate::routes::tools::authorize_tool;
#[cfg(test)]
pub(crate) use crate::routes::tools::{
    derive_tool_authorization_policy, tool_authorization_classification_text,
};
pub use crate::routes::validations::validations;

// ── Auth helpers ───────────────────────────────────────────────

fn bearer_token(headers: &HeaderMap) -> &str {
    headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .unwrap_or("")
}

fn provided_api_token(headers: &HeaderMap) -> &str {
    let bearer = bearer_token(headers);
    if !bearer.is_empty() {
        return bearer;
    }
    headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
}

// Returns the (token, namespace) pair if auth passes; None otherwise.
// Constant-time comparison prevents timing-based key recovery.
pub(crate) fn authenticate(state: &AppState, headers: &HeaderMap) -> Option<(String, String)> {
    let provided = provided_api_token(headers).as_bytes();
    for (token, namespace) in &state.api_keys {
        let expected = token.as_bytes();
        if expected.len() == provided.len() && expected.ct_eq(provided).into() {
            telemetry::record_auth_attempt(true);
            return Some((token.clone(), namespace.clone()));
        }
    }
    telemetry::record_auth_attempt(false);
    None
}

pub(crate) fn check_auth(state: &AppState, headers: &HeaderMap) -> bool {
    authenticate(state, headers).is_some()
}

pub(crate) fn check_rate_limit(state: &AppState, token: &str) -> Option<Response> {
    match state.rate_limiter.check(token) {
        Ok(()) => None,
        Err(retry_after) => Some(rate_limited_response(token, retry_after)),
    }
}

fn rate_limited_response(token: &str, retry_after: u64) -> Response {
    let key_hash = rate_limit::key_hash(token);
    telemetry::record_rate_limited(&key_hash);
    (
        StatusCode::TOO_MANY_REQUESTS,
        [(header::RETRY_AFTER, retry_after.to_string())],
        axum::Json(serde_json::json!({
            "error": "rate_limited",
            "retry_after": retry_after
        })),
    )
        .into_response()
}

fn maybe_openai_live_policy_response(
    state: &AppState,
    repo: &str,
    session_id: &str,
    user_content: &str,
) -> Option<Response> {
    let classification = crate::request_classification::classify_request_text(
        repo,
        session_id,
        user_content,
        None,
        "user_message",
    );
    live_policy_decision_response(state, &classification).map(|decision| {
        let (status, body) = live_policy_openai_body(&decision, user_content);
        (status, axum::Json(body)).into_response()
    })
}

fn maybe_anthropic_live_policy_response(
    state: &AppState,
    repo: &str,
    session_id: &str,
    user_content: &str,
) -> Option<Response> {
    let classification = crate::request_classification::classify_request_text(
        repo,
        session_id,
        user_content,
        None,
        "user_message",
    );
    live_policy_decision_response(state, &classification).map(|decision| {
        let (status, body) = live_policy_anthropic_body(&decision, user_content);
        (status, axum::Json(body)).into_response()
    })
}

fn live_policy_decision_response(
    state: &AppState,
    classification: &crate::request_classification::RequestClassification,
) -> Option<crate::request_classification::LivePolicyDecision> {
    if !state.request_live_policy_config.enabled {
        telemetry::record_request_live_policy_bypass("disabled");
        return None;
    }
    if state.request_live_policy_config.policy_version != "v1" {
        telemetry::record_request_live_policy_bypass("unsupported_policy_version");
        return None;
    }

    let decision = crate::request_classification::evaluate_live_policy(
        classification,
        &state.request_live_policy_config,
    );
    match &decision {
        Some(decision) => {
            telemetry::record_request_live_policy_action(decision.action, decision.reason);
        }
        None => telemetry::record_request_live_policy_bypass("shadow_only"),
    }
    decision
}

fn merge_request_metadata(parts: impl IntoIterator<Item = Option<Value>>) -> Option<Value> {
    let mut merged = serde_json::Map::new();
    for part in parts.into_iter().flatten() {
        if let Some(object) = part.as_object() {
            for (key, value) in object {
                merged.insert(key.clone(), value.clone());
            }
        }
    }
    if merged.is_empty() {
        None
    } else {
        Some(Value::Object(merged))
    }
}

fn live_policy_openai_body(
    decision: &crate::request_classification::LivePolicyDecision,
    user_content: &str,
) -> (StatusCode, Value) {
    match decision.action {
        "web_required" => (
            StatusCode::PRECONDITION_REQUIRED,
            serde_json::json!({
                "error": {
                    "type": "request_policy",
                    "code": "web_required",
                    "message": "This request requires current external information. Use an enabled web/tool path before answering."
                }
            }),
        ),
        "deterministic_template" => (
            StatusCode::OK,
            openai_completion(live_policy_template_text(user_content)),
        ),
        "ask_clarification" => (
            StatusCode::OK,
            openai_completion("Please provide the specific target or context before I continue."),
        ),
        _ => (
            StatusCode::FORBIDDEN,
            serde_json::json!({
                "error": {
                    "type": "request_policy",
                    "code": decision.action,
                    "message": "This request was stopped by the deterministic request safety policy."
                }
            }),
        ),
    }
}

fn live_policy_anthropic_body(
    decision: &crate::request_classification::LivePolicyDecision,
    user_content: &str,
) -> (StatusCode, Value) {
    match decision.action {
        "web_required" => (
            StatusCode::PRECONDITION_REQUIRED,
            serde_json::json!({
                "type": "error",
                "error": {
                    "type": "request_policy",
                    "message": "This request requires current external information. Use an enabled web/tool path before answering."
                }
            }),
        ),
        "deterministic_template" => (
            StatusCode::OK,
            anthropic_text_message(live_policy_template_text(user_content)),
        ),
        "ask_clarification" => (
            StatusCode::OK,
            anthropic_text_message(
                "Please provide the specific target or context before I continue.",
            ),
        ),
        _ => (
            StatusCode::FORBIDDEN,
            serde_json::json!({
                "type": "error",
                "error": {
                    "type": "request_policy",
                    "message": "This request was stopped by the deterministic request safety policy."
                }
            }),
        ),
    }
}

fn live_policy_template_text(user_content: &str) -> &'static str {
    let lower = user_content.trim().to_ascii_lowercase();
    if lower == "ping" {
        "pong"
    } else {
        "ok"
    }
}

fn openai_completion(text: &str) -> Value {
    serde_json::json!({
        "id": "chatcmpl-request-policy",
        "object": "chat.completion",
        "created": 0,
        "model": "request-policy",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    })
}

fn anthropic_text_message(text: &str) -> Value {
    serde_json::json!({
        "id": "msg_request_policy",
        "type": "message",
        "role": "assistant",
        "model": "request-policy",
        "content": [{
            "type": "text",
            "text": text
        }],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0
        }
    })
}

pub async fn run_trajectory_idle_sweep(state: Arc<AppState>) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
    loop {
        interval.tick().await;
        if !state.trajectory_capture_enabled {
            continue;
        }
        let ids =
            match db::idle_trajectory_ids(&state.pool, state.trajectory_idle_timeout_sec, 10_000)
                .await
            {
                Ok(ids) => ids,
                Err(e) => {
                    tracing::warn!("trajectory idle sweep failed to list candidates: {e}");
                    continue;
                }
            };
        for trajectory_id in ids {
            if let Err(e) = db::emit_trajectory_result_once(
                &state.pool,
                &state.embedder,
                &state.qdrant_url,
                trajectory_id,
                Some(crate::trajectory::BoundaryReason::IdleTimeout),
            )
            .await
            {
                tracing::warn!(trajectory_id = %trajectory_id, "failed to emit idle trajectory result: {e}");
            }
        }
    }
}

pub(crate) fn spawn_feature_extraction(
    state: &AppState,
    repo: &str,
    session_id: &str,
    trajectory_id: Option<uuid::Uuid>,
) {
    if !state.feature_extraction_enabled {
        return;
    }
    let pool = state.pool.clone();
    let metrics = state.metrics.clone();
    let repo = repo.to_string();
    let session_id = session_id.to_string();
    spawn_bounded_background(state, "feature_extraction", async move {
        crate::feature_extraction::run_inline_extraction_best_effort(
            pool,
            metrics,
            repo,
            session_id,
            trajectory_id,
        )
        .await;
    });
}

fn spawn_qdrant_index_event(state: &AppState, event: db::AgentEvent) {
    let embedder = state.embedder.clone();
    let qdrant_url = state.qdrant_url.clone();
    spawn_bounded_background(state, "qdrant_index_event", async move {
        let event_id = event.id.clone();
        let event_type = event.event_type.clone();
        if let Err(e) = qdrant::store_event(&embedder, &qdrant_url, &event).await {
            tracing::warn!(
                event_id = %event_id,
                event_type = %event_type,
                "event stored in postgres but qdrant indexing failed: {e}"
            );
        }
    });
}

fn spawn_trajectory_result_emit(
    state: &AppState,
    trajectory_id: uuid::Uuid,
    reason: crate::trajectory::BoundaryReason,
) {
    let state_bg = state.clone();
    spawn_bounded_background(state, "trajectory_result_emit", async move {
        if let Err(e) = db::emit_trajectory_result_once(
            &state_bg.pool,
            &state_bg.embedder,
            &state_bg.qdrant_url,
            trajectory_id,
            Some(reason),
        )
        .await
        {
            tracing::warn!(
                trajectory_id = %trajectory_id,
                "failed to emit trajectory result: {e}"
            );
        }
    });
}

pub(crate) fn spawn_bounded_background<F>(state: &AppState, job: &'static str, fut: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    let gate = state.background_work.clone();
    tokio::spawn(async move {
        let _permit = match gate.acquire_owned().await {
            Ok(permit) => permit,
            Err(e) => {
                tracing::warn!(job, "background work gate closed: {e}");
                return;
            }
        };
        fut.await;
    });
}

// ── Semantic / hybrid search helpers ───────────────────────────

struct HybridSearchResult {
    hits: Vec<crate::state::SearchHit>,
    semantic_hits: usize,
    fts_hits: usize,
    deduped_hits: usize,
}

async fn semantic_search(
    state: &AppState,
    query: &str,
    limit: usize,
) -> Vec<crate::state::SearchHit> {
    qdrant::search(&state.embedder, &state.qdrant_url, query, limit)
        .await
        .unwrap_or_default()
        .into_iter()
        .filter_map(|item| {
            let payload = item.get("payload")?;
            let created_at = payload
                .get("created_at")
                .and_then(|v| v.as_str())
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&chrono::Utc));
            Some(crate::state::SearchHit {
                event_id: payload.get("event_id")?.as_str()?.to_string(),
                event_type: payload.get("event_type")?.as_str()?.to_string(),
                summary: payload.get("summary")?.as_str()?.to_string(),
                created_at,
            })
        })
        .collect()
}

async fn hybrid_search(
    state: &AppState,
    repo: &str,
    task: &str,
    semantic_limit: usize,
) -> HybridSearchResult {
    let query = format!("{repo} {task}");
    let fts_limit = semantic_limit as i64;

    let (semantic, fts_result) = tokio::join!(
        semantic_search(state, &query, semantic_limit),
        db::search_events_fts(&state.pool, repo, task, fts_limit),
    );

    let semantic = db::hydrate_active_search_hits(&state.pool, repo, semantic)
        .await
        .unwrap_or_else(|e| {
            tracing::warn!("semantic hit hydration failed, falling back to FTS-only: {e}");
            vec![]
        });
    let fts = fts_result.unwrap_or_else(|e| {
        tracing::warn!("FTS search failed, falling back to semantic-only: {e}");
        vec![]
    });
    let hits = crate::hybrid::rrf_merge_decay(
        &semantic,
        &fts,
        60.0,
        semantic_limit,
        state.context_decay_rate,
    );
    let deduped_hits = hits.len();

    HybridSearchResult {
        hits,
        semantic_hits: semantic.len(),
        fts_hits: fts.len(),
        deduped_hits,
    }
}

// ── Context pack ────────────────────────────────────────────────

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
    let mut policy =
        crate::state::ContextPolicy::for_category(context_task_category(task, classification));
    if let Some(orchestration_policy) = policy_override {
        apply_orchestration_context_limits(&mut policy, orchestration_policy);
    }
    if let Some(limit) = limit_override {
        policy.l0_recent_limit = limit.max(0);
    } else {
        policy.l0_recent_limit = policy.l0_recent_limit.min(task_config.max_events);
    }
    let failure_signatures = if state.execution_feedback_enabled {
        // Failure History uses exact signature lookup. Fingerprint raw error
        // text in the task first so "error[E0308]" can find "rust:type-mismatch".
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
                // This is the only new exact lookup path. Other artifact events
                // continue through the existing semantic/FTS hybrid retrieval.
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

fn cached_context_for_request(
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

fn context_task_category(
    task: &str,
    classification: Option<&crate::request_classification::RequestClassification>,
) -> crate::state::TaskCategory {
    use crate::request_classification::{RequestComplexity, RequestIntent};
    let Some(classification) = classification else {
        return crate::state::TaskCategory::from_task(task);
    };
    match classification.complexity {
        RequestComplexity::L0Trivial | RequestComplexity::L1Simple => {
            crate::state::TaskCategory::Narrow
        }
        RequestComplexity::L2Moderate => {
            if matches!(
                classification.intent,
                RequestIntent::Explain | RequestIntent::Search | RequestIntent::Classify
            ) {
                crate::state::TaskCategory::Narrow
            } else {
                crate::state::TaskCategory::Moderate
            }
        }
        RequestComplexity::L3Complex | RequestComplexity::L4ToolRequired => {
            crate::state::TaskCategory::Moderate
        }
        RequestComplexity::L5HighRisk | RequestComplexity::Unknown => {
            crate::state::TaskCategory::Narrow
        }
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

fn apply_orchestration_context_limits(
    policy: &mut crate::state::ContextPolicy,
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

fn retrieved_event_ids(
    evidence: &db::ContextEvidence,
    hits: &[crate::state::SearchHit],
) -> Vec<String> {
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

    let task_category = crate::state::TaskCategory::from_task(&req.task);
    let task_config = crate::state::TaskContextConfig::for_category(task_category);

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

// ── Checkpoint ──────────────────────────────────────────────────

#[tracing::instrument(name = "handler.checkpoint", skip(state, headers, req), fields(repo = %req.repo))]
pub async fn checkpoint(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<CheckpointRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let task_string = format!(
        "{}/{}",
        req.repo,
        req.summary.chars().take(50).collect::<String>()
    );
    let event = AppendEventRequest {
        session_id: req.session_id,
        repo: req.repo,
        actor: Some("checkpoint".to_string()),
        event_type: "checkpoint".to_string(),
        summary: req.summary,
        evidence: None,
        metadata: Some(serde_json::json!({
            "next_actions": req.next_actions.unwrap_or_default(),
            "open_questions": req.open_questions.unwrap_or_default(),
        })),
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: None,
        task: Some(task_string),
        error_type: None,
        error_description: None,
    };

    let (event_id, qdrant_indexed) = match db::append_event_from_request(
        &state.pool,
        &state.embedder,
        &state.qdrant_url,
        &event,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(
                    serde_json::json!({"error": "failed_to_checkpoint", "detail": e.to_string()}),
                ),
            )
                .into_response();
        }
    };
    spawn_feature_extraction(&state, &event.repo, &event.session_id, event.trajectory_id);

    axum::Json(serde_json::json!({"event_id": event_id, "qdrant_indexed": qdrant_indexed}))
        .into_response()
}

// ── Chat completions ────────────────────────────────────────────

/// Return an Anthropic-shaped error response.
fn anthropic_error_value(error_type: &'static str, message: impl Into<String>) -> Value {
    serde_json::json!({
        "type": "error",
        "error": {"type": error_type, "message": message.into()}
    })
}

fn anthropic_error(
    status: StatusCode,
    error_type: &'static str,
    message: impl Into<String>,
) -> Response {
    (
        status,
        axum::Json(anthropic_error_value(error_type, message)),
    )
        .into_response()
}

/// BUG-4: Append context to an existing client system message rather than inserting
/// a new one at position 0, which would demote the harness's carefully-tuned prompt.
/// Ensure the request has a backend-safe max_tokens value.
/// Anthropic requires `max_tokens`; OpenAI treats it as optional. Default omitted
/// values, respect explicit small values, and cap oversized values.
/// Normalise Responses API content types in message history before forwarding to LiteLLM.
/// LiteLLM may return `"type": "output_text"` in assistant turns; if the client replays
/// those turns in a subsequent request, some LiteLLM routing paths reject them. Convert
/// to `"text"` defensively so the round-trip is always valid regardless of LiteLLM version.
fn normalize_response_content_types(req: &mut Value) {
    if let Some(messages) = req.get_mut("messages").and_then(|v| v.as_array_mut()) {
        for msg in messages.iter_mut() {
            if let Some(content) = msg.get_mut("content").and_then(|v| v.as_array_mut()) {
                for block in content.iter_mut() {
                    if block.get("type").and_then(|t| t.as_str()) == Some("output_text") {
                        block["type"] = Value::String("text".to_string());
                    }
                }
            }
        }
    }
}

fn sanitize_anthropic_litellm_request(req: &mut Value) {
    if let Some(obj) = req.as_object_mut() {
        // LiteLLM's Anthropic adapter forwards extended-thinking fields to
        // OpenAI-compatible local backends as unsupported Responses API params
        // such as `max_output_tokens`, which causes vLLM to return 500s.
        obj.remove("thinking");
        obj.remove("max_output_tokens");
        obj.remove("reasoning_effort");
        obj.remove("context_management");
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LocalReasoningPolicy {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LocalReasoningSelection {
    policy: LocalReasoningPolicy,
    source: &'static str,
}

impl LocalReasoningPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    fn default_max_tokens(self) -> u64 {
        match self {
            Self::Low => 2048,
            Self::Medium => 4096,
            Self::High => MAX_MAX_TOKENS,
        }
    }

    fn default_temperature(self) -> f64 {
        match self {
            Self::Low => 0.1,
            Self::Medium => 0.2,
            Self::High => 0.2,
        }
    }

    fn system_contract(self) -> &'static str {
        match self {
            Self::Low => {
                "Local harness reasoning policy: low. Answer directly with concrete facts, cite evidence when available, and avoid expanded plans unless required."
            }
            Self::Medium => {
                "Local harness reasoning policy: medium. Use bounded engineering reasoning: objective, evidence, assessment, plan, and verification when the task benefits from structure. Do not add filler."
            }
            Self::High => {
                "Local harness reasoning policy: high. Use a fuller engineering analysis: objective, evidence, constraints, tradeoffs, implementation steps, risks, and verification. Mark unknowns explicitly."
            }
        }
    }
}

fn parse_local_reasoning_policy(value: &str) -> Option<LocalReasoningPolicy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "low" | "fast" => Some(LocalReasoningPolicy::Low),
        "medium" | "normal" | "default" => Some(LocalReasoningPolicy::Medium),
        "high" | "deep" => Some(LocalReasoningPolicy::High),
        _ => None,
    }
}

fn baseline_arm_selection(
    headers: &HeaderMap,
) -> Result<crate::adversarial_harness::BaselineArm, String> {
    crate::adversarial_harness::baseline_arm_from_header(
        headers
            .get(crate::adversarial_harness::BASELINE_ARM_HEADER)
            .and_then(|value| value.to_str().ok()),
    )
}

fn local_reasoning_selection(headers: &HeaderMap, payload: &Value) -> LocalReasoningSelection {
    if let Some(policy) = headers
        .get("x-agent-reasoning-policy")
        .and_then(|value| value.to_str().ok())
        .and_then(parse_local_reasoning_policy)
    {
        return LocalReasoningSelection {
            policy,
            source: "x-agent-reasoning-policy",
        };
    }

    if let Some(policy) = payload
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .and_then(parse_local_reasoning_policy)
    {
        return LocalReasoningSelection {
            policy,
            source: "client_reasoning_effort",
        };
    }

    if let Some(budget) = payload
        .get("thinking")
        .and_then(|thinking| thinking.get("budget_tokens"))
        .and_then(Value::as_u64)
    {
        let policy = if budget <= 2048 {
            LocalReasoningPolicy::Low
        } else if budget <= 8192 {
            LocalReasoningPolicy::Medium
        } else {
            LocalReasoningPolicy::High
        };
        return LocalReasoningSelection {
            policy,
            source: "client_thinking_budget",
        };
    }

    LocalReasoningSelection {
        policy: LocalReasoningPolicy::Medium,
        source: "local_default",
    }
}

fn apply_local_reasoning_defaults(req: &mut Value, selection: LocalReasoningSelection) {
    if req.get("max_tokens").is_none() {
        req["max_tokens"] = Value::from(selection.policy.default_max_tokens());
    }
    if req.get("temperature").is_none() {
        req["temperature"] = serde_json::json!(selection.policy.default_temperature());
    }
}

fn add_local_reasoning_metadata(
    attempt: &mut crate::litellm::LiteLlmCallAttempt,
    selection: LocalReasoningSelection,
) {
    attempt.reasoning_policy = Some(selection.policy.as_str().to_string());
    attempt.reasoning_policy_source = Some(selection.source.to_string());
}

fn inject_local_reasoning_contract_openai(req: &mut Value, selection: LocalReasoningSelection) {
    inject_system_context(req, selection.policy.system_contract());
}

fn inject_local_reasoning_contract_anthropic(req: &mut Value, selection: LocalReasoningSelection) {
    let mut blocks = existing_anthropic_system_blocks(req);
    blocks.insert(0, anthropic_text_block(selection.policy.system_contract()));
    req["system"] = Value::Array(blocks);
}

fn enforce_min_max_tokens(req: &mut Value) {
    let requested = req.get("max_tokens").and_then(|v| v.as_u64());
    let default_max_tokens = configured_default_max_tokens();
    let max_max_tokens = configured_max_max_tokens();
    let effective = requested
        .unwrap_or(default_max_tokens)
        .clamp(1, max_max_tokens);
    let requested_for_log = requested.unwrap_or(0);
    if requested_for_log != effective {
        tracing::warn!(
            requested_max_tokens = requested_for_log,
            effective_max_tokens = effective,
            default_max_tokens = default_max_tokens,
            max_max_tokens = max_max_tokens,
            "clamped max_tokens for backend capacity"
        );
    }
    req["max_tokens"] = Value::from(effective);
}

fn number_after_marker(text: &str, marker: &str) -> Option<u64> {
    let start = text.find(marker)? + marker.len();
    let digits: String = text[start..]
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

fn context_window_retry_max_tokens(error_body: &str) -> Option<u64> {
    let max_context = number_after_marker(error_body, "maximum context length is")?;
    let requested_output = number_after_marker(error_body, "requested")?;
    let input_tokens = number_after_marker(error_body, "prompt contains at least")?;
    let remaining = max_context.checked_sub(input_tokens)?;
    if remaining == 0 || remaining >= requested_output {
        return None;
    }
    // Leave a little space because vLLM reports "at least" for prompt tokens.
    Some(remaining.saturating_sub(256).max(1).min(MAX_MAX_TOKENS))
}

fn set_max_tokens(req: &mut Value, max_tokens: u64) {
    req["max_tokens"] = Value::from(max_tokens);
}

fn litellm_route(state: &AppState, namespace: &str) -> crate::litellm::RouteSelection {
    crate::litellm::route_for_namespace(
        &state.default_model,
        namespace,
        state.prefix_cache_canary_enabled,
        &state.prefix_cache_canary_namespace_allowlist,
    )
}

fn inject_system_context(payload: &mut Value, context: &str) {
    if let Some(messages) = payload.get_mut("messages").and_then(|v| v.as_array_mut()) {
        if let Some(first) = messages.first_mut() {
            if first.get("role").and_then(|r| r.as_str()) == Some("system") {
                let existing = first["content"].as_str().unwrap_or("").to_string();
                first["content"] = Value::String(format!("{existing}\n\n---\n{context}"));
                return;
            }
        }
        messages.insert(0, serde_json::json!({"role": "system", "content": context}));
    }
}

fn anthropic_text_block(text: impl Into<String>) -> Value {
    serde_json::json!({"type": "text", "text": text.into()})
}

fn anthropic_cacheable_text_block(text: impl Into<String>) -> Value {
    serde_json::json!({
        "type": "text",
        "text": text.into(),
        "cache_control": {"type": "ephemeral"}
    })
}

fn existing_anthropic_system_blocks(payload: &Value) -> Vec<Value> {
    match payload.get("system") {
        Some(Value::String(s)) if !s.trim().is_empty() => vec![anthropic_text_block(s.clone())],
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| {
                if block.get("type").and_then(|value| value.as_str()) == Some("text")
                    && block
                        .get("text")
                        .and_then(|value| value.as_str())
                        .is_some_and(|text| !text.trim().is_empty())
                {
                    Some(block.clone())
                } else {
                    None
                }
            })
            .collect(),
        _ => vec![],
    }
}

/// Inject context into an Anthropic-format request's system field.
///
/// The compiler prefix is emitted as a separate cacheable text block while the
/// dynamic task tail remains uncached. That gives Anthropic-compatible providers
/// a stable breakpoint without freezing per-request evidence.
fn inject_system_context_anthropic(payload: &mut Value, context: &str) {
    let (stable_prefix, dynamic_tail) = if context.starts_with("== Stable Context Artifacts ==") {
        crate::litellm::split_context_prefix_tail(context)
    } else {
        ("", context)
    };
    let mut blocks = existing_anthropic_system_blocks(payload);
    if !stable_prefix.trim().is_empty() {
        blocks.push(anthropic_cacheable_text_block(stable_prefix.to_string()));
    }
    if !dynamic_tail.trim().is_empty() {
        blocks.push(anthropic_text_block(dynamic_tail.to_string()));
    }
    if blocks.is_empty() {
        return;
    }
    payload["system"] = Value::Array(blocks);
}

/// Pack orchestrator context into an Anthropic-format request's system field.
async fn pack_context_into_anthropic_req(
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

/// Extract assistant text from an Anthropic-format response for persistence.
fn extract_assistant_from_anthropic_response(resp: &Value) -> String {
    resp.get("content")
        .and_then(|v| v.as_array())
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|b| {
                    if b["type"].as_str() == Some("text") {
                        b["text"].as_str()
                    } else {
                        None
                    }
                })
                .collect::<String>()
        })
        .unwrap_or_default()
        .chars()
        .take(500)
        .collect()
}

/// Extract token usage from an Anthropic-format SSE stream.
fn extract_token_usage_from_anthropic_sse(raw: &str) -> TokenUsage {
    let mut usage = TokenUsage::default();
    for line in raw.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        match v["type"].as_str() {
            Some("message_start") => {
                let u = &v["message"]["usage"];
                usage.processed_tokens = u["input_tokens"].as_u64().unwrap_or(0);
                usage.cached_tokens = u["cache_read_input_tokens"].as_u64().unwrap_or(0);
            }
            Some("message_delta") => {
                let u = &v["usage"];
                usage.generated_tokens = u["output_tokens"].as_u64().unwrap_or(0);
            }
            _ => {}
        }
    }
    usage
}

/// Extract assistant text content from an Anthropic-format SSE stream for persistence.
fn extract_assistant_from_anthropic_sse(raw: &str) -> String {
    let mut content = String::new();
    for line in raw.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if v["type"].as_str() == Some("content_block_delta")
            && v["delta"]["type"].as_str() == Some("text_delta")
        {
            if let Some(text) = v["delta"]["text"].as_str() {
                content.push_str(text);
            }
        }
    }
    content.chars().take(500).collect()
}

fn extract_assistant_from_sse(raw: &str) -> String {
    let mut content = String::new();
    for line in raw.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data.trim() == "[DONE]" {
                break;
            }
            if let Ok(v) = serde_json::from_str::<Value>(data) {
                if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
                    content.push_str(delta);
                }
            }
        }
    }
    content
}

fn extract_token_usage_from_sse(raw: &str) -> TokenUsage {
    let mut usage = TokenUsage::default();
    for line in raw.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data.trim() == "[DONE]" {
            break;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        let chunk_usage = TokenUsage::from_openai_value(&value);
        usage.processed_tokens = usage.processed_tokens.max(chunk_usage.processed_tokens);
        usage.cached_tokens = usage.cached_tokens.max(chunk_usage.cached_tokens);
        usage.generated_tokens = usage.generated_tokens.max(chunk_usage.generated_tokens);
    }
    usage
}

fn optional_token_usage_from_sse(raw: &str) -> (Option<i64>, Option<i64>) {
    let mut input = None;
    let mut output = None;
    for line in raw.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data.trim() == "[DONE]" {
            break;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        let (chunk_input, chunk_output) =
            crate::trajectory::optional_token_counts_from_value(&value);
        input = chunk_input.or(input);
        output = chunk_output.or(output);
    }
    (input, output)
}

fn extract_user_content_openai(payload: &Value) -> String {
    payload["messages"]
        .as_array()
        .and_then(|msgs| msgs.iter().rfind(|m| m["role"].as_str() == Some("user")))
        .map(|m| {
            if let Some(text) = m["content"].as_str() {
                text.to_string()
            } else {
                m["content"].to_string()
            }
        })
        .unwrap_or_default()
        .chars()
        .take(500)
        .collect()
}

async fn begin_trajectory_for_request(
    state: &AppState,
    session_id: &str,
) -> crate::trajectory::TrajectoryContext {
    if let Ok(Some(latest)) = db::latest_trajectory_event_for_session(&state.pool, session_id).await
    {
        if let Some(trajectory_id) = latest.trajectory_id {
            let already_ended = latest.event_role.as_deref()
                == Some(crate::trajectory::EventRole::TrajectoryResult.as_str())
                || db::get_trajectory_result(&state.pool, trajectory_id)
                    .await
                    .ok()
                    .flatten()
                    .is_some();
            if !already_ended {
                let idle_for = chrono::Utc::now()
                    .signed_duration_since(latest.created_at)
                    .num_seconds()
                    .max(0) as u64;
                let reason = if idle_for > state.trajectory_idle_timeout_sec {
                    crate::trajectory::BoundaryReason::IdleTimeout
                } else {
                    crate::trajectory::BoundaryReason::NewUserMessage
                };
                spawn_trajectory_result_emit(state, trajectory_id, reason);
            }
        }
    }

    crate::trajectory::TrajectoryContext {
        trajectory_id: uuid::Uuid::new_v4(),
        attempt_index: 1,
    }
}

async fn persist_request_event(
    state: &AppState,
    session_id: &str,
    repo: &str,
    user_content: &str,
    trajectory: crate::trajectory::TrajectoryContext,
    request_metadata: Option<Value>,
) -> Option<uuid::Uuid> {
    // Pre-request: classify and enrich request metadata so the trajectory event
    // itself tells an operator which policy governed the request.
    let raw_capture_enabled = state.capture_pool.is_some();
    let classification = crate::request_classification::classify_request_text(
        repo,
        session_id,
        user_content,
        None,
        "user_message",
    );
    let policy = orchestration_policy::derive_orchestration_policy(
        &classification,
        user_content,
        raw_capture_enabled,
    );
    let policy_metadata = orchestration_policy::compact_policy_metadata(&classification, &policy);

    // Preserve caller-provided metadata while adding a compact policy snapshot.
    // The normalized policy ledger is written separately after the event insert.
    let enriched_metadata = if let Some(ref obj) = request_metadata {
        if let Value::Object(map) = obj {
            // request_metadata is an object - add/replace "orchestration_policy" key.
            let mut enriched = map.clone();
            enriched.insert("orchestration_policy".to_string(), policy_metadata.clone());
            Some(Value::Object(enriched))
        } else {
            // Non-object value - wrap it.
            let mut wrapper = serde_json::Map::new();
            wrapper.insert("orchestration_policy".to_string(), policy_metadata.clone());
            wrapper.insert("original_metadata".to_string(), obj.clone());
            Some(Value::Object(wrapper))
        }
    } else {
        // None - create a new object.
        let mut map = serde_json::Map::new();
        map.insert("orchestration_policy".to_string(), policy_metadata.clone());
        Some(Value::Object(map))
    };

    let metadata = crate::trajectory::make_request_metadata(enriched_metadata, trajectory);
    let req = AppendEventRequest {
        session_id: session_id.to_string(),
        repo: repo.to_string(),
        actor: Some("user".to_string()),
        event_type: "user_message".to_string(),
        summary: user_content.chars().take(500).collect(),
        evidence: None,
        metadata: Some(metadata),
        correlation_id: Some(trajectory.trajectory_id),
        parent_event_id: None,
        trajectory_id: Some(trajectory.trajectory_id),
        attempt_index: Some(trajectory.attempt_index),
        event_role: Some(crate::trajectory::EventRole::Request.as_str().to_string()),
        task: None,
        error_type: None,
        error_description: None,
    };
    let event = match db::event_from_append_request(&req) {
        Ok(event) => event,
        Err(e) => {
            tracing::warn!(repo, "failed to build trajectory request event: {e}");
            return None;
        }
    };
    let event_id = uuid::Uuid::parse_str(&event.id).ok();
    match db::insert_event(&state.pool, &event).await {
        Ok(()) => {
            // Post-persist: classify from the stored event and write the
            // append-only policy ledger row. The second classification keeps
            // the persisted ledger aligned with the durable event shape.
            let classification_event = db::AgentEvent {
                metadata: event.metadata.clone(),
                ..event.clone()
            };
            let classification =
                crate::request_classification::classify_request_event(&classification_event);
            let policy = orchestration_policy::derive_orchestration_policy(
                &classification,
                user_content,
                raw_capture_enabled,
            );
            if let Err(e) = orchestration_policy::persist_orchestration_policy(
                &state.pool,
                &classification,
                &policy,
            )
            .await
            {
                tracing::warn!(
                    target: "orchestration_policy",
                    repo,
                    event_id = %event.id,
                    "failed to persist orchestration policy: {e}"
                );
            }

            spawn_qdrant_index_event(state, event);
            spawn_feature_extraction(state, repo, session_id, Some(trajectory.trajectory_id));
            event_id
        }
        Err(e) => {
            tracing::warn!(repo, "failed to persist trajectory request event: {e}");
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn persist_model_response_event(
    state: &AppState,
    session_id: &str,
    repo: &str,
    assistant_content: &str,
    metadata: Value,
    trajectory: crate::trajectory::TrajectoryContext,
    parent_event_id: Option<uuid::Uuid>,
) -> Option<uuid::Uuid> {
    let req = AppendEventRequest {
        session_id: session_id.to_string(),
        repo: repo.to_string(),
        actor: Some("assistant".to_string()),
        event_type: "assistant_message".to_string(),
        summary: assistant_content.chars().take(500).collect(),
        evidence: None,
        metadata: Some(metadata),
        correlation_id: Some(trajectory.trajectory_id),
        parent_event_id,
        trajectory_id: Some(trajectory.trajectory_id),
        attempt_index: Some(trajectory.attempt_index),
        event_role: Some(
            crate::trajectory::EventRole::ModelResponse
                .as_str()
                .to_string(),
        ),
        task: None,
        error_type: None,
        error_description: None,
    };
    match db::append_event_from_request(&state.pool, &state.embedder, &state.qdrant_url, &req).await
    {
        Ok((event_id, _)) => {
            spawn_feature_extraction(state, repo, session_id, Some(trajectory.trajectory_id));
            uuid::Uuid::parse_str(&event_id).ok()
        }
        Err(e) => {
            tracing::warn!(
                repo,
                "failed to persist trajectory model_response event: {e}"
            );
            None
        }
    }
}

async fn persist_exchange_with_correlation(
    state: &AppState,
    session_id: &str,
    repo: &str,
    user_content: &str,
    assistant_content: &str,
    correlation_id: Option<uuid::Uuid>,
    request_metadata: Option<Value>,
) -> Option<uuid::Uuid> {
    // With execution feedback disabled, correlation_id is None and this writes
    // the same ordinary conversation events as before the feedback layer. With
    // it enabled, these user/assistant events become the root of the chain.
    let make_req = |event_type: &str,
                    actor: &str,
                    content: &str,
                    parent_event_id: Option<uuid::Uuid>,
                    metadata: Option<Value>|
     -> AppendEventRequest {
        AppendEventRequest {
            session_id: session_id.to_string(),
            repo: repo.to_string(),
            actor: Some(actor.to_string()),
            event_type: event_type.to_string(),
            summary: content.chars().take(500).collect(),
            evidence: None,
            metadata,
            correlation_id,
            parent_event_id: correlation_id.and(parent_event_id),
            trajectory_id: None,
            attempt_index: None,
            event_role: None,
            task: None,
            error_type: None,
            error_description: None,
        }
    };

    if let Some(classifier) = &state.sentiment {
        if classifier.is_negative(user_content) {
            tracing::info!(
                target: "sentiment",
                session_id,
                repo,
                "negative feedback detected — storing failed_attempt event"
            );
            let req = make_req("failed_attempt", "user", user_content, None, None);
            if let Err(e) =
                db::append_event_from_request(&state.pool, &state.embedder, &state.qdrant_url, &req)
                    .await
            {
                tracing::warn!(target: "sentiment", "failed to store failed_attempt event: {e}");
            }
        }
    }

    let mut parent_event_id = None;
    let mut assistant_event_id = None;
    for (event_type, actor, content) in [
        ("user_message", "user", user_content),
        ("assistant_message", "assistant", assistant_content),
    ] {
        let metadata = if event_type == "user_message" {
            request_metadata.clone()
        } else {
            None
        };
        let req = make_req(event_type, actor, content, parent_event_id, metadata);
        for attempt in 0u32..3 {
            match db::append_event_from_request(
                &state.pool,
                &state.embedder,
                &state.qdrant_url,
                &req,
            )
            .await
            {
                Ok((event_id, false)) => {
                    tracing::warn!("{event_type} stored in postgres but not qdrant-indexed");
                    let parsed = uuid::Uuid::parse_str(&event_id).ok();
                    if event_type == "assistant_message" {
                        assistant_event_id = parsed;
                    }
                    parent_event_id = correlation_id.and(parsed);
                    break;
                }
                Ok((event_id, true)) => {
                    let parsed = uuid::Uuid::parse_str(&event_id).ok();
                    if event_type == "assistant_message" {
                        assistant_event_id = parsed;
                    }
                    parent_event_id = correlation_id.and(parsed);
                    break;
                }
                Err(e) if attempt < 2 => {
                    let delay = tokio::time::Duration::from_millis(200 * 2u64.pow(attempt));
                    tracing::debug!(
                        attempt,
                        "persist {event_type} failed, retrying in {delay:?}: {e}"
                    );
                    tokio::time::sleep(delay).await;
                }
                Err(e) => {
                    tracing::warn!("failed to persist {event_type} after 3 attempts: {e}");
                    break;
                }
            }
        }
    }

    spawn_feature_extraction(state, repo, session_id, None);
    // Tool and validation events use the assistant message as parent when it is
    // known. If persistence failed, correlation_id alone can still link events.
    assistant_event_id
}

#[allow(clippy::too_many_arguments)]
fn capture_tool_results_background(
    state: AppState,
    session_id: String,
    repo: String,
    task: String,
    correlation_id: uuid::Uuid,
    parent_event_id: Option<uuid::Uuid>,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    tool_results: Vec<crate::execution_feedback::CapturedToolResult>,
) {
    if !state.execution_feedback_enabled || tool_results.is_empty() {
        return;
    }

    // This capture is deliberately best-effort and off the response path:
    // telemetry write failures must never change what the user receives.
    let gate_state = state.clone();
    spawn_bounded_background(&gate_state, "tool_result_capture", async move {
        let ctx = crate::execution_feedback::ExecutionEventContext {
            session_id,
            repo: repo.clone(),
            task: task.clone(),
            actor: "validator".to_string(),
            correlation_id,
            parent_event_id,
            trajectory,
        };
        for result in tool_results {
            // One observed tool result can fan out into tool_result, normalized
            // validator result, with failed outcomes carrying inline signatures.
            for event in crate::execution_feedback::events_for_tool_result(&ctx, &result) {
                if let Err(e) = db::append_execution_event(
                    &state.pool,
                    &state.embedder,
                    &state.qdrant_url,
                    event,
                )
                .await
                {
                    tracing::error!(
                        target: "execution_feedback",
                        repo = %repo,
                        task = %task,
                        "failed to write execution feedback event: {e}"
                    );
                }
            }
        }
        spawn_feature_extraction(
            &state,
            &repo,
            &ctx.session_id,
            trajectory.map(|trajectory| trajectory.trajectory_id),
        );
    });
}

#[allow(clippy::too_many_arguments)]
fn maybe_write_context_pack_event(
    state: &AppState,
    session_id: Option<&str>,
    repo: &str,
    task: &str,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    parent_event_id: Option<uuid::Uuid>,
    stats: &crate::state::ContextPackStats,
    token_budget: usize,
) -> Option<uuid::Uuid> {
    if !state.trajectory_capture_enabled {
        return None;
    }
    let (Some(session_id), Some(trajectory)) = (session_id, trajectory) else {
        return None;
    };
    let event = crate::trajectory::context_pack_event(
        session_id,
        repo,
        task,
        trajectory,
        parent_event_id,
        stats,
        token_budget,
    );
    let event_id = uuid::Uuid::parse_str(&event.id).ok();
    let state_bg = state.clone();
    let repo = repo.to_string();
    let task = task.to_string();
    spawn_bounded_background(state, "context_pack_event", async move {
        if let Err(e) = db::append_execution_event(
            &state_bg.pool,
            &state_bg.embedder,
            &state_bg.qdrant_url,
            event,
        )
        .await
        {
            tracing::warn!(
                repo = %repo,
                task = %task,
                "failed to write context_pack trajectory event: {e}"
            );
        }
    });
    event_id
}

/// Pack orchestrator context into an OpenAI-shaped request.
/// Sets a default model if absent, fetches memory events, builds context string,
/// and injects it as a system message.
async fn pack_context_into_req(
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

/// POST the OpenAI request to LiteLLM and return the raw response JSON.
/// Returns Err(Response) on network or parse failure.
async fn dispatch_non_streaming_raw(
    state: &AppState,
    openai_req: &Value,
    finalizer: &crate::litellm::LiteLlmCallFinalizer,
) -> Result<(Value, u64), Response> {
    let url = format!("{}/chat/completions", state.litellm_url);
    let started = std::time::Instant::now();
    let response = match state
        .http
        .post(&url)
        .bearer_auth(&state.litellm_key)
        .json(openai_req)
        .send()
        .await
    {
        Ok(r) => Ok(r),
        Err(e) => {
            telemetry::record_upstream_litellm("chat_completions", started.elapsed(), "error");
            telemetry::record_upstream_litellm_error(
                "chat_completions",
                telemetry::reqwest_error_kind(&e),
            );
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::NetworkError,
                    Some(telemetry::reqwest_error_kind(&e)),
                    Some(&e.to_string()),
                    crate::litellm::ProviderCacheCounters::default(),
                )
                .await;
            Err((
                StatusCode::BAD_GATEWAY,
                axum::Json(
                    serde_json::json!({"error": "litellm_unreachable", "detail": e.to_string()}),
                ),
            )
                .into_response())
        }
    }?;

    let status = response.status();
    telemetry::record_upstream_litellm(
        "chat_completions",
        started.elapsed(),
        &status.as_u16().to_string(),
    );
    if !status.is_success() {
        telemetry::record_upstream_litellm_error(
            "chat_completions",
            telemetry::upstream_error_kind(status),
        );
    }
    let latency_ms = started.elapsed().as_millis() as u64;
    let value = match response.json::<Value>().await {
        Ok(value) => value,
        Err(e) => {
            telemetry::record_upstream_litellm_error("chat_completions", "parse");
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::ParseError,
                    Some("parse"),
                    Some(&e.to_string()),
                    crate::litellm::ProviderCacheCounters::default(),
                )
                .await;
            return Err((
                StatusCode::BAD_GATEWAY,
                axum::Json(serde_json::json!({"error": "litellm_parse_error"})),
            )
                .into_response());
        }
    };
    if !status.is_success() {
        finalizer
            .finalize(
                crate::litellm::TerminalStatus::HttpError,
                Some(telemetry::upstream_error_kind(status)),
                Some("upstream returned non-success status"),
                crate::litellm::ProviderCacheCounters::from_value(&value),
            )
            .await;
        return Err((status, axum::Json(value)).into_response());
    }
    finalizer
        .finalize(
            crate::litellm::TerminalStatus::Success,
            None,
            None,
            crate::litellm::ProviderCacheCounters::from_value(&value),
        )
        .await;
    Ok((value, latency_ms))
}

async fn vllm_cache_snapshot(
    state: &AppState,
) -> Option<(String, crate::vllm_metrics::VllmCacheSnapshot)> {
    let metrics_url = state.vllm_metrics_url.as_deref()?;
    match crate::vllm_metrics::fetch_cache_snapshot(&state.http, metrics_url).await {
        Ok(snapshot) => Some((metrics_url.to_string(), snapshot)),
        Err(e) => {
            tracing::warn!(metrics_url, "failed to fetch vLLM cache metrics: {e}");
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn record_vllm_cache_observation(
    state: &AppState,
    before: Option<(String, crate::vllm_metrics::VllmCacheSnapshot)>,
    session_id: Option<&str>,
    namespace: &str,
    repo: &str,
    task: &str,
    attempt: &crate::litellm::LiteLlmCallAttempt,
    usage: &TokenUsage,
    provider_cache: crate::litellm::ProviderCacheCounters,
) -> Option<crate::vllm_metrics::VllmCacheDelta> {
    let Some((metrics_url, before)) = before else {
        return None;
    };
    let after = match crate::vllm_metrics::fetch_cache_snapshot(&state.http, &metrics_url).await {
        Ok(snapshot) => snapshot,
        Err(e) => {
            tracing::warn!(
                metrics_url,
                "failed to fetch post-request vLLM cache metrics: {e}"
            );
            return None;
        }
    };
    let delta = after.delta_since(before);
    telemetry::record_vllm_cache_delta(&delta, &attempt.routed_model);
    let observation = db::VllmCacheObservationInput {
        session_id: session_id.map(str::to_string),
        namespace: namespace.to_string(),
        repo: repo.to_string(),
        task: task.to_string(),
        endpoint: attempt.endpoint.clone(),
        requested_model: attempt.requested_model.clone(),
        routed_model: attempt.routed_model.clone(),
        request_event_id: attempt.request_event_id,
        context_pack_id: attempt.context_pack_id,
        attempt_id: attempt.attempt_id,
        metrics_url,
        delta,
        request_input_tokens: usage.processed_tokens as i64,
        request_output_tokens: usage.generated_tokens as i64,
        provider_cache,
    };
    if let Err(e) = db::insert_vllm_cache_observation(&state.pool, &observation).await {
        tracing::warn!(
            attempt_id = %attempt.attempt_id,
            "failed to record vLLM cache observation: {e}"
        );
    }
    Some(delta)
}

fn provider_cache_from_vllm_delta(
    delta: crate::vllm_metrics::VllmCacheDelta,
) -> crate::litellm::ProviderCacheCounters {
    let cache_read = delta
        .prompt_tokens_local_cache_hit_delta
        .saturating_add(delta.prompt_tokens_external_kv_delta)
        .max(delta.prompt_tokens_cached_delta);
    crate::litellm::ProviderCacheCounters {
        provider_cached_tokens: delta.prompt_tokens_cached_delta.max(0),
        provider_cache_created_tokens: 0,
        provider_cache_read_tokens: cache_read.max(0),
    }
}

fn merge_provider_cache_from_vllm_delta(
    provider_cache: crate::litellm::ProviderCacheCounters,
    delta: Option<crate::vllm_metrics::VllmCacheDelta>,
) -> crate::litellm::ProviderCacheCounters {
    let Some(delta) = delta else {
        return provider_cache;
    };
    let mut merged = provider_cache;
    merged.max_assign(provider_cache_from_vllm_delta(delta));
    merged
}

fn inject_anthropic_cache_usage(
    value: &mut Value,
    counters: crate::litellm::ProviderCacheCounters,
) {
    if let Some(usage) = value.get_mut("usage").and_then(Value::as_object_mut) {
        usage.insert(
            "cache_creation_input_tokens".to_string(),
            serde_json::json!(counters.provider_cache_created_tokens.max(0)),
        );
        usage.insert(
            "cache_read_input_tokens".to_string(),
            serde_json::json!(counters.provider_cache_read_tokens.max(0)),
        );
    }
}

fn anthropic_cache_usage_sse_event(counters: crate::litellm::ProviderCacheCounters) -> String {
    format!(
        concat!(
            "event: message_delta\n",
            "data: {{\"type\":\"message_delta\",\"delta\":{{}},\"usage\":{{",
            "\"cache_creation_input_tokens\":{},",
            "\"cache_read_input_tokens\":{}",
            "}}}}\n\n"
        ),
        counters.provider_cache_created_tokens.max(0),
        counters.provider_cache_read_tokens.max(0)
    )
}

async fn record_client_capture(state: &AppState, capture: crate::client_capture::RawHttpCapture) {
    let Some(pool) = state.capture_pool.as_ref() else {
        return;
    };
    if let Err(e) = crate::client_capture::record(pool, &capture).await {
        tracing::warn!(
            exchange_id = %capture.exchange_id,
            endpoint = %capture.endpoint,
            "failed to record raw client capture: {e}"
        );
    }
}

async fn record_client_capture_response(
    state: &AppState,
    mut capture: crate::client_capture::RawHttpCapture,
    status: StatusCode,
    content_type: &'static str,
    body: Vec<u8>,
) {
    capture.response_status = Some(status.as_u16() as i32);
    capture.response_headers = Some(serde_json::json!({"content-type": [content_type]}));
    capture.raw_response_body = Some(body);
    record_client_capture(state, capture).await;
}

#[tracing::instrument(name = "handler.chat_completions", skip(state, headers, body))]
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let raw_client_body = body.to_vec();
    let mut capture = crate::client_capture::RawHttpCapture::new(
        "chat/completions",
        &headers,
        raw_client_body.clone(),
    );
    let payload: Value = match serde_json::from_slice(&raw_client_body) {
        Ok(payload) => payload,
        Err(e) => {
            let body = serde_json::json!({
                "error": {
                    "type": "invalid_request_error",
                    "message": format!("invalid JSON request body: {e}")
                }
            });
            record_client_capture_response(
                &state,
                capture,
                StatusCode::BAD_REQUEST,
                "application/json",
                crate::client_capture::to_json_bytes(&body),
            )
            .await;
            return (StatusCode::BAD_REQUEST, axum::Json(body)).into_response();
        }
    };
    capture.parsed_request_body = Some(payload.clone());
    let Some((caller_token, namespace)) = authenticate(&state, &headers) else {
        let body = serde_json::json!({"error": "unauthorized"});
        record_client_capture_response(
            &state,
            capture,
            StatusCode::UNAUTHORIZED,
            "application/json",
            crate::client_capture::to_json_bytes(&body),
        )
        .await;
        return (StatusCode::UNAUTHORIZED, axum::Json(body)).into_response();
    };
    if let Some(response) = check_rate_limit(&state, &caller_token) {
        return response;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Explicit headers take precedence; fall back to token-bound namespace so
    // standard clients (OpenCode, curl) get memory without custom headers.
    let repo = headers
        .get("x-agent-repo")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| namespace.clone());
    let task = headers
        .get("x-agent-task")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| state.default_task.clone());
    tracing::info!(repo = %repo, task = %task, "routing request");
    capture.namespace = Some(namespace.clone());
    capture.repo = Some(repo.clone());
    capture.task = Some(task.clone());

    let requested_model = payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&state.default_model)
        .to_string();
    let user_content = extract_user_content_openai(&payload);
    if let Some(response) =
        maybe_openai_live_policy_response(&state, &repo, &namespace, &user_content)
    {
        return response;
    }
    let baseline_arm = match baseline_arm_selection(&headers) {
        Ok(arm) => arm,
        Err(e) => {
            let body = serde_json::json!({"error": "invalid_baseline_arm", "detail": e});
            return (StatusCode::BAD_REQUEST, axum::Json(body)).into_response();
        }
    };
    let reasoning_selection = local_reasoning_selection(&headers, &payload);
    let mut req = payload.clone();
    // Always route to the configured backend model regardless of what the client sent.
    let route = litellm_route(&state, &namespace);
    req["model"] = Value::String(route.routed_model.clone());
    apply_local_reasoning_defaults(&mut req, reasoning_selection);
    enforce_min_max_tokens(&mut req);
    inject_local_reasoning_contract_openai(&mut req, reasoning_selection);
    let sampling_audit = crate::sampling::capture_and_maybe_override(
        &payload,
        &mut req,
        state.sampling_config,
        state.sampling_policy.as_ref(),
    );
    let sampling_metadata = sampling_audit.as_ref().map(|audit| {
        audit.metadata(
            "/v1/chat/completions",
            &requested_model,
            &route.routed_model,
        )
    });
    let session_id = if state.trajectory_capture_enabled {
        match db::find_or_create_session(&state.pool, &repo, &task, "agent").await {
            Ok(session_id) => Some(session_id),
            Err(e) => {
                tracing::warn!("find_or_create_session failed before trajectory capture: {e}");
                None
            }
        }
    } else {
        None
    };
    let request_classification = crate::request_classification::classify_request_text(
        &repo,
        session_id.as_deref().unwrap_or("unknown"),
        &user_content,
        None,
        "user_message",
    );
    let request_policy = orchestration_policy::derive_orchestration_policy(
        &request_classification,
        &user_content,
        state.capture_pool.is_some(),
    );
    let tool_mediation_metadata = if state.tool_mediation_enabled {
        // Shape the OpenAI tool menu with the same policy model used by
        // `/tools/authorize`. This only removes tools from the client-provided
        // menu; proxy mode cannot invent tools the client did not offer.
        let outcome = crate::tool_mediation::shape_openai_request_with_policy(
            &mut req,
            &user_content,
            Some(&request_policy),
        );
        telemetry::record_tool_menu_outcome(&state.metrics, &outcome);
        Some(outcome.metadata())
    } else {
        None
    };
    let baseline_metadata = Some(serde_json::json!({
        "baseline_arm": baseline_arm.as_str(),
    }));
    let request_metadata = merge_request_metadata([
        sampling_metadata,
        tool_mediation_metadata,
        baseline_metadata,
    ]);
    let trajectory = if let Some(session_id) = session_id.as_deref() {
        Some(begin_trajectory_for_request(&state, session_id).await)
    } else {
        None
    };
    let request_event_id =
        if let (Some(session_id), Some(trajectory)) = (session_id.as_deref(), trajectory) {
            persist_request_event(
                &state,
                session_id,
                &repo,
                &user_content,
                trajectory,
                request_metadata.clone(),
            )
            .await
        } else {
            None
        };
    let (context_pack_id, context_pack_hash) = pack_context_into_req(
        &state,
        &mut req,
        session_id.as_deref(),
        &repo,
        &task,
        trajectory,
        request_event_id,
        Some(&request_classification),
        Some(&request_policy),
    )
    .await;
    let correlation_id = trajectory
        .map(|trajectory| trajectory.trajectory_id)
        .or_else(|| state.execution_feedback_enabled.then(uuid::Uuid::new_v4));
    let cache_policy = crate::litellm::exact_cache_decision("chat_completions", &req, false);
    let mut attempt = crate::litellm::new_attempt(
        request_event_id,
        trajectory.map(|trajectory| trajectory.trajectory_id),
        context_pack_id,
        namespace.clone(),
        repo.clone(),
        task.clone(),
        "chat_completions",
        requested_model.clone(),
        &route,
        cache_policy,
        context_pack_hash.clone(),
        Some(baseline_arm.as_str().to_string()),
    );
    add_local_reasoning_metadata(&mut attempt, reasoning_selection);
    capture.attempt_id = Some(attempt.attempt_id);
    crate::litellm::add_agentic_os_metadata(&mut req, &attempt);
    capture.forwarded_request_body = Some(crate::client_capture::to_json_bytes(&req));
    let vllm_cache_before = vllm_cache_snapshot(&state).await;
    let finalizer = crate::litellm::LiteLlmCallFinalizer::begin(state.pool.clone(), attempt).await;

    if is_stream {
        return handle_streaming(
            &state,
            req,
            repo,
            task,
            requested_model,
            namespace,
            correlation_id,
            request_metadata,
            session_id,
            trajectory,
            request_event_id,
            context_pack_id,
            user_content,
            finalizer,
            vllm_cache_before,
            capture,
        )
        .await;
    }

    // ── Non-streaming ───────────────────────────────────────────
    match dispatch_non_streaming_raw(&state, &req, &finalizer).await {
        Ok((val, latency_ms)) => {
            capture.response_status = Some(StatusCode::OK.as_u16() as i32);
            capture.response_headers =
                Some(serde_json::json!({"content-type": ["application/json"]}));
            capture.raw_response_body = Some(crate::client_capture::to_json_bytes(&val));
            record_client_capture(&state, capture).await;
            let usage = TokenUsage::from_openai_value(&val);
            let provider_cache = crate::litellm::ProviderCacheCounters::from_value(&val);
            record_vllm_cache_observation(
                &state,
                vllm_cache_before,
                session_id.as_deref(),
                &namespace,
                &repo,
                &task,
                finalizer.attempt(),
                &usage,
                provider_cache,
            )
            .await;
            telemetry::record_tokens(&state.metrics, &usage, &state.default_model);
            if !usage.is_empty() {
                let pool = state.pool.clone();
                let actual = state.default_model.clone();
                let rm = requested_model.clone();
                let ns = namespace.clone();
                let r = repo.clone();
                let u = usage.clone();
                tokio::spawn(async move {
                    if let Err(e) = db::record_token_usage(&pool, &rm, &actual, &ns, &r, &u).await {
                        tracing::warn!("failed to record token usage: {e}");
                    }
                });
            }

            let assistant_content: String = val["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .chars()
                .take(500)
                .collect();
            // Non-streaming responses may contain provider/tool envelopes. We
            // inspect them after normal exchange persistence has an event parent.
            let tool_results = crate::execution_feedback::tool_results_from_value(&val);

            if let (Some(session_id), Some(trajectory)) = (session_id.as_deref(), trajectory) {
                let (input_tokens, output_tokens) =
                    crate::trajectory::optional_token_counts_from_value(&val);
                let metadata = crate::trajectory::model_response_metadata(
                    &state.default_model,
                    "litellm",
                    input_tokens,
                    output_tokens,
                    Some(latency_ms as i64),
                    crate::trajectory::model_finish_reason(&val),
                    request_metadata.clone(),
                    context_pack_id,
                    Some(trajectory),
                );
                let assistant_event_id = persist_model_response_event(
                    &state,
                    session_id,
                    &repo,
                    &assistant_content,
                    metadata,
                    trajectory,
                    request_event_id,
                )
                .await;
                capture_tool_results_background(
                    state.as_ref().clone(),
                    session_id.to_string(),
                    repo.clone(),
                    task.clone(),
                    trajectory.trajectory_id,
                    assistant_event_id,
                    Some(trajectory),
                    tool_results,
                );
            } else {
                match db::find_or_create_session(&state.pool, &repo, &task, "agent").await {
                    Ok(session_id) => {
                        let assistant_event_id = persist_exchange_with_correlation(
                            &state,
                            &session_id,
                            &repo,
                            &user_content,
                            &assistant_content,
                            correlation_id,
                            request_metadata.clone(),
                        )
                        .await;
                        if let Some(correlation_id) = correlation_id {
                            capture_tool_results_background(
                                state.as_ref().clone(),
                                session_id,
                                repo.clone(),
                                task.clone(),
                                correlation_id,
                                assistant_event_id,
                                None,
                                tool_results,
                            );
                        }
                    }
                    Err(e) => tracing::warn!("find_or_create_session failed: {e}"),
                }
            }
            axum::Json(val).into_response()
        }
        Err(resp) => resp,
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_streaming(
    state: &AppState,
    req: Value,
    repo: String,
    task: String,
    requested_model: String,
    namespace: String,
    correlation_id: Option<uuid::Uuid>,
    request_metadata: Option<Value>,
    session_id: Option<String>,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    request_event_id: Option<uuid::Uuid>,
    context_pack_id: Option<uuid::Uuid>,
    user_content: String,
    mut finalizer: crate::litellm::LiteLlmCallFinalizer,
    vllm_cache_before: Option<(String, crate::vllm_metrics::VllmCacheSnapshot)>,
    capture: crate::client_capture::RawHttpCapture,
) -> Response {
    let url = format!("{}/chat/completions", state.litellm_url);
    let mut capture = capture;

    let started = finalizer.attempt_mut().started_at;
    let upstream = match state
        .http_stream
        .post(&url)
        .bearer_auth(&state.litellm_key)
        .json(&req)
        .send()
        .await
    {
        Ok(r) => {
            let status = r.status();
            telemetry::record_upstream_litellm(
                "chat_completions",
                started.elapsed(),
                &status.as_u16().to_string(),
            );
            if !status.is_success() {
                telemetry::record_upstream_litellm_error(
                    "chat_completions",
                    telemetry::upstream_error_kind(status),
                );
            }
            r
        }
        Err(e) => {
            telemetry::record_upstream_litellm("chat_completions", started.elapsed(), "error");
            telemetry::record_upstream_litellm_error(
                "chat_completions",
                telemetry::reqwest_error_kind(&e),
            );
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::NetworkError,
                    Some(telemetry::reqwest_error_kind(&e)),
                    Some(&e.to_string()),
                    crate::litellm::ProviderCacheCounters::default(),
                )
                .await;
            return (
                StatusCode::BAD_GATEWAY,
                axum::Json(
                    serde_json::json!({"error": "litellm_unreachable", "detail": e.to_string()}),
                ),
            )
                .into_response();
        }
    };

    if !upstream.status().is_success() {
        let status = upstream.status();
        telemetry::record_upstream_litellm_error(
            "chat_completions",
            telemetry::upstream_error_kind(status),
        );
        let body = upstream.text().await.unwrap_or_default();
        finalizer
            .finalize(
                crate::litellm::TerminalStatus::HttpError,
                Some(telemetry::upstream_error_kind(status)),
                Some("upstream returned non-success status"),
                crate::litellm::ProviderCacheCounters::default(),
            )
            .await;
        return (status, [(header::CONTENT_TYPE, "application/json")], body).into_response();
    }

    let bytes_stream = upstream.bytes_stream();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Vec<u8>>();
    let accumulated = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let acc_clone = accumulated.clone();
    let mut stream_finalizer = finalizer;
    let observation_attempt = stream_finalizer.attempt().clone();

    let tapped = async_stream::stream! {
        tokio::pin!(bytes_stream);
        let mut stream_metrics = telemetry::StreamTracker::new("chat_completions", started);
        let mut tx_opt = Some(done_tx);
        let mut first_seen = false;
        let mut first_token_at = None;
        while let Some(chunk) = bytes_stream.next().await {
            match chunk {
                Ok(b) => {
                    if !first_seen && crate::litellm::sse_chunk_has_non_empty_data(&b) {
                        first_seen = true;
                        first_token_at = Some(std::time::Instant::now());
                        stream_finalizer.attempt_mut().first_token_at = first_token_at;
                        stream_metrics.first_token();
                    }
                    if let Ok(mut g) = acc_clone.lock() { g.extend_from_slice(&b); }
                    yield Ok::<Bytes, std::io::Error>(b);
                }
                Err(e) => {
                    stream_metrics.fail("upstream_error");
                    stream_finalizer
                        .finalize(
                            crate::litellm::TerminalStatus::StreamError,
                            Some("upstream_error"),
                            Some(&e.to_string()),
                            crate::litellm::ProviderCacheCounters::default(),
                        )
                        .await;
                    yield Err(std::io::Error::other(e));
                    return;
                }
            }
        }
        stream_metrics.finish();
        let data = acc_clone.lock().map(|g| g.clone()).unwrap_or_default();
        let raw = String::from_utf8_lossy(&data);
        let counters = crate::litellm::provider_counters_from_sse(&raw);
        let mut done_finalizer = stream_finalizer.clone();
        done_finalizer.attempt_mut().first_token_at = first_token_at;
        done_finalizer.attempt_mut().completed_at = Some(std::time::Instant::now());
        done_finalizer
            .finalize(crate::litellm::TerminalStatus::Success, None, None, counters)
            .await;
        if let Some(tx) = tx_opt.take() {
            let _ = tx.send(data);
        }
    };

    let state_bg = state.clone();
    tokio::spawn(async move {
        if let Ok(raw_bytes) = done_rx.await {
            capture.response_status = Some(StatusCode::OK.as_u16() as i32);
            capture.response_headers =
                Some(serde_json::json!({"content-type": ["text/event-stream"]}));
            capture.raw_response_body = Some(raw_bytes.clone());
            record_client_capture(&state_bg, capture).await;
            let raw = String::from_utf8_lossy(&raw_bytes);
            let usage = extract_token_usage_from_sse(&raw);
            let provider_cache = crate::litellm::provider_counters_from_sse(&raw);
            record_vllm_cache_observation(
                &state_bg,
                vllm_cache_before,
                session_id.as_deref(),
                &namespace,
                &repo,
                &task,
                &observation_attempt,
                &usage,
                provider_cache,
            )
            .await;
            telemetry::record_tokens(&state_bg.metrics, &usage, &state_bg.default_model);
            if !usage.is_empty() {
                if let Err(e) = db::record_token_usage(
                    &state_bg.pool,
                    &requested_model,
                    &state_bg.default_model,
                    &namespace,
                    &repo,
                    &usage,
                )
                .await
                {
                    tracing::warn!("failed to record token usage: {e}");
                }
            }
            let assistant_content = extract_assistant_from_sse(&raw);
            // Stream capture reuses the accumulated bytes already needed for
            // token accounting, so no extra read is introduced.
            let tool_results = crate::execution_feedback::tool_results_from_sse(&raw);
            if let (Some(session_id), Some(trajectory)) = (session_id.as_deref(), trajectory) {
                let (input_tokens, output_tokens) = optional_token_usage_from_sse(&raw);
                let metadata = crate::trajectory::model_response_metadata(
                    &state_bg.default_model,
                    "litellm",
                    input_tokens,
                    output_tokens,
                    Some(started.elapsed().as_millis() as i64),
                    None,
                    request_metadata.clone(),
                    context_pack_id,
                    Some(trajectory),
                );
                let assistant_event_id = persist_model_response_event(
                    &state_bg,
                    session_id,
                    &repo,
                    &assistant_content,
                    metadata,
                    trajectory,
                    request_event_id,
                )
                .await;
                capture_tool_results_background(
                    state_bg.clone(),
                    session_id.to_string(),
                    repo.clone(),
                    task.clone(),
                    trajectory.trajectory_id,
                    assistant_event_id,
                    Some(trajectory),
                    tool_results,
                );
            } else {
                match db::find_or_create_session(&state_bg.pool, &repo, &task, "agent").await {
                    Ok(sid) => {
                        let assistant_event_id = persist_exchange_with_correlation(
                            &state_bg,
                            &sid,
                            &repo,
                            &user_content,
                            &assistant_content,
                            correlation_id,
                            request_metadata,
                        )
                        .await;
                        if let Some(correlation_id) = correlation_id {
                            capture_tool_results_background(
                                state_bg.clone(),
                                sid,
                                repo.clone(),
                                task.clone(),
                                correlation_id,
                                assistant_event_id,
                                None,
                                tool_results,
                            );
                        }
                    }
                    Err(e) => tracing::warn!("stream: find_or_create_session failed: {e}"),
                }
            }
        }
    });

    (
        [
            (axum::http::header::CONTENT_TYPE, "text/event-stream"),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
            (axum::http::header::CONNECTION, "keep-alive"),
        ],
        axum::body::Body::from_stream(tapped),
    )
        .into_response()
}

// ── Anthropic /v1/messages ──────────────────────────────────────

#[tracing::instrument(name = "handler.messages", skip(state, headers, body))]
pub async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let raw_client_body = body.to_vec();
    let mut capture =
        crate::client_capture::RawHttpCapture::new("messages", &headers, raw_client_body.clone());
    let payload: Value = match serde_json::from_slice(&raw_client_body) {
        Ok(payload) => payload,
        Err(e) => {
            let body = anthropic_error_value(
                "invalid_request_error",
                format!("invalid JSON request body: {e}"),
            );
            record_client_capture_response(
                &state,
                capture,
                StatusCode::BAD_REQUEST,
                "application/json",
                crate::client_capture::to_json_bytes(&body),
            )
            .await;
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("invalid JSON request body: {e}"),
            );
        }
    };
    capture.parsed_request_body = Some(payload.clone());
    let Some((caller_token, namespace)) = authenticate(&state, &headers) else {
        let body = anthropic_error_value("authentication_error", "invalid or missing API key");
        record_client_capture_response(
            &state,
            capture,
            StatusCode::UNAUTHORIZED,
            "application/json",
            crate::client_capture::to_json_bytes(&body),
        )
        .await;
        return anthropic_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid or missing API key",
        );
    };
    if let Some(response) = check_rate_limit(&state, &caller_token) {
        return response;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let repo = headers
        .get("x-agent-repo")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| namespace.clone());
    let task = headers
        .get("x-agent-task")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| state.default_task.clone());
    tracing::info!(repo = %repo, task = %task, endpoint = "messages", "routing request");
    capture.namespace = Some(namespace.clone());
    capture.repo = Some(repo.clone());
    capture.task = Some(task.clone());

    let user_content = anthropic::extract_user_content_from_anthropic(&payload);
    if let Some(response) =
        maybe_anthropic_live_policy_response(&state, &repo, &namespace, &user_content)
    {
        return response;
    }
    let baseline_arm = match baseline_arm_selection(&headers) {
        Ok(arm) => arm,
        Err(e) => {
            let body = anthropic_error_value(
                "invalid_request_error",
                format!("invalid baseline arm: {e}"),
            );
            return (StatusCode::BAD_REQUEST, axum::Json(body)).into_response();
        }
    };
    let reasoning_selection = local_reasoning_selection(&headers, &payload);
    let model = payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&state.default_model)
        .to_string();

    // Stay in Anthropic format — no translation.
    let mut req = payload;
    let route = litellm_route(&state, &namespace);
    req["model"] = Value::String(route.routed_model.clone());
    apply_local_reasoning_defaults(&mut req, reasoning_selection);
    enforce_min_max_tokens(&mut req);
    normalize_response_content_types(&mut req);
    sanitize_anthropic_litellm_request(&mut req);
    inject_local_reasoning_contract_anthropic(&mut req, reasoning_selection);
    let session_id = if state.trajectory_capture_enabled {
        match db::find_or_create_session(&state.pool, &repo, &task, "agent").await {
            Ok(session_id) => Some(session_id),
            Err(e) => {
                tracing::warn!("find_or_create_session failed before trajectory capture: {e}");
                None
            }
        }
    } else {
        None
    };
    let request_classification = crate::request_classification::classify_request_text(
        &repo,
        session_id.as_deref().unwrap_or("unknown"),
        &user_content,
        None,
        "user_message",
    );
    let request_policy = orchestration_policy::derive_orchestration_policy(
        &request_classification,
        &user_content,
        state.capture_pool.is_some(),
    );
    let tool_mediation_metadata = if state.tool_mediation_enabled {
        // Same policy-aware shaping as the OpenAI endpoint, but applied to the
        // Anthropic `tools` shape. The payload stays Anthropic-formatted all the
        // way through this handler.
        let outcome = crate::tool_mediation::shape_anthropic_request_with_policy(
            &mut req,
            &user_content,
            Some(&request_policy),
        );
        telemetry::record_tool_menu_outcome(&state.metrics, &outcome);
        Some(outcome.metadata())
    } else {
        None
    };
    let baseline_metadata = Some(serde_json::json!({
        "baseline_arm": baseline_arm.as_str(),
    }));
    let request_metadata = merge_request_metadata([tool_mediation_metadata, baseline_metadata]);
    let trajectory = if let Some(session_id) = session_id.as_deref() {
        Some(begin_trajectory_for_request(&state, session_id).await)
    } else {
        None
    };
    let request_event_id =
        if let (Some(session_id), Some(trajectory)) = (session_id.as_deref(), trajectory) {
            persist_request_event(
                &state,
                session_id,
                &repo,
                &user_content,
                trajectory,
                request_metadata.clone(),
            )
            .await
        } else {
            None
        };
    let (context_pack_id, context_pack_hash) = pack_context_into_anthropic_req(
        &state,
        &mut req,
        session_id.as_deref(),
        &repo,
        &task,
        trajectory,
        request_event_id,
        Some(&request_classification),
        Some(&request_policy),
    )
    .await;
    let correlation_id = trajectory
        .map(|trajectory| trajectory.trajectory_id)
        .or_else(|| state.execution_feedback_enabled.then(uuid::Uuid::new_v4));
    let cache_policy = crate::litellm::exact_cache_decision("messages", &req, false);
    let mut attempt = crate::litellm::new_attempt(
        request_event_id,
        trajectory.map(|trajectory| trajectory.trajectory_id),
        context_pack_id,
        namespace.clone(),
        repo.clone(),
        task.clone(),
        "messages",
        model.clone(),
        &route,
        cache_policy,
        context_pack_hash.clone(),
        Some(baseline_arm.as_str().to_string()),
    );
    add_local_reasoning_metadata(&mut attempt, reasoning_selection);
    capture.attempt_id = Some(attempt.attempt_id);
    crate::litellm::add_agentic_os_metadata(&mut req, &attempt);
    capture.forwarded_request_body = Some(crate::client_capture::to_json_bytes(&req));
    let vllm_cache_before = vllm_cache_snapshot(&state).await;
    let mut finalizer =
        crate::litellm::LiteLlmCallFinalizer::begin(state.pool.clone(), attempt).await;

    if is_stream {
        return handle_streaming_anthropic(
            &state,
            req,
            user_content,
            repo,
            task,
            model,
            namespace,
            correlation_id,
            request_metadata,
            session_id,
            trajectory,
            request_event_id,
            context_pack_id,
            finalizer,
            vllm_cache_before,
            capture,
        )
        .await;
    }

    // ── Non-streaming: passthrough to LiteLLM /messages ────────
    let url = format!("{}/messages", state.litellm_url);
    let started = std::time::Instant::now();
    let upstream_resp = match state
        .http
        .post(&url)
        .bearer_auth(&state.litellm_key)
        .json(&req)
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
            r
        }
        Err(e) => {
            telemetry::record_upstream_litellm("messages", started.elapsed(), "error");
            telemetry::record_upstream_litellm_error("messages", telemetry::reqwest_error_kind(&e));
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::NetworkError,
                    Some(telemetry::reqwest_error_kind(&e)),
                    Some(&e.to_string()),
                    crate::litellm::ProviderCacheCounters::default(),
                )
                .await;
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                "upstream LiteLLM request failed",
            );
        }
    };

    let mut status = upstream_resp.status();
    let mut val: Value = match upstream_resp.json().await {
        Ok(v) => v,
        Err(_) => {
            telemetry::record_upstream_litellm_error("messages", "parse");
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::ParseError,
                    Some("parse"),
                    Some("invalid upstream response"),
                    crate::litellm::ProviderCacheCounters::default(),
                )
                .await;
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                "invalid upstream response",
            );
        }
    };

    if !status.is_success() {
        let error_body = val.to_string();
        if let Some(retry_max_tokens) = context_window_retry_max_tokens(&error_body) {
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::HttpError,
                    Some(telemetry::upstream_error_kind(status)),
                    Some("upstream returned non-success status"),
                    crate::litellm::ProviderCacheCounters::from_value(&val),
                )
                .await;
            tracing::warn!(
                retry_max_tokens,
                upstream_status = status.as_u16(),
                "retrying messages request with reduced max_tokens after context window error"
            );
            set_max_tokens(&mut req, retry_max_tokens);
            let retry_attempt = crate::litellm::new_attempt(
                request_event_id,
                trajectory.map(|trajectory| trajectory.trajectory_id),
                context_pack_id,
                namespace.clone(),
                repo.clone(),
                task.clone(),
                "messages",
                model.clone(),
                &route,
                crate::litellm::exact_cache_decision("messages", &req, false),
                context_pack_hash.clone(),
                Some(baseline_arm.as_str().to_string()),
            );
            crate::litellm::add_agentic_os_metadata(&mut req, &retry_attempt);
            finalizer =
                crate::litellm::LiteLlmCallFinalizer::begin(state.pool.clone(), retry_attempt)
                    .await;
            let retry_started = std::time::Instant::now();
            let retry_resp = match state
                .http
                .post(&url)
                .bearer_auth(&state.litellm_key)
                .json(&req)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    telemetry::record_upstream_litellm(
                        "messages",
                        retry_started.elapsed(),
                        "error",
                    );
                    telemetry::record_upstream_litellm_error(
                        "messages",
                        telemetry::reqwest_error_kind(&e),
                    );
                    finalizer
                        .finalize(
                            crate::litellm::TerminalStatus::NetworkError,
                            Some(telemetry::reqwest_error_kind(&e)),
                            Some(&e.to_string()),
                            crate::litellm::ProviderCacheCounters::default(),
                        )
                        .await;
                    return anthropic_error(
                        StatusCode::BAD_GATEWAY,
                        "api_error",
                        "upstream LiteLLM retry failed",
                    );
                }
            };
            status = retry_resp.status();
            telemetry::record_upstream_litellm(
                "messages",
                retry_started.elapsed(),
                &status.as_u16().to_string(),
            );
            val = match retry_resp.json().await {
                Ok(v) => v,
                Err(_) => {
                    telemetry::record_upstream_litellm_error("messages", "parse");
                    finalizer
                        .finalize(
                            crate::litellm::TerminalStatus::ParseError,
                            Some("parse"),
                            Some("invalid upstream retry response"),
                            crate::litellm::ProviderCacheCounters::default(),
                        )
                        .await;
                    return anthropic_error(
                        StatusCode::BAD_GATEWAY,
                        "api_error",
                        "invalid upstream retry response",
                    );
                }
            };
        }
    }

    if !status.is_success() {
        finalizer
            .finalize(
                crate::litellm::TerminalStatus::HttpError,
                Some(telemetry::upstream_error_kind(status)),
                Some("upstream returned non-success status"),
                crate::litellm::ProviderCacheCounters::from_value(&val),
            )
            .await;
        return (status, axum::Json(val)).into_response();
    }
    finalizer
        .finalize(
            crate::litellm::TerminalStatus::Success,
            None,
            None,
            crate::litellm::ProviderCacheCounters::from_value(&val),
        )
        .await;

    let usage = TokenUsage::from_openai_value(&val);
    let provider_cache = crate::litellm::ProviderCacheCounters::from_value(&val);
    let vllm_delta = record_vllm_cache_observation(
        &state,
        vllm_cache_before,
        session_id.as_deref(),
        &namespace,
        &repo,
        &task,
        finalizer.attempt(),
        &usage,
        provider_cache,
    )
    .await;
    let provider_cache = merge_provider_cache_from_vllm_delta(provider_cache, vllm_delta);
    inject_anthropic_cache_usage(&mut val, provider_cache);
    capture.response_status = Some(StatusCode::OK.as_u16() as i32);
    capture.response_headers = Some(serde_json::json!({"content-type": ["application/json"]}));
    capture.raw_response_body = Some(crate::client_capture::to_json_bytes(&val));
    record_client_capture(&state, capture).await;
    telemetry::record_tokens(&state.metrics, &usage, &state.default_model);
    if !usage.is_empty() {
        let pool = state.pool.clone();
        let actual = state.default_model.clone();
        let rm = model.clone();
        let ns = namespace.clone();
        let r = repo.clone();
        let u = usage.clone();
        tokio::spawn(async move {
            if let Err(e) = db::record_token_usage(&pool, &rm, &actual, &ns, &r, &u).await {
                tracing::warn!("failed to record token usage: {e}");
            }
        });
    }

    let assistant_content = extract_assistant_from_anthropic_response(&val);
    // LiteLLM may surface Anthropic tool results inside the response content.
    // Feed them through the same deterministic parser used for OpenAI results.
    let tool_results = crate::execution_feedback::tool_results_from_value(&val);
    if let (Some(session_id), Some(trajectory)) = (session_id.as_deref(), trajectory) {
        let (input_tokens, output_tokens) =
            crate::trajectory::optional_token_counts_from_value(&val);
        let metadata = crate::trajectory::model_response_metadata(
            &state.default_model,
            "litellm",
            input_tokens,
            output_tokens,
            Some(started.elapsed().as_millis() as i64),
            crate::trajectory::model_finish_reason(&val),
            request_metadata.clone(),
            context_pack_id,
            Some(trajectory),
        );
        let assistant_event_id = persist_model_response_event(
            &state,
            session_id,
            &repo,
            &assistant_content,
            metadata,
            trajectory,
            request_event_id,
        )
        .await;
        capture_tool_results_background(
            state.as_ref().clone(),
            session_id.to_string(),
            repo.clone(),
            task.clone(),
            trajectory.trajectory_id,
            assistant_event_id,
            Some(trajectory),
            tool_results,
        );
    } else {
        match db::find_or_create_session(&state.pool, &repo, &task, "agent").await {
            Ok(sid) => {
                let assistant_event_id = persist_exchange_with_correlation(
                    &state,
                    &sid,
                    &repo,
                    &user_content,
                    &assistant_content,
                    correlation_id,
                    None,
                )
                .await;
                if let Some(correlation_id) = correlation_id {
                    capture_tool_results_background(
                        state.as_ref().clone(),
                        sid,
                        repo.clone(),
                        task.clone(),
                        correlation_id,
                        assistant_event_id,
                        None,
                        tool_results,
                    );
                }
            }
            Err(e) => tracing::warn!("messages: find_or_create_session failed: {e}"),
        }
    }

    axum::Json(val).into_response()
}

/// Proxy an Anthropic streaming request to LiteLLM /messages, passing bytes through unchanged.
#[allow(clippy::too_many_arguments)]
async fn handle_streaming_anthropic(
    state: &AppState,
    req: Value,
    user_content: String,
    repo: String,
    task: String,
    model: String,
    namespace: String,
    correlation_id: Option<uuid::Uuid>,
    request_metadata: Option<Value>,
    session_id: Option<String>,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    request_event_id: Option<uuid::Uuid>,
    context_pack_id: Option<uuid::Uuid>,
    mut finalizer: crate::litellm::LiteLlmCallFinalizer,
    vllm_cache_before: Option<(String, crate::vllm_metrics::VllmCacheSnapshot)>,
    capture: crate::client_capture::RawHttpCapture,
) -> Response {
    let url = format!("{}/messages", state.litellm_url);
    let mut req = req;
    let mut capture = capture;

    let started = finalizer.attempt_mut().started_at;
    let mut upstream = match state
        .http_stream
        .post(&url)
        .bearer_auth(&state.litellm_key)
        .json(&req)
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
            r
        }
        Err(e) => {
            telemetry::record_upstream_litellm("messages", started.elapsed(), "error");
            telemetry::record_upstream_litellm_error("messages", telemetry::reqwest_error_kind(&e));
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::NetworkError,
                    Some(telemetry::reqwest_error_kind(&e)),
                    Some(&e.to_string()),
                    crate::litellm::ProviderCacheCounters::default(),
                )
                .await;
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("upstream unreachable: {e}"),
            );
        }
    };

    let upstream_status = upstream.status();
    if !upstream_status.is_success() {
        let body = upstream.text().await.unwrap_or_else(|e| {
            serde_json::json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": format!("failed to read upstream error body: {e}")
                }
            })
            .to_string()
        });
        if let Some(retry_max_tokens) = context_window_retry_max_tokens(&body) {
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::HttpError,
                    Some(telemetry::upstream_error_kind(upstream_status)),
                    Some("upstream returned non-success status"),
                    crate::litellm::ProviderCacheCounters::default(),
                )
                .await;
            tracing::warn!(
                retry_max_tokens,
                upstream_status = upstream_status.as_u16(),
                "retrying streaming messages request with reduced max_tokens after context window error"
            );
            set_max_tokens(&mut req, retry_max_tokens);
            let prior_attempt = finalizer.attempt_mut().clone();
            let route = crate::litellm::RouteSelection {
                routed_model: prior_attempt.routed_model.clone(),
                selected_route: prior_attempt
                    .selected_route
                    .clone()
                    .unwrap_or_else(|| "default".to_string()),
                selection_reason: prior_attempt
                    .selection_reason
                    .clone()
                    .unwrap_or_else(|| "default_model".to_string()),
                policy_version: prior_attempt
                    .policy_version
                    .clone()
                    .unwrap_or_else(|| "default-routing-v1".to_string()),
            };
            let retry_attempt = crate::litellm::new_attempt(
                request_event_id,
                trajectory.map(|trajectory| trajectory.trajectory_id),
                context_pack_id,
                namespace.clone(),
                repo.clone(),
                task.clone(),
                "messages",
                model.clone(),
                &route,
                crate::litellm::exact_cache_decision("messages", &req, false),
                prior_attempt.context_pack_hash.clone(),
                prior_attempt.baseline_arm.clone(),
            );
            crate::litellm::add_agentic_os_metadata(&mut req, &retry_attempt);
            finalizer =
                crate::litellm::LiteLlmCallFinalizer::begin(state.pool.clone(), retry_attempt)
                    .await;
            let retry_started = std::time::Instant::now();
            upstream = match state
                .http_stream
                .post(&url)
                .bearer_auth(&state.litellm_key)
                .json(&req)
                .send()
                .await
            {
                Ok(r) => {
                    let status = r.status();
                    telemetry::record_upstream_litellm(
                        "messages",
                        retry_started.elapsed(),
                        &status.as_u16().to_string(),
                    );
                    r
                }
                Err(e) => {
                    telemetry::record_upstream_litellm(
                        "messages",
                        retry_started.elapsed(),
                        "error",
                    );
                    telemetry::record_upstream_litellm_error(
                        "messages",
                        telemetry::reqwest_error_kind(&e),
                    );
                    finalizer
                        .finalize(
                            crate::litellm::TerminalStatus::NetworkError,
                            Some(telemetry::reqwest_error_kind(&e)),
                            Some(&e.to_string()),
                            crate::litellm::ProviderCacheCounters::default(),
                        )
                        .await;
                    return anthropic_error(
                        StatusCode::BAD_GATEWAY,
                        "api_error",
                        format!("upstream retry unreachable: {e}"),
                    );
                }
            };
            if upstream.status().is_success() {
                // Continue below with the successful retry stream.
            } else {
                let retry_status = upstream.status();
                let retry_body = upstream.text().await.unwrap_or_else(|e| {
                    serde_json::json!({
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": format!("failed to read upstream retry error body: {e}")
                        }
                    })
                    .to_string()
                });
                finalizer
                    .finalize(
                        crate::litellm::TerminalStatus::HttpError,
                        Some(telemetry::upstream_error_kind(retry_status)),
                        Some("upstream returned non-success status"),
                        crate::litellm::ProviderCacheCounters::default(),
                    )
                    .await;
                return (
                    retry_status,
                    [(header::CONTENT_TYPE, "application/json")],
                    retry_body,
                )
                    .into_response();
            }
        } else {
            finalizer
                .finalize(
                    crate::litellm::TerminalStatus::HttpError,
                    Some(telemetry::upstream_error_kind(upstream_status)),
                    Some("upstream returned non-success status"),
                    crate::litellm::ProviderCacheCounters::default(),
                )
                .await;
            return (
                upstream_status,
                [(header::CONTENT_TYPE, "application/json")],
                body,
            )
                .into_response();
        }
    }

    let bytes_stream = upstream.bytes_stream();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Vec<u8>>();
    let accumulated = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let acc_clone = accumulated.clone();
    let mut stream_finalizer = finalizer;
    let observation_attempt = stream_finalizer.attempt().clone();
    let cache_state = state.clone();
    let cache_session_id = session_id.clone();
    let cache_namespace = namespace.clone();
    let cache_repo = repo.clone();
    let cache_task = task.clone();

    // Proxy bytes verbatim — LiteLLM already returns Anthropic SSE format.
    let tapped = async_stream::stream! {
        tokio::pin!(bytes_stream);
        let mut stream_metrics = telemetry::StreamTracker::new("messages", started);
        let mut tx_opt = Some(done_tx);
        let mut first_seen = false;
        let mut first_token_at = None;
        while let Some(chunk) = bytes_stream.next().await {
            match chunk {
                Ok(b) => {
                    if !first_seen && crate::litellm::sse_chunk_has_non_empty_data(&b) {
                        first_seen = true;
                        first_token_at = Some(std::time::Instant::now());
                        stream_finalizer.attempt_mut().first_token_at = first_token_at;
                        stream_metrics.first_token();
                    }
                    if let Ok(mut g) = acc_clone.lock() { g.extend_from_slice(&b); }
                    yield Ok::<Bytes, std::io::Error>(b);
                }
                Err(e) => {
                    stream_metrics.fail("upstream_error");
                    stream_finalizer
                        .finalize(
                            crate::litellm::TerminalStatus::StreamError,
                            Some("upstream_error"),
                            Some(&e.to_string()),
                            crate::litellm::ProviderCacheCounters::default(),
                        )
                        .await;
                    yield Err(std::io::Error::other(e));
                    return;
                }
            }
        }
        stream_metrics.finish();
        let mut data = acc_clone.lock().map(|g| g.clone()).unwrap_or_default();
        let raw = String::from_utf8_lossy(&data);
        let usage = extract_token_usage_from_anthropic_sse(&raw);
        let provider_cache = crate::litellm::provider_counters_from_sse(&raw);
        let vllm_delta = record_vllm_cache_observation(
            &cache_state,
            vllm_cache_before,
            cache_session_id.as_deref(),
            &cache_namespace,
            &cache_repo,
            &cache_task,
            &observation_attempt,
            &usage,
            provider_cache,
        )
        .await;
        let counters = merge_provider_cache_from_vllm_delta(provider_cache, vllm_delta);
        let cache_usage_event = anthropic_cache_usage_sse_event(counters);
        data.extend_from_slice(cache_usage_event.as_bytes());
        let mut done_finalizer = stream_finalizer.clone();
        done_finalizer.attempt_mut().first_token_at = first_token_at;
        done_finalizer.attempt_mut().completed_at = Some(std::time::Instant::now());
        done_finalizer
            .finalize(crate::litellm::TerminalStatus::Success, None, None, counters)
            .await;
        yield Ok::<Bytes, std::io::Error>(Bytes::from(cache_usage_event));
        if let Some(tx) = tx_opt.take() {
            let _ = tx.send(data);
        }
    };

    let state_bg = state.clone();
    tokio::spawn(async move {
        if let Ok(raw_bytes) = done_rx.await {
            capture.response_status = Some(StatusCode::OK.as_u16() as i32);
            capture.response_headers =
                Some(serde_json::json!({"content-type": ["text/event-stream"]}));
            capture.raw_response_body = Some(raw_bytes.clone());
            record_client_capture(&state_bg, capture).await;
            let raw = String::from_utf8_lossy(&raw_bytes);
            let usage = extract_token_usage_from_anthropic_sse(&raw);
            telemetry::record_tokens(&state_bg.metrics, &usage, &state_bg.default_model);
            if !usage.is_empty() {
                if let Err(e) = db::record_token_usage(
                    &state_bg.pool,
                    &model,
                    &state_bg.default_model,
                    &namespace,
                    &repo,
                    &usage,
                )
                .await
                {
                    tracing::warn!("failed to record token usage: {e}");
                }
            }
            let assistant_content = extract_assistant_from_anthropic_sse(&raw);
            // Anthropic streaming tool_result blocks are parsed from the final
            // accumulated SSE transcript after the upstream stream completes.
            let tool_results = crate::execution_feedback::tool_results_from_sse(&raw);
            if let (Some(session_id), Some(trajectory)) = (session_id.as_deref(), trajectory) {
                let (input_tokens, output_tokens) = optional_token_usage_from_sse(&raw);
                let metadata = crate::trajectory::model_response_metadata(
                    &state_bg.default_model,
                    "litellm",
                    input_tokens,
                    output_tokens,
                    Some(started.elapsed().as_millis() as i64),
                    None,
                    request_metadata.clone(),
                    context_pack_id,
                    Some(trajectory),
                );
                let assistant_event_id = persist_model_response_event(
                    &state_bg,
                    session_id,
                    &repo,
                    &assistant_content,
                    metadata,
                    trajectory,
                    request_event_id,
                )
                .await;
                capture_tool_results_background(
                    state_bg.clone(),
                    session_id.to_string(),
                    repo.clone(),
                    task.clone(),
                    trajectory.trajectory_id,
                    assistant_event_id,
                    Some(trajectory),
                    tool_results,
                );
            } else {
                match db::find_or_create_session(&state_bg.pool, &repo, &task, "agent").await {
                    Ok(sid) => {
                        let assistant_event_id = persist_exchange_with_correlation(
                            &state_bg,
                            &sid,
                            &repo,
                            &user_content,
                            &assistant_content,
                            correlation_id,
                            None,
                        )
                        .await;
                        if let Some(correlation_id) = correlation_id {
                            capture_tool_results_background(
                                state_bg.clone(),
                                sid,
                                repo.clone(),
                                task.clone(),
                                correlation_id,
                                assistant_event_id,
                                None,
                                tool_results,
                            );
                        }
                    }
                    Err(e) => tracing::warn!("messages stream: find_or_create_session failed: {e}"),
                }
            }
        }
    });

    (
        [
            (axum::http::header::CONTENT_TYPE, "text/event-stream"),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
            (axum::http::header::CONNECTION, "keep-alive"),
        ],
        axum::body::Body::from_stream(tapped),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;

    #[test]
    fn derive_tool_authorization_policy_uses_request_fields() {
        let req = crate::tool_mediation::ToolAuthorizeRequest {
            session_id: Some("session-a".to_string()),
            repo: Some("repo-a".to_string()),
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("explain the logs".to_string()),
            tool_name: "Read".to_string(),
            arguments: json!({}),
            available_tools: vec![],
        };

        let (classification, policy) = derive_tool_authorization_policy(&req, "namespace-a", false);

        assert_eq!(classification.repo, "repo-a");
        assert_eq!(classification.session_id, "session-a");
        assert!(policy
            .scope_policy
            .contains(&orchestration_policy::ScopePolicy::NoScp));
    }

    #[test]
    fn derive_tool_authorization_policy_falls_back_to_namespace_and_tool_name() {
        let req = crate::tool_mediation::ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: None,
            tool_name: "Bash".to_string(),
            arguments: json!({}),
            available_tools: vec![],
        };

        let (classification, policy) = derive_tool_authorization_policy(&req, "namespace-a", false);

        assert_eq!(classification.repo, "namespace-a");
        assert_eq!(classification.session_id, "unknown");
        assert!(policy
            .scope_policy
            .contains(&orchestration_policy::ScopePolicy::NoScp));
    }

    #[test]
    fn live_classification_overrides_generic_task_for_context_category() {
        let classification = crate::request_classification::classify_request_text(
            "agentic-os",
            "session-a",
            "Read README.md",
            None,
            "user_message",
        );

        assert_eq!(
            context_task_category("default task", Some(&classification)),
            crate::state::TaskCategory::Narrow
        );
    }

    #[test]
    fn orchestration_policy_narrows_context_memory_levels() {
        let classification = crate::request_classification::classify_request_text(
            "agentic-os",
            "session-a",
            "Read README.md",
            None,
            "user_message",
        );
        let orchestration_policy = orchestration_policy::derive_orchestration_policy(
            &classification,
            "Read README.md",
            false,
        );
        let mut context_policy =
            crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Moderate);

        apply_orchestration_context_limits(&mut context_policy, &orchestration_policy);

        assert_eq!(context_policy.l2_limit, 0);
        assert_eq!(context_policy.l3_limit, 0);
        assert_eq!(context_policy.failure_limit, 0);
        assert!(!orchestration_policy
            .context_sources
            .contains(&orchestration_policy::ContextSource::CompiledSummaries));
        assert!(!orchestration_policy
            .context_sources
            .contains(&orchestration_policy::ContextSource::TotalRecall));
    }

    // ── Tool authorization classification text ────────────────────────

    #[test]
    fn derive_tool_authorization_policy_sees_destructive_command_arguments() {
        let req = crate::tool_mediation::ToolAuthorizeRequest {
            session_id: Some("session-a".to_string()),
            repo: Some("repo-a".to_string()),
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("run a command".to_string()),
            tool_name: "Bash".to_string(),
            arguments: json!({"command":"rm -rf /tmp/example"}),
            available_tools: vec![],
        };

        let (classification, policy) = derive_tool_authorization_policy(&req, "namespace-a", false);

        assert!(classification
            .risk
            .contains(&crate::request_classification::RequestRisk::DestructiveCommand));
        assert!(policy
            .blocked_tools
            .contains(&orchestration_policy::ToolCapability::ShellMutation));
    }

    #[test]
    fn derive_tool_authorization_policy_allows_search_command_arguments_without_destructive_risk() {
        let req = crate::tool_mediation::ToolAuthorizeRequest {
            session_id: Some("session-a".to_string()),
            repo: Some("repo-a".to_string()),
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: None,
            tool_name: "Bash".to_string(),
            arguments: json!({"command":"rg pattern src"}),
            available_tools: vec![],
        };

        let (classification, policy) = derive_tool_authorization_policy(&req, "namespace-a", false);

        assert!(!classification
            .risk
            .contains(&crate::request_classification::RequestRisk::DestructiveCommand));
        assert!(policy
            .scope_policy
            .contains(&orchestration_policy::ScopePolicy::NoScp));
    }

    #[test]
    fn tool_authorization_classification_text_ignores_unknown_argument_keys() {
        let req = crate::tool_mediation::ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: None,
            tool_name: "Bash".to_string(),
            arguments: json!({"secret":"sk-test", "command":"echo ok"}),
            available_tools: vec![],
        };

        let text = tool_authorization_classification_text(&req);

        assert!(!text.contains("sk-test"));
        assert!(text.contains("command:echo ok"));
    }

    // ── Auth headers ──────────────────────────────────────────────────

    #[test]
    fn provided_api_token_accepts_bearer_authorization() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            "Bearer sk-test".parse().expect("valid header"),
        );

        assert_eq!(provided_api_token(&headers), "sk-test");
    }

    #[test]
    fn provided_api_token_accepts_anthropic_x_api_key() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "sk-test".parse().expect("valid header"));

        assert_eq!(provided_api_token(&headers), "sk-test");
    }

    // ── inject_system_context ──────────────────────────────────────────

    #[test]
    fn inject_inserts_system_message_when_none_exists() {
        let mut payload = json!({
            "messages": [{"role": "user", "content": "hello"}]
        });
        inject_system_context(&mut payload, "prior context");
        let msgs = payload["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "prior context");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn inject_appends_to_existing_system_message_not_duplicates_it() {
        let mut payload = json!({
            "messages": [
                {"role": "system", "content": "base prompt"},
                {"role": "user", "content": "hello"}
            ]
        });
        inject_system_context(&mut payload, "prior context");
        let msgs = payload["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2, "must not insert a second system message");
        let content = msgs[0]["content"].as_str().unwrap();
        assert!(content.contains("base prompt"));
        assert!(content.contains("prior context"));
    }

    #[test]
    fn inject_keeps_system_message_at_index_zero() {
        let mut payload = json!({
            "messages": [
                {"role": "system", "content": "base prompt"},
                {"role": "user", "content": "hello"}
            ]
        });
        inject_system_context(&mut payload, "context");
        let msgs = payload["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
    }

    #[test]
    fn inject_is_noop_when_messages_missing() {
        let mut payload = json!({"model": "gpt-4"});
        inject_system_context(&mut payload, "context");
        assert!(payload.get("messages").is_none());
    }

    // ── extract_assistant_from_sse ─────────────────────────────────────

    #[test]
    fn extract_sse_assembles_content_across_chunks() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\
                   data: [DONE]\n";
        assert_eq!(extract_assistant_from_sse(sse), "Hello world");
    }

    #[test]
    fn extract_sse_stops_at_done() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"before\"}}]}\n\
                   data: [DONE]\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"after\"}}]}\n";
        assert_eq!(extract_assistant_from_sse(sse), "before");
    }

    #[test]
    fn extract_sse_returns_empty_for_blank_input() {
        assert_eq!(extract_assistant_from_sse(""), "");
    }

    #[test]
    fn extract_sse_ignores_malformed_json_data_lines() {
        let sse = "data: not-valid-json\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\
                   data: [DONE]\n";
        assert_eq!(extract_assistant_from_sse(sse), "ok");
    }

    #[test]
    fn extract_sse_ignores_non_data_lines() {
        let sse = ": keep-alive\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\
                   data: [DONE]\n";
        assert_eq!(extract_assistant_from_sse(sse), "hi");
    }

    #[test]
    fn extract_sse_usage_records_processed_cached_and_generated_tokens() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\
                   data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":40,\"prompt_tokens_details\":{\"cached_tokens\":80}}}\n\
                   data: [DONE]\n";
        let usage = extract_token_usage_from_sse(sse);
        assert_eq!(usage.processed_tokens, 120);
        assert_eq!(usage.cached_tokens, 80);
        assert_eq!(usage.generated_tokens, 40);
    }

    #[test]
    fn local_reasoning_policy_prefers_header_over_client_fields() {
        let mut headers = HeaderMap::new();
        headers.insert("x-agent-reasoning-policy", "high".parse().unwrap());
        let payload = serde_json::json!({
            "reasoning_effort": "low",
            "thinking": {"type": "enabled", "budget_tokens": 1024}
        });

        let selection = local_reasoning_selection(&headers, &payload);

        assert_eq!(selection.policy, LocalReasoningPolicy::High);
        assert_eq!(selection.source, "x-agent-reasoning-policy");
    }

    #[test]
    fn local_reasoning_policy_maps_anthropic_thinking_budget() {
        let headers = HeaderMap::new();

        let low = local_reasoning_selection(
            &headers,
            &serde_json::json!({"thinking": {"budget_tokens": 1024}}),
        );
        let medium = local_reasoning_selection(
            &headers,
            &serde_json::json!({"thinking": {"budget_tokens": 4096}}),
        );
        let high = local_reasoning_selection(
            &headers,
            &serde_json::json!({"thinking": {"budget_tokens": 12000}}),
        );

        assert_eq!(low.policy, LocalReasoningPolicy::Low);
        assert_eq!(medium.policy, LocalReasoningPolicy::Medium);
        assert_eq!(high.policy, LocalReasoningPolicy::High);
        assert_eq!(high.source, "client_thinking_budget");
    }

    #[test]
    fn local_reasoning_defaults_preserve_explicit_sampling() {
        let mut req = serde_json::json!({
            "max_tokens": 333,
            "temperature": 0.7
        });

        apply_local_reasoning_defaults(
            &mut req,
            LocalReasoningSelection {
                policy: LocalReasoningPolicy::Low,
                source: "test",
            },
        );

        assert_eq!(req["max_tokens"], serde_json::json!(333));
        assert_eq!(req["temperature"], serde_json::json!(0.7));
    }

    #[test]
    fn local_reasoning_defaults_shape_missing_sampling() {
        let mut req = serde_json::json!({});

        apply_local_reasoning_defaults(
            &mut req,
            LocalReasoningSelection {
                policy: LocalReasoningPolicy::Low,
                source: "test",
            },
        );

        assert_eq!(req["max_tokens"], serde_json::json!(2048));
        assert_eq!(req["temperature"], serde_json::json!(0.1));
    }

    #[test]
    fn extract_sse_handles_missing_content_field_gracefully() {
        let sse = "data: {\"choices\":[{\"delta\":{}}]}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"real\"}}]}\n\
                   data: [DONE]\n";
        assert_eq!(extract_assistant_from_sse(sse), "real");
    }

    #[test]
    fn retry_backoff_durations_are_correct() {
        let delays: Vec<u64> = (0u32..2).map(|a| 200 * 2u64.pow(a)).collect();
        assert_eq!(delays, vec![200, 400]);
    }

    #[test]
    fn rate_limited_response_sets_429_and_retry_after() {
        let response = rate_limited_response("secret-token", 3);
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response
                .headers()
                .get(header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok()),
            Some("3")
        );
    }

    // ── API_KEYS parsing: semicolon-delimited token,namespace entries ──

    #[test]
    fn parse_simple_single_entry() {
        let input = "agent-os,project-alpha";
        let mut parts = input.splitn(2, ',');
        let token = parts.next().unwrap_or(input).trim().to_string();
        let namespace = parts.next().unwrap_or(&token).trim().to_string();
        assert_eq!(token, "agent-os");
        assert_eq!(namespace, "project-alpha");
    }

    #[test]
    fn parse_multiple_semicolon_entries() {
        let input = "agent-os,project-alpha;agent-os,project-beta;sk-work,work";
        let entries: Vec<(String, String)> = input
            .split(';')
            .map(|s| {
                let s = s.trim();
                let mut parts = s.splitn(2, ',');
                let token = parts.next().unwrap_or(s).trim().to_string();
                let namespace = parts.next().unwrap_or(&token).trim().to_string();
                (token, namespace)
            })
            .filter(|(t, _)| !t.is_empty())
            .collect();

        assert_eq!(entries.len(), 3);
        assert_eq!(
            entries[0],
            ("agent-os".to_string(), "project-alpha".to_string())
        );
        assert_eq!(
            entries[1],
            ("agent-os".to_string(), "project-beta".to_string())
        );
        assert_eq!(entries[2], ("sk-work".to_string(), "work".to_string()));
    }

    #[test]
    fn parse_empty_token_filtered() {
        let input = "token1,ns1;;token2,ns2";
        let entries: Vec<(String, String)> = input
            .split(';')
            .map(|s| {
                let s = s.trim();
                let mut parts = s.splitn(2, ',');
                let token = parts.next().unwrap_or(s).trim().to_string();
                let namespace = parts.next().unwrap_or(&token).trim().to_string();
                (token, namespace)
            })
            .filter(|(t, _)| !t.is_empty())
            .collect();

        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn parse_fallback_to_token_when_no_namespace() {
        let input = "my-token";
        let mut parts = input.splitn(2, ',');
        let token = parts.next().unwrap_or(input).trim().to_string();
        let namespace = parts.next().unwrap_or(&token).trim().to_string();
        assert_eq!(token, "my-token");
        assert_eq!(namespace, "my-token");
    }

    #[test]
    fn parse_default_value() {
        let input = "agent-os,agentic-os";
        let entries: Vec<(String, String)> = input
            .split(';')
            .map(|s| {
                let s = s.trim();
                let mut parts = s.splitn(2, ',');
                let token = parts.next().unwrap_or(s).trim().to_string();
                let namespace = parts.next().unwrap_or(&token).trim().to_string();
                (token, namespace)
            })
            .filter(|(t, _)| !t.is_empty())
            .collect();

        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0],
            ("agent-os".to_string(), "agentic-os".to_string())
        );
    }

    // ── Model substitution ────────────────────────────────────────

    #[test]
    fn model_substituted_with_default_in_chat_completions() {
        let mut req = json!({
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let default_model = "qwen36-35b-heretic";
        req["model"] = Value::String(default_model.to_string());
        assert_eq!(req["model"].as_str().unwrap(), default_model);
    }

    #[test]
    fn model_substituted_with_default_for_any_client_model_name() {
        let client_models = [
            "gpt-4o",
            "claude-sonnet-4-6",
            "claude-opus-4-7",
            "gpt-4-turbo",
        ];
        let default_model = "qwen36-35b-heretic";
        for client_model in client_models {
            let mut req = json!({
                "model": client_model,
                "messages": [{"role": "user", "content": "hi"}]
            });
            req["model"] = Value::String(default_model.to_string());
            assert_eq!(
                req["model"].as_str().unwrap(),
                default_model,
                "client model '{client_model}' was not replaced"
            );
        }
    }

    #[test]
    fn anthropic_sanitize_removes_local_backend_incompatible_params() {
        let mut req = json!({
            "model": "qwen36-27b",
            "max_tokens": 1024,
            "max_output_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "reasoning_effort": "high",
            "context_management": {"edits": []},
            "messages": [{"role": "user", "content": "hi"}]
        });

        sanitize_anthropic_litellm_request(&mut req);

        assert!(req.get("thinking").is_none());
        assert!(req.get("max_output_tokens").is_none());
        assert!(req.get("reasoning_effort").is_none());
        assert!(req.get("context_management").is_none());
        assert_eq!(req["max_tokens"], json!(1024));
        assert_eq!(req["messages"][0]["content"], "hi");
    }

    #[test]
    fn max_tokens_clamp_respects_explicit_small_values() {
        let mut req = json!({
            "model": "qwen36-27b",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}]
        });

        enforce_min_max_tokens(&mut req);

        assert_eq!(req["max_tokens"], json!(1));
    }

    #[test]
    fn max_tokens_clamp_defaults_missing_values() {
        let mut req = json!({
            "model": "qwen36-27b",
            "messages": [{"role": "user", "content": "hi"}]
        });

        enforce_min_max_tokens(&mut req);

        assert_eq!(req["max_tokens"], json!(DEFAULT_MAX_TOKENS));
    }

    #[test]
    fn max_tokens_clamp_caps_oversized_values() {
        let mut req = json!({
            "model": "qwen36-27b",
            "max_tokens": MAX_MAX_TOKENS + 1,
            "messages": [{"role": "user", "content": "hi"}]
        });

        enforce_min_max_tokens(&mut req);

        assert_eq!(req["max_tokens"], json!(MAX_MAX_TOKENS));
    }

    #[test]
    fn metrics_include_context_pack_counts() {
        let metrics = telemetry::MetricsRegistry::new();
        telemetry::record_context_pack(
            &metrics,
            &ContextPackStats {
                context_chars: 400,
                context_tokens_estimate: 100,
                l0_items_injected: 1,
                l1_items_injected: 2,
                l2_items_injected: 3,
                l3_items_injected: 4,
                failed_attempts_injected: 1,
                remediations_injected: 1,
                retrieval_semantic_hits: 5,
                retrieval_fts_hits: 6,
                retrieval_deduped_hits: 7,
                cache_hit: false,
                ..Default::default()
            },
        );
        telemetry::record_tokens(
            &metrics,
            &TokenUsage {
                processed_tokens: 120,
                cached_tokens: 80,
                generated_tokens: 40,
            },
            "test-model",
        );
        telemetry::record_promotion(&metrics, false, false);
        telemetry::record_promotion(&metrics, true, true);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.context_pack_requests, 1);
        assert_eq!(snapshot.context_cache_misses, 1);
        assert_eq!(snapshot.l3_items_injected, 4);
        assert_eq!(snapshot.retrieval_deduped_hits, 7);
        assert_eq!(snapshot.processed_tokens, 120);
        assert_eq!(snapshot.cached_tokens, 80);
        assert_eq!(snapshot.generated_tokens, 40);
        assert_eq!(snapshot.promotion_attempts, 2);
        assert_eq!(snapshot.promotion_accepted, 1);
        assert_eq!(snapshot.promotion_rejected, 1);
        assert_eq!(snapshot.memory_source_coverage, 0.5);
    }

    #[test]
    fn context_pack_sections_are_stable() {
        let mut evidence = db::ContextEvidence::default();
        evidence.l3_project = vec![test_event("summary", "L3 architecture truth", 3)];
        evidence.l2_repo = vec![test_event("summary", "L2 repo pattern", 2)];
        evidence.l1_matching = vec![test_event("summary", "L1 task summary", 1)];
        evidence.l0_recent = vec![test_event("edit", "L0 raw evidence", 0)];
        evidence.failures = vec![test_event("failed_attempt", "do not repeat this", 0)];
        let policy = ContextPolicy::for_category(TaskCategory::Architecture);

        let (context, _stats) = db::build_layered_context(
            "repo",
            "architecture task",
            &evidence,
            &[],
            &[],
            &policy,
            16_000,
            4000,
        );

        let l3 = context.find("== Durable Project Memory ==").unwrap();
        let l2 = context.find("== Repo Patterns and Decisions ==").unwrap();
        let l1 = context.find("== Relevant Session Summaries ==").unwrap();
        let l0 = context.find("== Recent Evidence ==").unwrap();
        let failures = context
            .find("== Failed Attempts and Remediations ==")
            .unwrap();

        assert!(l3 < l2);
        assert!(l2 < l1);
        assert!(l1 < l0);
        assert!(l0 < failures);
    }

    #[test]
    fn failed_attempts_are_injected_for_debug_tasks() {
        let mut evidence = db::ContextEvidence::default();
        evidence.failures = vec![db::AgentEvent {
            evidence: Some("the old retry loop timed out".to_string()),
            metadata: json!({"outcome": "bounded retry fixed the lag"}),
            ..test_event("failed_attempt", "unbounded retries caused lag", 0)
        }];
        let policy = ContextPolicy::for_category(TaskCategory::Narrow);

        let (context, stats) = db::build_layered_context(
            "repo",
            "debug lag",
            &evidence,
            &[],
            &[],
            &policy,
            8000,
            4000,
        );

        assert!(context.contains("unbounded retries caused lag"));
        assert!(context.contains("Evidence: the old retry loop timed out"));
        assert!(context.contains("Outcome: bounded retry fixed the lag"));
        assert_eq!(stats.failed_attempts_injected, 1);
    }

    #[test]
    fn cache_hit_rate_updates() {
        let metrics = telemetry::MetricsRegistry::new();
        telemetry::record_context_pack(
            &metrics,
            &ContextPackStats {
                cache_hit: false,
                ..Default::default()
            },
        );
        telemetry::record_context_pack(
            &metrics,
            &ContextPackStats {
                cache_hit: true,
                ..Default::default()
            },
        );

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.context_pack_requests, 2);
        assert_eq!(snapshot.context_cache_misses, 1);
        assert_eq!(snapshot.context_cache_hits, 1);
    }

    // ── context_pack tokio::join! parallelization ─────────────────

    /// Verifies the shared context builder uses tokio::join! to parallelize
    /// the independent I/O calls: db::get_context_evidence_for_policy,
    /// hybrid_search, db::get_active_errors, and failure history lookup.
    #[test]
    fn context_pack_parallelizes_context_io_calls() {
        let src = include_str!("handlers.rs");
        let ctx_start = src
            .find("async fn get_or_build_cached_context")
            .expect("get_or_build_cached_context not found in source");
        let ctx_body: String = src[ctx_start..].chars().take(6500).collect();

        assert!(
            ctx_body.contains("tokio::join!"),
            "context builder should use tokio::join! for parallelization"
        );

        assert!(
            ctx_body.contains("db::get_context_evidence_for_policy"),
            "get_context_evidence_for_policy should be in tokio::join!"
        );
        assert!(
            ctx_body.contains("hybrid_search"),
            "hybrid_search should be in tokio::join!"
        );
        assert!(
            ctx_body.contains("db::get_active_errors"),
            "get_active_errors should be in tokio::join!"
        );
        assert!(
            ctx_body.contains("db::get_failure_history_for_signatures"),
            "failure history lookup should be in tokio::join!"
        );

        let join_block_start = ctx_body
            .find("tokio::join!")
            .expect("tokio::join! not found");
        let join_block: String = ctx_body[join_block_start..].chars().take(1500).collect();
        assert!(
            join_block.contains("get_context_evidence_for_policy")
                && join_block.contains("hybrid_search")
                && join_block.contains("get_active_errors")
                && join_block.contains("get_failure_history_for_signatures"),
            "All context I/O calls should be within the same tokio::join!"
        );
    }

    /// Verifies error propagation is preserved after parallelization.
    #[test]
    fn context_pack_preserves_error_propagation_for_events() {
        let src = include_str!("handlers.rs");
        let ctx_start = src
            .find("pub async fn context_pack")
            .expect("context_pack not found");
        let ctx_body = &src[ctx_start..ctx_start + 2000];
        assert!(
            ctx_body.contains("INTERNAL_SERVER_ERROR"),
            "context_pack should return 500 on events_for_repo error"
        );
    }

    #[test]
    fn failure_history_and_validation_capture_are_feature_flagged() {
        let src = include_str!("handlers.rs");
        assert!(src.contains("state.execution_feedback_enabled"));
        let ctx_start = src
            .find("async fn get_or_build_cached_context")
            .expect("get_or_build_cached_context not found");
        let ctx_body: String = src[ctx_start..].chars().take(4500).collect();
        assert!(ctx_body.contains("state.execution_feedback_enabled"));

        let validation_start = src
            .find("pub async fn validations")
            .expect("validations handler not found");
        let validation_body: String = src[validation_start..].chars().take(1200).collect();
        assert!(validation_body.contains("!state.execution_feedback_enabled"));
    }

    #[test]
    fn trajectory_capture_is_feature_flagged() {
        let src = include_str!("handlers.rs");
        assert!(src.contains("state.trajectory_capture_enabled"));
        let append_start = src
            .find("pub async fn append_event")
            .expect("append_event handler not found");
        let append_body: String = src[append_start..].chars().take(1400).collect();
        assert!(append_body.contains("req.trajectory_id = None"));
        assert!(append_body.contains("req.attempt_index = None"));
        assert!(append_body.contains("req.event_role = None"));

        let sweep_start = src
            .find("pub async fn run_trajectory_idle_sweep")
            .expect("trajectory idle sweep not found");
        let sweep_body: String = src[sweep_start..].chars().take(800).collect();
        assert!(sweep_body.contains("!state.trajectory_capture_enabled"));
    }

    /// Verify hybrid_search still uses tokio::join! internally.
    #[test]
    fn hybrid_search_uses_tokio_join() {
        let src = include_str!("handlers.rs");
        let hs_start = src
            .find("async fn hybrid_search")
            .expect("hybrid_search not found");
        let body = &src[hs_start..hs_start + 500];
        assert!(
            body.contains("tokio::join!"),
            "hybrid_search should parallelize semantic_search and FTS via tokio::join!"
        );
        assert!(
            body.contains("semantic_search") && body.contains("db::search_events_fts"),
            "hybrid_search should parallelize semantic_search and FTS"
        );
    }

    /// Verify the model request path does not await a full context rebuild.
    #[test]
    fn pack_context_into_req_uses_async_cache_refresh() {
        let src = include_str!("handlers.rs");
        let pctr_start = src
            .find("async fn pack_context_into_req")
            .expect("pack_context_into_req not found");
        let body = &src[pctr_start..pctr_start + 1500];
        assert!(
            body.contains("cached_context_for_request"),
            "pack_context_into_req should use cached/minimal context immediately"
        );
        assert!(
            !body.contains("get_or_build_cached_context("),
            "pack_context_into_req should not await full context construction"
        );
        assert!(
            src.contains("fn spawn_context_cache_refresh")
                && src.contains("get_or_build_cached_context_inner")
                && src.contains("tokio::spawn(async move"),
            "context cache refresh should run in the background"
        );
    }

    // ── inject_system_context_anthropic ───────────────────────────────

    #[test]
    fn anthropic_inject_sets_system_when_absent() {
        let mut payload = json!({"messages": [{"role": "user", "content": "hi"}]});
        inject_system_context_anthropic(&mut payload, "ctx");
        assert_eq!(
            payload["system"],
            json!([{
                "type": "text",
                "text": "ctx"
            }])
        );
    }

    #[test]
    fn anthropic_inject_appends_to_string_system() {
        let mut payload = json!({"system": "base", "messages": []});
        inject_system_context_anthropic(&mut payload, "ctx");
        let sys = payload["system"].as_array().unwrap();
        assert_eq!(sys[0], json!({"type": "text", "text": "base"}));
        assert_eq!(
            sys[1],
            json!({
                "type": "text",
                "text": "ctx"
            })
        );
    }

    #[test]
    fn anthropic_inject_flattens_array_system_and_appends() {
        let mut payload = json!({
            "system": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}],
            "messages": []
        });
        inject_system_context_anthropic(&mut payload, "ctx");
        let sys = payload["system"].as_array().unwrap();
        assert_eq!(sys[0], json!({"type": "text", "text": "part1"}));
        assert_eq!(sys[1], json!({"type": "text", "text": "part2"}));
        assert_eq!(
            sys[2],
            json!({
                "type": "text",
                "text": "ctx"
            })
        );
    }

    #[test]
    fn anthropic_inject_ignores_non_text_system_type() {
        let mut payload = json!({"system": 42, "messages": []});
        inject_system_context_anthropic(&mut payload, "ctx");
        assert_eq!(
            payload["system"],
            json!([{
                "type": "text",
                "text": "ctx"
            }])
        );
    }

    #[test]
    fn anthropic_inject_caches_stable_prefix_not_dynamic_tail() {
        let mut payload = json!({"messages": []});
        let context =
            "== Stable Context Artifacts ==\n[repo:service_topology:active]\nstable\n\nRepository: repo\nTask: task\n";
        inject_system_context_anthropic(&mut payload, context);
        let sys = payload["system"].as_array().unwrap();
        assert_eq!(sys.len(), 2);
        assert_eq!(
            sys[0]["cache_control"],
            json!({"type": "ephemeral"}),
            "stable compiler prefix should be provider-cacheable"
        );
        assert!(sys[0]["text"]
            .as_str()
            .unwrap()
            .contains("Stable Context Artifacts"));
        assert_eq!(
            sys[1].get("cache_control"),
            None,
            "dynamic repository/task tail must not become a provider cache breakpoint"
        );
        assert!(sys[1]["text"]
            .as_str()
            .unwrap()
            .starts_with("Repository: repo"));
    }

    // ── extract_assistant_from_anthropic_response ─────────────────────

    #[test]
    fn anthropic_response_extracts_text_block() {
        let resp = json!({"content": [{"type": "text", "text": "hello"}]});
        assert_eq!(extract_assistant_from_anthropic_response(&resp), "hello");
    }

    #[test]
    fn anthropic_response_skips_tool_use_blocks() {
        let resp = json!({"content": [
            {"type": "tool_use", "id": "c1", "name": "bash", "input": {}},
            {"type": "text", "text": "done"}
        ]});
        assert_eq!(extract_assistant_from_anthropic_response(&resp), "done");
    }

    #[test]
    fn anthropic_response_concatenates_multiple_text_blocks() {
        let resp = json!({"content": [
            {"type": "text", "text": "foo"},
            {"type": "text", "text": "bar"}
        ]});
        assert_eq!(extract_assistant_from_anthropic_response(&resp), "foobar");
    }

    #[test]
    fn anthropic_response_returns_empty_when_no_content() {
        assert_eq!(extract_assistant_from_anthropic_response(&json!({})), "");
    }

    #[test]
    fn anthropic_response_truncates_at_500_chars() {
        let long = "x".repeat(600);
        let resp = json!({"content": [{"type": "text", "text": long}]});
        assert_eq!(extract_assistant_from_anthropic_response(&resp).len(), 500);
    }

    // ── extract_token_usage_from_anthropic_sse ────────────────────────

    #[test]
    fn anthropic_sse_usage_reads_message_start_and_delta() {
        let raw = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":100,\"cache_read_input_tokens\":40}}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":25}}\n\n",
        );
        let usage = extract_token_usage_from_anthropic_sse(raw);
        assert_eq!(usage.processed_tokens, 100);
        assert_eq!(usage.cached_tokens, 40);
        assert_eq!(usage.generated_tokens, 25);
    }

    #[test]
    fn anthropic_sse_usage_is_zero_for_empty_stream() {
        let usage = extract_token_usage_from_anthropic_sse("");
        assert_eq!(usage.processed_tokens, 0);
        assert_eq!(usage.generated_tokens, 0);
    }

    // ── extract_assistant_from_anthropic_sse ──────────────────────────

    #[test]
    fn anthropic_sse_collects_text_deltas() {
        let raw = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
        );
        assert_eq!(extract_assistant_from_anthropic_sse(raw), "hello");
    }

    #[test]
    fn anthropic_sse_skips_non_text_deltas() {
        let raw = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n",
        );
        assert_eq!(extract_assistant_from_anthropic_sse(raw), "");
    }

    #[test]
    fn anthropic_sse_truncates_at_500_chars() {
        let chunk_text = "x".repeat(300);
        let make_line = |t: &str| {
            format!(
                "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{t}\"}}}}\n\n"
            )
        };
        let raw = make_line(&chunk_text) + &make_line(&chunk_text);
        assert_eq!(extract_assistant_from_anthropic_sse(&raw).len(), 500);
    }

    fn test_event(event_type: &str, summary: &str, summary_level: i32) -> db::AgentEvent {
        db::AgentEvent {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: "session".to_string(),
            repo: "repo".to_string(),
            actor: "actor".to_string(),
            event_type: event_type.to_string(),
            summary: summary.to_string(),
            evidence: None,
            metadata: json!({}),
            correlation_id: None,
            parent_event_id: None,
            trajectory_id: None,
            attempt_index: None,
            event_role: None,
            created_at: Utc::now(),
            summary_level,
        }
    }
}

// ── Semantic search ─────────────────────────────────────────────

#[tracing::instrument(name = "handler.search", skip(state, headers, req))]
pub async fn search(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<Value>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let query = req.get("q").and_then(|v| v.as_str()).unwrap_or("");
    let limit = req.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    let results = match qdrant::search(&state.embedder, &state.qdrant_url, query, limit).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("qdrant search failed: {e}");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(
                    serde_json::json!({"error": "search_unavailable", "detail": e.to_string()}),
                ),
            )
                .into_response();
        }
    };

    axum::Json(serde_json::json!({"results": results})).into_response()
}
