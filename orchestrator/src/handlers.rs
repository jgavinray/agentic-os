use axum::extract::State;
use axum::http::StatusCode;
use axum::http::{header, HeaderMap};
use axum::response::IntoResponse;
use axum::response::Response;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;
use std::sync::Arc;
use subtle::ConstantTimeEq;

use crate::anthropic;
use crate::db;
use crate::qdrant;
use crate::rate_limit;
use crate::state::*;
use crate::telemetry;

// ── Auth helpers ───────────────────────────────────────────────

fn bearer_token(headers: &HeaderMap) -> &str {
    headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .unwrap_or("")
}

// Returns the (token, namespace) pair if auth passes; None otherwise.
// Constant-time comparison prevents timing-based key recovery.
fn authenticate(state: &AppState, headers: &HeaderMap) -> Option<(String, String)> {
    let provided = bearer_token(headers).as_bytes();
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

fn check_auth(state: &AppState, headers: &HeaderMap) -> bool {
    authenticate(state, headers).is_some()
}

fn check_rate_limit(state: &AppState, token: &str) -> Option<Response> {
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

// ── Health checks (no auth) ─────────────────────────────────────

#[tracing::instrument(name = "handler.health")]
pub async fn health() -> axum::Json<Value> {
    axum::Json(serde_json::json!({"status": "ok"}))
}

#[tracing::instrument(name = "handler.health_live")]
pub async fn health_live() -> axum::Json<Value> {
    axum::Json(serde_json::json!({"status": "ok"}))
}

#[tracing::instrument(name = "handler.health_ready", skip(state))]
pub async fn health_ready(
    State(state): State<Arc<AppState>>,
) -> Result<axum::Json<Value>, (StatusCode, axum::Json<Value>)> {
    let http = state.http.clone();
    let mut healthy = Vec::new();
    let mut unhealthy = Vec::new();

    if db::check_ready(&state.pool).await.is_ok() {
        healthy.push("postgres");
    } else {
        unhealthy.push("postgres");
    }

    let qdrant_started = std::time::Instant::now();
    if http
        .get(format!("{}/collections", state.qdrant_url))
        .send()
        .await
        .map(|r| {
            let status = r.status();
            telemetry::record_qdrant_request(
                "health",
                qdrant_started.elapsed(),
                &status.as_u16().to_string(),
            );
            status.is_success()
        })
        .unwrap_or(false)
    {
        healthy.push("qdrant");
    } else {
        unhealthy.push("qdrant");
    }

    // LITELLM_URL already includes /v1 — no extra path segment needed here.
    if http
        .get(format!("{}/models", state.litellm_url))
        .send()
        .await
        .map(|r| r.status().is_success() || r.status() == 401)
        .unwrap_or(false)
    {
        healthy.push("litellm");
    } else {
        unhealthy.push("litellm");
    }

    if !unhealthy.is_empty() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(serde_json::json!({
                "status": "degraded",
                "healthy": healthy,
                "unhealthy": unhealthy,
            })),
        ));
    }

    Ok(axum::Json(serde_json::json!({
        "status": "ready",
        "services": healthy,
    })))
}

// ── Model listing — BUG-10: proxy to LiteLLM ───────────────────

#[tracing::instrument(name = "handler.list_models", skip(state, headers))]
pub async fn list_models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let url = format!("{}/models", state.litellm_url);
    let started = std::time::Instant::now();
    match state
        .http
        .get(&url)
        .bearer_auth(&state.litellm_key)
        .send()
        .await
    {
        Ok(r) => {
            let status = r.status();
            telemetry::record_upstream_litellm(
                "models",
                started.elapsed(),
                &status.as_u16().to_string(),
            );
            if !status.is_success() {
                telemetry::record_upstream_litellm_error(
                    "models",
                    telemetry::upstream_error_kind(status),
                );
            }
            match r.json::<Value>().await {
                Ok(v) => (status, axum::Json(v)).into_response(),
                Err(_) => {
                    telemetry::record_upstream_litellm_error("models", "parse");
                    fallback_model_list(&state).into_response()
                }
            }
        }
        Err(e) => {
            telemetry::record_upstream_litellm("models", started.elapsed(), "error");
            telemetry::record_upstream_litellm_error("models", telemetry::reqwest_error_kind(&e));
            fallback_model_list(&state).into_response()
        }
    }
}

fn fallback_model_list(state: &AppState) -> axum::Json<Value> {
    axum::Json(serde_json::json!({
        "data": [{"id": state.default_model, "object": "model", "owned_by": "orchestrator"}],
        "object": "list"
    }))
}

#[tracing::instrument(name = "handler.cache_stats", skip(state, headers))]
pub async fn cache_stats(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    axum::Json(state.cache.stats()).into_response()
}

#[tracing::instrument(name = "handler.metrics", skip(state, headers))]
pub async fn metrics(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    telemetry::record_pool_gauges(&state.pool);
    telemetry::record_process_metrics();
    (
        telemetry::prometheus_content_type(),
        state.prometheus.render(),
    )
        .into_response()
}

#[tracing::instrument(name = "handler.metrics_json", skip(state, headers))]
pub async fn metrics_json(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    axum::Json(state.metrics.snapshot()).into_response()
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

// ── Session management ──────────────────────────────────────────

#[tracing::instrument(name = "handler.start_session", skip(state, headers, req), fields(repo = %req.repo, task = %req.task))]
pub async fn start_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<StartSessionRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let id = match db::start_session_from_request(&state.pool, &req).await {
        Ok(id) => id,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": "failed_to_create_session", "detail": e.to_string()})),
            ).into_response();
        }
    };

    axum::Json(StartSessionResponse { session_id: id }).into_response()
}

// ── Event management ────────────────────────────────────────────

#[tracing::instrument(name = "handler.append_event", skip(state, headers, req), fields(repo = %req.repo, event_type = %req.event_type))]
pub async fn append_event(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(mut req): axum::Json<AppendEventRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }
    if !state.trajectory_capture_enabled {
        req.trajectory_id = None;
        req.attempt_index = None;
        req.event_role = None;
        if let Some(metadata) = req.metadata.as_mut().and_then(Value::as_object_mut) {
            metadata.remove("trajectory_id");
            metadata.remove("attempt_index");
            metadata.remove("event_role");
        }
    }

    let (event_id, qdrant_indexed) =
        match db::append_event_from_request(&state.pool, &state.embedder, &state.qdrant_url, &req)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(
                    serde_json::json!({"error": "failed_to_append_event", "detail": e.to_string()}),
                ),
            )
                .into_response();
            }
        };
    spawn_feature_extraction(&state, &req.repo, &req.session_id, req.trajectory_id);

    // Detect and record errors from the event.
    if let Some(error_type) = &req.error_type {
        let error_desc = req.error_description.as_deref().unwrap_or("");
        let task = req.task.as_deref().unwrap_or("");
        let severity = "medium";
        if let Err(e) = db::insert_error_record(
            &state.pool,
            &req.repo,
            task,
            error_type,
            error_desc,
            severity,
        )
        .await
        {
            tracing::warn!(%error_type, "failed to record error: {e}");
        }
    }

    state
        .cache
        .invalidate(&req.repo, req.task.as_deref().unwrap_or(""));
    telemetry::record_cache_invalidation(&state.metrics);

    axum::Json(serde_json::json!({"event_id": event_id, "qdrant_indexed": qdrant_indexed}))
        .into_response()
}

#[tracing::instrument(name = "handler.validations", skip(state, headers, req), fields(repo = %req.repo, task = %req.task))]
pub async fn validations(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<crate::execution_feedback::ValidationReportRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    if !state.execution_feedback_enabled {
        return axum::Json(crate::execution_feedback::ValidationReportResponse {
            captured: false,
            event_ids: vec![],
            qdrant_indexed: false,
        })
        .into_response();
    }

    let session_id = match &req.session_id {
        Some(session_id) => session_id.clone(),
        None => match db::find_or_create_session(
            &state.pool,
            &req.repo,
            &req.task,
            req.actor.as_deref().unwrap_or("validator"),
        )
        .await
        {
            Ok(session_id) => session_id,
            Err(e) => {
                tracing::error!(
                    target: "execution_feedback",
                    repo = %req.repo,
                    task = %req.task,
                    "failed to create validation session: {e}"
                );
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({
                        "error": "failed_to_create_validation_session",
                        "detail": e.to_string()
                    })),
                )
                    .into_response();
            }
        },
    };

    let ctx = crate::execution_feedback::ExecutionEventContext {
        session_id,
        repo: req.repo.clone(),
        task: req.task.clone(),
        actor: req.actor.clone().unwrap_or_else(|| "validator".to_string()),
        correlation_id: req.correlation_id.unwrap_or_else(uuid::Uuid::new_v4),
        parent_event_id: req.parent_event_id,
        trajectory: if state.trajectory_capture_enabled {
            req.trajectory_id
                .map(|trajectory_id| crate::trajectory::TrajectoryContext {
                    trajectory_id,
                    attempt_index: req.attempt_index.unwrap_or(1),
                })
        } else {
            None
        },
    };
    let events = match crate::execution_feedback::events_for_validation_report(&ctx, &req) {
        Ok(events) => events,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({"error": "invalid_validation_report", "detail": e})),
            )
                .into_response();
        }
    };
    let mut event_ids = Vec::new();
    let mut qdrant_indexed = true;
    for event in events {
        match db::append_execution_event(&state.pool, &state.embedder, &state.qdrant_url, event)
            .await
        {
            Ok((event_id, indexed)) => {
                event_ids.push(event_id);
                qdrant_indexed &= indexed;
            }
            Err(e) => {
                tracing::error!(
                    target: "execution_feedback",
                    repo = %req.repo,
                    task = %req.task,
                    "failed to write validation event: {e}"
                );
            }
        }
    }
    if !event_ids.is_empty() {
        spawn_feature_extraction(
            &state,
            &req.repo,
            &ctx.session_id,
            ctx.trajectory.map(|trajectory| trajectory.trajectory_id),
        );
    }

    state.cache.invalidate(&req.repo, &req.task);
    telemetry::record_cache_invalidation(&state.metrics);

    axum::Json(crate::execution_feedback::ValidationReportResponse {
        captured: !event_ids.is_empty(),
        event_ids,
        qdrant_indexed,
    })
    .into_response()
}

fn spawn_feature_extraction(
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
    tokio::spawn(async move {
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

async fn get_or_build_cached_context(
    state: &AppState,
    repo: &str,
    task: &str,
    session_id: Option<&str>,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    limit_override: Option<i64>,
    task_config: &TaskContextConfig,
) -> Result<CachedContext, anyhow::Error> {
    let event_count = db::count_events_for_repo(&state.pool, repo).await?;
    let cache_task = limit_override
        .map(|limit| format!("{task}:limit={limit}"))
        .unwrap_or_else(|| task.to_string());
    let cache_task = if state.feature_extraction_enabled {
        if let Some(trajectory) = trajectory {
            format!("{cache_task}:trajectory={}", trajectory.trajectory_id)
        } else if let Some(session_id) = session_id {
            format!("{cache_task}:session={session_id}")
        } else {
            cache_task
        }
    } else {
        cache_task
    };
    let cache_key = context_cache_key(repo, &cache_task, event_count);
    if let Some(cached) = state.cache.get(&cache_key) {
        let mut cached = cached;
        cached.stats.cache_hit = true;
        telemetry::record_context_pack(&state.metrics, &cached.stats);
        return Ok(cached);
    }

    let build_started = std::time::Instant::now();
    let mut policy = crate::state::ContextPolicy::for_task(task);
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
        hybrid_search(state, repo, task, task_config.semantic_limit),
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
    stats.cache_hit = false;

    let cached = CachedContext {
        context,
        memories,
        cached_at: std::time::Instant::now(),
        stats,
    };

    state.cache.put(cache_key, cached.clone());
    telemetry::record_context_pack(&state.metrics, &cached.stats);
    Ok(cached)
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

    state
        .cache
        .invalidate(&event.repo, event.task.as_deref().unwrap_or(""));
    telemetry::record_cache_invalidation(&state.metrics);

    axum::Json(serde_json::json!({"event_id": event_id, "qdrant_indexed": qdrant_indexed}))
        .into_response()
}

// ── Chat completions ────────────────────────────────────────────

/// Return an Anthropic-shaped error response.
fn anthropic_error(
    status: StatusCode,
    error_type: &'static str,
    message: impl Into<String>,
) -> Response {
    (
        status,
        axum::Json(serde_json::json!({
            "type": "error",
            "error": {"type": error_type, "message": message.into()}
        })),
    )
        .into_response()
}

/// BUG-4: Append context to an existing client system message rather than inserting
/// a new one at position 0, which would demote the harness's carefully-tuned prompt.
/// Ensure the request asks for at least `MIN_MAX_TOKENS` output tokens.
/// Anthropic requires `max_tokens`; OpenAI treats it as optional. We floor
/// whatever the client sent so reasoning models don't get starved mid-response.
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

fn enforce_min_max_tokens(req: &mut Value) {
    let current = req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    if current < MIN_MAX_TOKENS {
        req["max_tokens"] = Value::from(MIN_MAX_TOKENS);
    }
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

/// Inject context into an Anthropic-format request's system field.
fn inject_system_context_anthropic(payload: &mut Value, context: &str) {
    let existing = match payload.get("system") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|b| {
                if b["type"].as_str() == Some("text") {
                    b["text"].as_str()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    };
    let combined = if existing.is_empty() {
        context.to_string()
    } else {
        format!("{existing}\n\n---\n{context}")
    };
    payload["system"] = Value::String(combined);
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
) -> Option<uuid::Uuid> {
    let task_category = crate::state::TaskCategory::from_task(task);
    let task_config = crate::state::TaskContextConfig::for_category(task_category);
    match get_or_build_cached_context(
        state,
        repo,
        task,
        session_id,
        trajectory,
        None,
        &task_config,
    )
    .await
    {
        Ok(cached) => {
            let context_pack_id = maybe_write_context_pack_event(
                state,
                session_id,
                repo,
                task,
                trajectory,
                parent_event_id,
                &cached.stats,
                task_config.char_budget / 4,
            )
            .await;
            inject_system_context_anthropic(req, &cached.context);
            context_pack_id
        }
        Err(e) => {
            tracing::warn!(repo, task, "failed to build cached context: {e}");
            let context = db::build_context(repo, task, &[], &[], &[], task_config.char_budget);
            inject_system_context_anthropic(req, &context);
            None
        }
    }
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
                if let Err(e) = db::emit_trajectory_result_once(
                    &state.pool,
                    &state.embedder,
                    &state.qdrant_url,
                    trajectory_id,
                    Some(reason),
                )
                .await
                {
                    tracing::warn!(
                        trajectory_id = %trajectory_id,
                        "failed to finalize previous trajectory at boundary: {e}"
                    );
                }
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
    let metadata = crate::trajectory::make_request_metadata(request_metadata, trajectory);
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
    match db::append_event_from_request(&state.pool, &state.embedder, &state.qdrant_url, &req).await
    {
        Ok((event_id, _)) => {
            spawn_feature_extraction(state, repo, session_id, Some(trajectory.trajectory_id));
            uuid::Uuid::parse_str(&event_id).ok()
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

    state.cache.invalidate(repo, "");
    telemetry::record_cache_invalidation(&state.metrics);
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
    tokio::spawn(async move {
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
        state.cache.invalidate(&repo, &task);
        telemetry::record_cache_invalidation(&state.metrics);
    });
}

#[allow(clippy::too_many_arguments)]
async fn maybe_write_context_pack_event(
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
    if let Err(e) =
        db::append_execution_event(&state.pool, &state.embedder, &state.qdrant_url, event).await
    {
        tracing::warn!(
            repo,
            task,
            "failed to write context_pack trajectory event: {e}"
        );
        return None;
    }
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
) -> Option<uuid::Uuid> {
    if req.get("model").is_none() {
        req["model"] = Value::String(state.default_model.clone());
    }
    let task_category = crate::state::TaskCategory::from_task(task);
    let task_config = crate::state::TaskContextConfig::for_category(task_category);
    match get_or_build_cached_context(
        state,
        repo,
        task,
        session_id,
        trajectory,
        None,
        &task_config,
    )
    .await
    {
        Ok(cached) => {
            let context_pack_id = maybe_write_context_pack_event(
                state,
                session_id,
                repo,
                task,
                trajectory,
                parent_event_id,
                &cached.stats,
                task_config.char_budget / 4,
            )
            .await;
            inject_system_context(req, &cached.context);
            context_pack_id
        }
        Err(e) => {
            tracing::warn!(repo, task, "failed to build cached context: {e}");
            let context = db::build_context(repo, task, &[], &[], &[], task_config.char_budget);
            inject_system_context(req, &context);
            None
        }
    }
}

/// POST the OpenAI request to LiteLLM and return the raw response JSON.
/// Returns Err(Response) on network or parse failure.
async fn dispatch_non_streaming_raw(
    state: &AppState,
    openai_req: &Value,
) -> Result<(Value, u64), Response> {
    let url = format!("{}/chat/completions", state.litellm_url);
    let started = std::time::Instant::now();
    match state
        .http
        .post(&url)
        .bearer_auth(&state.litellm_key)
        .json(openai_req)
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
            let latency_ms = started.elapsed().as_millis() as u64;
            r.json::<Value>()
                .await
                .map(|value| (value, latency_ms))
                .map_err(|_| {
                    telemetry::record_upstream_litellm_error("chat_completions", "parse");
                    (
                        StatusCode::BAD_GATEWAY,
                        axum::Json(serde_json::json!({"error": "litellm_parse_error"})),
                    )
                        .into_response()
                })
        }
        Err(e) => {
            telemetry::record_upstream_litellm("chat_completions", started.elapsed(), "error");
            telemetry::record_upstream_litellm_error(
                "chat_completions",
                telemetry::reqwest_error_kind(&e),
            );
            Err((
                StatusCode::BAD_GATEWAY,
                axum::Json(
                    serde_json::json!({"error": "litellm_unreachable", "detail": e.to_string()}),
                ),
            )
                .into_response())
        }
    }
}

#[tracing::instrument(name = "handler.chat_completions", skip(state, headers, payload))]
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(payload): axum::Json<Value>,
) -> Response {
    let Some((caller_token, namespace)) = authenticate(&state, &headers) else {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
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

    let requested_model = payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&state.default_model)
        .to_string();
    let user_content = extract_user_content_openai(&payload);
    let mut req = payload.clone();
    // Always route to the configured backend model regardless of what the client sent.
    req["model"] = Value::String(state.default_model.clone());
    enforce_min_max_tokens(&mut req);
    let sampling_audit = crate::sampling::capture_and_maybe_override(
        &payload,
        &mut req,
        state.sampling_config,
        state.sampling_policy.as_ref(),
    );
    let request_metadata = sampling_audit.as_ref().map(|audit| {
        audit.metadata(
            "/v1/chat/completions",
            &requested_model,
            &state.default_model,
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
    let context_pack_id = pack_context_into_req(
        &state,
        &mut req,
        session_id.as_deref(),
        &repo,
        &task,
        trajectory,
        request_event_id,
    )
    .await;
    let correlation_id = trajectory
        .map(|trajectory| trajectory.trajectory_id)
        .or_else(|| state.execution_feedback_enabled.then(uuid::Uuid::new_v4));

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
        )
        .await;
    }

    // ── Non-streaming ───────────────────────────────────────────
    match dispatch_non_streaming_raw(&state, &req).await {
        Ok((val, latency_ms)) => {
            let usage = TokenUsage::from_openai_value(&val);
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
) -> Response {
    let url = format!("{}/chat/completions", state.litellm_url);

    let started = std::time::Instant::now();
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
            return (
                StatusCode::BAD_GATEWAY,
                axum::Json(
                    serde_json::json!({"error": "litellm_unreachable", "detail": e.to_string()}),
                ),
            )
                .into_response();
        }
    };

    let bytes_stream = upstream.bytes_stream();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Vec<u8>>();
    let accumulated = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let acc_clone = accumulated.clone();

    let tapped = async_stream::stream! {
        tokio::pin!(bytes_stream);
        let mut stream_metrics = telemetry::StreamTracker::new("chat_completions", started);
        let mut tx_opt = Some(done_tx);
        while let Some(chunk) = bytes_stream.next().await {
            match chunk {
                Ok(b) => {
                    stream_metrics.first_token();
                    if let Ok(mut g) = acc_clone.lock() { g.extend_from_slice(&b); }
                    yield Ok::<Bytes, std::io::Error>(b);
                }
                Err(e) => {
                    stream_metrics.fail("upstream_error");
                    yield Err(std::io::Error::other(e));
                    return;
                }
            }
        }
        stream_metrics.finish();
        if let Some(tx) = tx_opt.take() {
            let data = acc_clone.lock().map(|g| g.clone()).unwrap_or_default();
            let _ = tx.send(data);
        }
    };

    let state_bg = state.clone();
    tokio::spawn(async move {
        if let Ok(raw_bytes) = done_rx.await {
            let raw = String::from_utf8_lossy(&raw_bytes);
            let usage = extract_token_usage_from_sse(&raw);
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

#[tracing::instrument(name = "handler.messages", skip(state, headers, payload))]
pub async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(payload): axum::Json<Value>,
) -> Response {
    let Some((caller_token, namespace)) = authenticate(&state, &headers) else {
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

    let user_content = anthropic::extract_user_content_from_anthropic(&payload);
    let model = payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&state.default_model)
        .to_string();

    // Stay in Anthropic format — no translation.
    let mut req = payload;
    req["model"] = Value::String(state.default_model.clone());
    enforce_min_max_tokens(&mut req);
    normalize_response_content_types(&mut req);
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
    let trajectory = if let Some(session_id) = session_id.as_deref() {
        Some(begin_trajectory_for_request(&state, session_id).await)
    } else {
        None
    };
    let request_event_id =
        if let (Some(session_id), Some(trajectory)) = (session_id.as_deref(), trajectory) {
            persist_request_event(&state, session_id, &repo, &user_content, trajectory, None).await
        } else {
            None
        };
    let context_pack_id = pack_context_into_anthropic_req(
        &state,
        &mut req,
        session_id.as_deref(),
        &repo,
        &task,
        trajectory,
        request_event_id,
    )
    .await;
    let correlation_id = trajectory
        .map(|trajectory| trajectory.trajectory_id)
        .or_else(|| state.execution_feedback_enabled.then(uuid::Uuid::new_v4));

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
            session_id,
            trajectory,
            request_event_id,
            context_pack_id,
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
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                "upstream LiteLLM request failed",
            );
        }
    };

    let status = upstream_resp.status();
    let val: Value = match upstream_resp.json().await {
        Ok(v) => v,
        Err(_) => {
            telemetry::record_upstream_litellm_error("messages", "parse");
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                "invalid upstream response",
            );
        }
    };

    if !status.is_success() {
        return (status, axum::Json(val)).into_response();
    }

    let usage = TokenUsage::from_openai_value(&val);
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
            None,
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
    session_id: Option<String>,
    trajectory: Option<crate::trajectory::TrajectoryContext>,
    request_event_id: Option<uuid::Uuid>,
    context_pack_id: Option<uuid::Uuid>,
) -> Response {
    let url = format!("{}/messages", state.litellm_url);

    let started = std::time::Instant::now();
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
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("upstream unreachable: {e}"),
            );
        }
    };

    if !upstream.status().is_success() {
        return anthropic_error(
            StatusCode::BAD_GATEWAY,
            "api_error",
            format!("upstream returned {}", upstream.status()),
        );
    }

    let bytes_stream = upstream.bytes_stream();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Vec<u8>>();
    let accumulated = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let acc_clone = accumulated.clone();

    // Proxy bytes verbatim — LiteLLM already returns Anthropic SSE format.
    let tapped = async_stream::stream! {
        tokio::pin!(bytes_stream);
        let mut stream_metrics = telemetry::StreamTracker::new("messages", started);
        let mut tx_opt = Some(done_tx);
        while let Some(chunk) = bytes_stream.next().await {
            match chunk {
                Ok(b) => {
                    stream_metrics.first_token();
                    if let Ok(mut g) = acc_clone.lock() { g.extend_from_slice(&b); }
                    yield Ok::<Bytes, std::io::Error>(b);
                }
                Err(e) => {
                    stream_metrics.fail("upstream_error");
                    yield Err(std::io::Error::other(e));
                    return;
                }
            }
        }
        stream_metrics.finish();
        if let Some(tx) = tx_opt.take() {
            let data = acc_clone.lock().map(|g| g.clone()).unwrap_or_default();
            let _ = tx.send(data);
        }
    };

    let state_bg = state.clone();
    tokio::spawn(async move {
        if let Ok(raw_bytes) = done_rx.await {
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
                    None,
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
        let ctx_body: String = src[ctx_start..].chars().take(4000).collect();

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
        let ctx_body: String = src[ctx_start..].chars().take(2500).collect();
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

    /// Verify pack_context_into_req is sequential (doesn't use tokio::join!).
    #[test]
    fn pack_context_into_req_is_sequential() {
        let src = include_str!("handlers.rs");
        let pctr_start = src
            .find("async fn pack_context_into_req")
            .expect("pack_context_into_req not found");
        let body = &src[pctr_start..pctr_start + 1500];
        assert!(
            !body.contains("tokio::join!"),
            "pack_context_into_req should NOT use tokio::join!"
        );
    }

    // ── inject_system_context_anthropic ───────────────────────────────

    #[test]
    fn anthropic_inject_sets_system_when_absent() {
        let mut payload = json!({"messages": [{"role": "user", "content": "hi"}]});
        inject_system_context_anthropic(&mut payload, "ctx");
        assert_eq!(payload["system"], "ctx");
    }

    #[test]
    fn anthropic_inject_appends_to_string_system() {
        let mut payload = json!({"system": "base", "messages": []});
        inject_system_context_anthropic(&mut payload, "ctx");
        let sys = payload["system"].as_str().unwrap();
        assert!(sys.contains("base"));
        assert!(sys.contains("ctx"));
    }

    #[test]
    fn anthropic_inject_flattens_array_system_and_appends() {
        let mut payload = json!({
            "system": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}],
            "messages": []
        });
        inject_system_context_anthropic(&mut payload, "ctx");
        let sys = payload["system"].as_str().unwrap();
        assert!(sys.contains("part1"));
        assert!(sys.contains("part2"));
        assert!(sys.contains("ctx"));
        // Result must be a plain string, not an array.
        assert!(payload["system"].is_string());
    }

    #[test]
    fn anthropic_inject_ignores_non_text_system_type() {
        let mut payload = json!({"system": 42, "messages": []});
        inject_system_context_anthropic(&mut payload, "ctx");
        // Non-string/array system treated as absent — context becomes the full value.
        assert_eq!(payload["system"], "ctx");
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
