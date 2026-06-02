use crate::state::ErrorRecord;
use deadpool_postgres::{Config, Pool, PoolConfig};
use tokio_postgres::NoTls;
use uuid::Uuid;

const SINGLE_WRITER_LOCK_KEY: i64 = 0x4167_656e_7469_634f;

// ── Reusable event type ──────────────────────────────────────

#[derive(Clone, Debug)]
pub struct AgentEvent {
    pub id: String,
    pub session_id: String,
    pub repo: String,
    pub actor: String,
    pub event_type: String,
    pub summary: String,
    pub evidence: Option<String>,
    pub metadata: serde_json::Value,
    pub correlation_id: Option<Uuid>,
    pub parent_event_id: Option<Uuid>,
    pub trajectory_id: Option<Uuid>,
    pub attempt_index: Option<i32>,
    pub event_role: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub summary_level: i32,
}

impl AgentEvent {
    pub fn to_memory(&self) -> crate::state::EventMemory {
        crate::state::EventMemory {
            event_type: self.event_type.clone(),
            summary: self.summary.clone(),
            evidence: self.evidence.clone(),
            metadata: self.metadata.clone(),
            created_at: self.created_at,
            summary_level: self.summary_level,
        }
    }

    pub fn payload(&self) -> serde_json::Value {
        serde_json::json!({
            "event_id": self.id,
            "session_id": self.session_id,
            "repo": self.repo,
            "actor": self.actor,
            "event_type": self.event_type,
            "summary": self.summary,
            "evidence": self.evidence,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id.map(|id| id.to_string()),
            "parent_event_id": self.parent_event_id.map(|id| id.to_string()),
            "trajectory_id": self.trajectory_id.map(|id| id.to_string()),
            "attempt_index": self.attempt_index,
            "event_role": self.event_role,
            "created_at": self.created_at,
            "summary_level": self.summary_level
        })
    }

    pub fn vector_text(&self) -> String {
        format!(
            "{}\n{}\n{}\n{}\n{}",
            self.repo,
            self.event_type,
            self.summary,
            self.evidence.as_deref().unwrap_or(""),
            self.metadata
        )
    }
}

#[derive(Default)]
pub struct ContextEvidence {
    pub l0_recent: Vec<AgentEvent>,
    pub l1_matching: Vec<AgentEvent>,
    pub l2_repo: Vec<AgentEvent>,
    pub l3_project: Vec<AgentEvent>,
    pub failures: Vec<AgentEvent>,
    pub failure_history: Vec<FailureHistoryItem>,
    pub operational_constraints: Vec<crate::feature_extraction::OperationalConstraint>,
}

impl ContextEvidence {
    pub fn memories(&self) -> Vec<crate::state::EventMemory> {
        self.l0_recent
            .iter()
            .chain(self.l1_matching.iter())
            .chain(self.l2_repo.iter())
            .chain(self.l3_project.iter())
            .chain(self.failures.iter())
            .chain(
                self.failure_history
                    .iter()
                    .flat_map(|item| std::iter::once(&item.failure).chain(item.remediation.iter())),
            )
            .map(AgentEvent::to_memory)
            .collect()
    }

    pub fn stats(&self) -> crate::state::ContextPackStats {
        crate::state::ContextPackStats {
            l0_items_injected: self.l0_recent.len(),
            l1_items_injected: self.l1_matching.len(),
            l2_items_injected: self.l2_repo.len(),
            l3_items_injected: self.l3_project.len(),
            failed_attempts_injected: self
                .failures
                .iter()
                .filter(|e| e.event_type == "failed_attempt")
                .count(),
            remediations_injected: self
                .failures
                .iter()
                .filter(|e| e.event_type == "remediation")
                .count(),
            failure_history_items_injected: self.failure_history.len(),
            operational_constraints_injected: self.operational_constraints.len(),
            failure_history_remediation_signatures: self
                .failure_history
                .iter()
                .filter_map(|item| item.remediation.as_ref().map(|_| item.signature.clone()))
                .collect(),
            ..Default::default()
        }
    }
}

#[derive(Clone)]
pub struct FailureHistoryItem {
    pub signature: String,
    pub category: String,
    pub failure: AgentEvent,
    pub remediation: Option<AgentEvent>,
}

// ── Connection pool ────────────────────────────────────────────

pub fn create_pool(database_url: &str) -> Result<Pool, anyhow::Error> {
    let mut cfg = Config::new();
    cfg.url = Some(database_url.to_string());
    // BUG-9: assign explicit PoolConfig — the previous code mutated cfg.pool while it was None,
    // so max_size was never applied and the pool used deadpool's unbounded default.
    cfg.pool = Some(PoolConfig::new(16));
    let pool = cfg.create_pool(None, NoTls)?;
    Ok(pool)
}

pub struct SingleWriterGuard {
    conn: deadpool_postgres::Object,
}

pub async fn acquire_single_writer_lock(pool: &Pool) -> Result<SingleWriterGuard, anyhow::Error> {
    let conn = pool.get().await?;
    let acquired: bool = conn
        .query_one(
            "SELECT pg_try_advisory_lock($1)",
            &[&SINGLE_WRITER_LOCK_KEY],
        )
        .await?
        .get(0);
    if !acquired {
        anyhow::bail!(
            "another orchestrator process already owns this Postgres database; \
             single-writer advisory lock {SINGLE_WRITER_LOCK_KEY} is held"
        );
    }
    tracing::info!(
        target: "db",
        lock_key = SINGLE_WRITER_LOCK_KEY,
        "acquired single-writer advisory lock"
    );
    Ok(SingleWriterGuard { conn })
}

impl SingleWriterGuard {
    pub async fn release(self) {
        match self
            .conn
            .execute("SELECT pg_advisory_unlock($1)", &[&SINGLE_WRITER_LOCK_KEY])
            .await
        {
            Ok(_) => tracing::info!(
                target: "db",
                lock_key = SINGLE_WRITER_LOCK_KEY,
                "released single-writer advisory lock"
            ),
            Err(e) => tracing::warn!(
                target: "db",
                lock_key = SINGLE_WRITER_LOCK_KEY,
                "failed to release single-writer advisory lock: {e}"
            ),
        }
    }
}

pub async fn record_token_usage(
    pool: &Pool,
    requested_model: &str,
    actual_model: &str,
    namespace: &str,
    repo: &str,
    usage: &crate::state::TokenUsage,
) -> Result<(), anyhow::Error> {
    let started = std::time::Instant::now();
    let result: Result<(), anyhow::Error> = async {
        let conn = pool.get().await?;
        conn.execute(
            "INSERT INTO token_usage \
             (requested_model, actual_model, namespace, repo, processed_tokens, cached_tokens, generated_tokens) \
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
            &[
                &requested_model,
                &actual_model,
                &namespace,
                &repo,
                &(usage.processed_tokens as i64),
                &(usage.cached_tokens as i64),
                &(usage.generated_tokens as i64),
            ],
        )
        .await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("record_token_usage", started.elapsed(), result.is_ok());
    result
}

// ── DB query functions ────────────────────────────────────────

pub async fn create_session(
    pool: &Pool,
    session_id: &str,
    repo: &str,
    task: &str,
    actor: &str,
) -> Result<(), anyhow::Error> {
    let started = std::time::Instant::now();
    let result: Result<(), anyhow::Error> = async {
        let conn = pool.get().await?;
        conn.execute(
            "INSERT INTO agent_sessions (id, repo, task, actor) VALUES ($1, $2, $3, $4)",
            &[&session_id, &repo, &task, &actor],
        )
        .await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("create_session", started.elapsed(), result.is_ok());
    result
}

/// BUG-2: Find an existing session for (repo, task) from the last 4 hours, or create one.
/// Prevents callers from needing a separate /sessions/start call for every completion.
pub async fn find_or_create_session(
    pool: &Pool,
    repo: &str,
    task: &str,
    actor: &str,
) -> Result<String, anyhow::Error> {
    let started = std::time::Instant::now();
    let result: Result<String, anyhow::Error> = async {
        let conn = pool.get().await?;
        let row = conn
            .query_opt(
                "SELECT id FROM agent_sessions
                 WHERE repo = $1 AND task = $2
                   AND created_at > now() - interval '4 hours'
                 ORDER BY created_at DESC LIMIT 1",
                &[&repo, &task],
            )
            .await?;

        if let Some(r) = row {
            return Ok(r.get("id"));
        }

        let id = Uuid::new_v4().to_string();
        conn.execute(
            "INSERT INTO agent_sessions (id, repo, task, actor) VALUES ($1, $2, $3, $4)",
            &[&id, &repo, &task, &actor],
        )
        .await?;
        Ok(id)
    }
    .await;
    crate::telemetry::record_db_query("find_or_create_session", started.elapsed(), result.is_ok());
    result
}

pub async fn insert_event(pool: &Pool, event: &AgentEvent) -> Result<(), anyhow::Error> {
    crate::trajectory::validate_event_role(event.event_role.as_deref())?;
    if matches!(event.attempt_index, Some(attempt) if attempt < 1) {
        anyhow::bail!("attempt_index must be positive");
    }
    let lineage_fields = [
        event.trajectory_id.is_some(),
        event.attempt_index.is_some(),
        event.event_role.is_some(),
    ];
    if lineage_fields.iter().any(|present| *present)
        && !lineage_fields.iter().all(|present| *present)
    {
        anyhow::bail!("trajectory_id, attempt_index, and event_role must be written together");
    }
    let metadata = crate::harness_feedback::annotate_event_metadata(
        &event.event_type,
        &event.summary,
        event.evidence.as_deref(),
        event.metadata.clone(),
    );
    let metadata = crate::feature_extraction::annotate_event_metadata(
        &event.event_type,
        &event.summary,
        event.evidence.as_deref(),
        metadata,
    );
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        conn.execute(
            "INSERT INTO agent_events
             (id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, summary_level)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)",
            &[
                &event.id,
                &event.session_id,
                &event.repo,
                &event.actor,
                &event.event_type,
                &event.summary,
                &event.evidence,
                &metadata,
                &event.correlation_id,
                &event.parent_event_id,
                &event.trajectory_id,
                &event.attempt_index,
                &event.event_role,
                &event.summary_level,
            ],
        )
        .await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("insert_event", started.elapsed(), result.is_ok());
    if result.is_ok() {
        crate::harness_feedback::record_metadata_metrics(&metadata);
        let classification_event = AgentEvent {
            metadata,
            ..event.clone()
        };
        if let Err(e) =
            crate::request_classification::classify_and_persist_event(pool, &classification_event)
                .await
        {
            tracing::warn!(
                target: "request_classification",
                event_id = %event.id,
                event_type = %event.event_type,
                "failed to persist request classification for new event: {e}"
            );
        }
        tracing::info!(
            target: "execution_feedback",
            event_type = %event.event_type,
            event_id = %event.id,
            correlation_id = ?event.correlation_id,
            "event written"
        );
    }
    result
}

pub async fn upsert_context_artifact(
    pool: &Pool,
    artifact: &crate::context_artifacts::ContextArtifact,
) -> Result<(), anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        conn.execute(
            "INSERT INTO context_artifacts
             (id, repo, scope, artifact_type, status, content_raw, content_compact,
              content_rendered, content_hash, invalidation_key, source_event_ids,
              source_file_paths, token_estimate)
             VALUES
             ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
             ON CONFLICT (repo, scope, artifact_type, invalidation_key) DO UPDATE
             SET status = EXCLUDED.status,
                 content_raw = EXCLUDED.content_raw,
                 content_compact = EXCLUDED.content_compact,
                 content_rendered = EXCLUDED.content_rendered,
                 content_hash = EXCLUDED.content_hash,
                 source_event_ids = EXCLUDED.source_event_ids,
                 source_file_paths = EXCLUDED.source_file_paths,
                 token_estimate = EXCLUDED.token_estimate,
                 updated_at = now()",
            &[
                &artifact.id,
                &artifact.repo,
                &artifact.scope,
                &artifact.artifact_type,
                &artifact.status,
                &artifact.content_raw,
                &artifact.content_compact,
                &artifact.content_rendered,
                &artifact.content_hash,
                &artifact.invalidation_key,
                &artifact.source_event_ids,
                &artifact.source_file_paths,
                &artifact.token_estimate,
            ],
        )
        .await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("upsert_context_artifact", started.elapsed(), result.is_ok());
    result
}

pub async fn get_active_context_artifacts(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<crate::context_artifacts::ContextArtifact>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, repo, scope, artifact_type, status, content_raw, content_compact,
                        content_rendered, content_hash, invalidation_key, source_event_ids,
                        source_file_paths, token_estimate
                 FROM context_artifacts
                 WHERE repo = $1
                   AND status = 'active'
                 ORDER BY
                   CASE artifact_type
                     WHEN 'service_topology' THEN 0
                     WHEN 'active_instruction' THEN 1
                     WHEN 'repo_map' THEN 2
                     WHEN 'durable_project_memory' THEN 3
                     WHEN 'repo_decisions' THEN 4
                     WHEN 'failure_history' THEN 5
                     ELSE 99
                   END,
                   updated_at DESC
                 LIMIT $2",
                &[&repo, &limit],
            )
            .await?;

        Ok(rows
            .into_iter()
            .map(|row| crate::context_artifacts::ContextArtifact {
                id: row.get("id"),
                repo: row.get("repo"),
                scope: row.get("scope"),
                artifact_type: row.get("artifact_type"),
                status: row.get("status"),
                content_raw: row.get("content_raw"),
                content_compact: row.get("content_compact"),
                content_rendered: row.get("content_rendered"),
                content_hash: row.get("content_hash"),
                invalidation_key: row.get("invalidation_key"),
                source_event_ids: row.get("source_event_ids"),
                source_file_paths: row.get("source_file_paths"),
                token_estimate: row.get("token_estimate"),
            })
            .collect())
    }
    .await;
    crate::telemetry::record_db_query(
        "get_active_context_artifacts",
        started.elapsed(),
        result.is_ok(),
    );
    result
}

pub async fn get_recent_instruction_candidates(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let instruction_patterns: &[&str] = &[
            "%explicit%",
            "%instruction%",
            "%must%",
            "%need%",
            "%should%",
            "%do not%",
            "%don't%",
            "%never%",
            "%stop%",
            "%default%",
            "%config%",
        ];
        let rows = conn
            .query(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata,
                        correlation_id, parent_event_id, trajectory_id, attempt_index,
                        event_role, created_at, summary_level
                 FROM agent_events
                 WHERE repo = $1
                   AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
                   AND (
                     event_role = 'user'
                     OR actor = 'user'
                     OR event_type IN ('user_message', 'instruction', 'preference')
                   )
                   AND (
                     summary ILIKE ANY($2)
                     OR coalesce(evidence, '') ILIKE ANY($2)
                   )
                 ORDER BY created_at DESC
                 LIMIT $3",
                &[&repo, &instruction_patterns, &limit],
            )
            .await?;

        Ok(rows
            .into_iter()
            .map(|row| AgentEvent {
                id: row.get("id"),
                session_id: row.get("session_id"),
                repo: row.get("repo"),
                actor: row.get("actor"),
                event_type: row.get("event_type"),
                summary: row.get("summary"),
                evidence: row.get("evidence"),
                metadata: row.get("metadata"),
                correlation_id: row.get("correlation_id"),
                parent_event_id: row.get("parent_event_id"),
                trajectory_id: row.get("trajectory_id"),
                attempt_index: row.get("attempt_index"),
                event_role: row.get("event_role"),
                created_at: row.get("created_at"),
                summary_level: row.get("summary_level"),
            })
            .collect())
    }
    .await;
    crate::telemetry::record_db_query(
        "get_recent_instruction_candidates",
        started.elapsed(),
        result.is_ok(),
    );
    result
}

/// Store an execution artifact through the same event log and best-effort
/// vector indexing path used by ordinary memory events.
pub async fn append_execution_event(
    pool: &Pool,
    embedder: &crate::embedder::Embedder,
    qdrant_url: &str,
    event: AgentEvent,
) -> Result<(String, bool), anyhow::Error> {
    let id = insert_execution_artifact_event(pool, &event).await?;
    let qdrant_indexed = match crate::qdrant::store_event(embedder, qdrant_url, &event).await {
        Ok(_) => true,
        Err(e) => {
            tracing::warn!(
                event_id = %id,
                event_type = %event.event_type,
                "execution event stored in postgres but qdrant indexing failed: {e}"
            );
            false
        }
    };
    Ok((id, qdrant_indexed))
}

pub async fn insert_execution_artifact_event(
    pool: &Pool,
    event: &AgentEvent,
) -> Result<String, anyhow::Error> {
    insert_event(pool, event).await?;
    // Metrics are updated only after the DB write succeeds so counters describe
    // persisted outcome memory, not attempted capture.
    crate::telemetry::record_execution_artifact(event);
    Ok(event.id.clone())
}

#[allow(dead_code)]
pub async fn insert_tool_result_event(
    pool: &Pool,
    ctx: &crate::execution_feedback::ExecutionEventContext,
    result: &crate::execution_feedback::CapturedToolResult,
) -> Result<String, anyhow::Error> {
    let event = crate::execution_feedback::build_execution_event(
        ctx,
        crate::execution_feedback::ExecutionEventKind::ToolResult,
        result.exit_code == 0,
        crate::execution_feedback::tool_result_payload(result),
    );
    insert_execution_artifact_event(pool, &event).await
}

#[allow(dead_code)]
pub async fn insert_compile_result_event(
    pool: &Pool,
    ctx: &crate::execution_feedback::ExecutionEventContext,
    language: &str,
    target: &str,
    exit_code: i32,
    content: &str,
) -> Result<String, anyhow::Error> {
    let payload =
        crate::execution_feedback::compile_result_payload(language, target, exit_code, content);
    let success = exit_code == 0 && payload["error_count"].as_u64().unwrap_or_default() == 0;
    let event = crate::execution_feedback::build_execution_event(
        ctx,
        crate::execution_feedback::ExecutionEventKind::CompileResult,
        success,
        payload,
    );
    insert_execution_artifact_event(pool, &event).await
}

#[allow(dead_code)]
pub async fn insert_test_result_event(
    pool: &Pool,
    ctx: &crate::execution_feedback::ExecutionEventContext,
    framework: &str,
    content: &str,
) -> Result<String, anyhow::Error> {
    let payload = crate::execution_feedback::test_result_payload(framework, content);
    let success = payload["failed"].as_u64().unwrap_or_default() == 0;
    let event = crate::execution_feedback::build_execution_event(
        ctx,
        crate::execution_feedback::ExecutionEventKind::TestResult,
        success,
        payload,
    );
    insert_execution_artifact_event(pool, &event).await
}

#[allow(dead_code)]
pub async fn insert_lint_result_event(
    pool: &Pool,
    ctx: &crate::execution_feedback::ExecutionEventContext,
    tool_name: &str,
    content: &str,
) -> Result<String, anyhow::Error> {
    let payload = crate::execution_feedback::lint_result_payload(tool_name, content);
    let success = payload["error_count"].as_u64().unwrap_or_default() == 0;
    let event = crate::execution_feedback::build_execution_event(
        ctx,
        crate::execution_feedback::ExecutionEventKind::LintResult,
        success,
        payload,
    );
    insert_execution_artifact_event(pool, &event).await
}

#[allow(dead_code)]
pub async fn insert_validation_result_event(
    pool: &Pool,
    ctx: &crate::execution_feedback::ExecutionEventContext,
    validator_name: &str,
    pass: bool,
    failure_reason: &str,
) -> Result<String, anyhow::Error> {
    let event = crate::execution_feedback::build_execution_event(
        ctx,
        crate::execution_feedback::ExecutionEventKind::ValidationResult,
        pass,
        crate::execution_feedback::validation_result_payload(validator_name, pass, failure_reason),
    );
    insert_execution_artifact_event(pool, &event).await
}

#[allow(dead_code)]
pub async fn insert_patch_result_event(
    pool: &Pool,
    ctx: &crate::execution_feedback::ExecutionEventContext,
    files_touched: Vec<String>,
    outcome: &str,
    validation_event_ids: Vec<Uuid>,
) -> Result<String, anyhow::Error> {
    let event = crate::execution_feedback::build_execution_event(
        ctx,
        crate::execution_feedback::ExecutionEventKind::PatchResult,
        outcome == "applied",
        crate::execution_feedback::patch_result_payload(
            files_touched,
            outcome,
            validation_event_ids,
        ),
    );
    insert_execution_artifact_event(pool, &event).await
}

#[allow(dead_code)]
pub async fn insert_remediation_event(
    pool: &Pool,
    ctx: &crate::execution_feedback::ExecutionEventContext,
    failure_event_id: Uuid,
    resolving_event_id: Uuid,
    signature: &str,
) -> Result<String, anyhow::Error> {
    let event = crate::execution_feedback::build_execution_event(
        ctx,
        crate::execution_feedback::ExecutionEventKind::Remediation,
        true,
        crate::execution_feedback::remediation_payload(
            failure_event_id,
            resolving_event_id,
            signature,
        ),
    );
    insert_execution_artifact_event(pool, &event).await
}

#[allow(dead_code)]
pub async fn get_events_for_repo(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let conn = pool.get().await?;
    let rows = conn
        .query(
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
             FROM agent_events
             WHERE repo = $1
               AND summarized = false
               AND summary_level = 0
               AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
             ORDER BY created_at DESC
             LIMIT $2",
            &[&repo, &limit],
        )
        .await?;

    let events = rows
        .iter()
        .map(|row| AgentEvent {
            id: row.get("id"),
            session_id: row.get("session_id"),
            repo: row.get("repo"),
            actor: row.get("actor"),
            event_type: row.get("event_type"),
            summary: row.get("summary"),
            evidence: row.get("evidence"),
            metadata: row.get("metadata"),
            correlation_id: row.get("correlation_id"),
            parent_event_id: row.get("parent_event_id"),
            trajectory_id: row.get("trajectory_id"),
            attempt_index: row.get("attempt_index"),
            event_role: row.get("event_role"),
            created_at: row.get("created_at"),
            summary_level: row.get("summary_level"),
        })
        .collect();

    Ok(events)
}

pub async fn count_events_for_repo(pool: &Pool, repo: &str) -> Result<i64, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let row = conn
            .query_one(
                "SELECT count(*)::BIGINT AS count
                 FROM agent_events
                 WHERE repo = $1
                   AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'",
                &[&repo],
            )
            .await?;
        Ok(row.get("count"))
    }
    .await;
    crate::telemetry::record_db_query("count_events_for_repo", started.elapsed(), result.is_ok());
    result
}

pub fn preferred_summary_levels(event_count: i64) -> Vec<i32> {
    match event_count {
        n if n < 20 => vec![0],
        n if n < 200 => vec![1],
        n if n < 2000 => vec![2],
        _ => vec![3, 2],
    }
}

#[allow(dead_code)]
pub async fn get_context_events_for_repo(
    pool: &Pool,
    repo: &str,
    event_count: i64,
    limit: i64,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let levels = preferred_summary_levels(event_count);
    let mut events = get_events_for_repo_by_levels(pool, repo, &levels, limit).await?;

    // Fresh or partially migrated repositories may not yet have promoted summaries.
    // Fall back through lower levels so context remains available while the
    // background summarizer catches up.
    if events.is_empty() && !levels.contains(&0) {
        events = get_events_for_repo_by_levels(pool, repo, &[1, 0], limit).await?;
    }
    if events.is_empty() {
        events = get_events_for_repo(pool, repo, limit).await?;
    }

    Ok(events)
}

pub async fn get_context_evidence_for_policy(
    pool: &Pool,
    repo: &str,
    policy: &crate::state::ContextPolicy,
) -> Result<ContextEvidence, anyhow::Error> {
    let started = std::time::Instant::now();
    let (l0_recent, l1_matching, l2_repo, l3_project, failures) = tokio::join!(
        get_events_for_repo_by_level(
            pool,
            repo,
            crate::state::MemoryLevel::L0,
            policy.l0_recent_limit
        ),
        get_events_for_repo_by_level(pool, repo, crate::state::MemoryLevel::L1, policy.l1_limit),
        get_events_for_repo_by_level(pool, repo, crate::state::MemoryLevel::L2, policy.l2_limit),
        get_events_for_repo_by_level(pool, repo, crate::state::MemoryLevel::L3, policy.l3_limit),
        get_failure_events_for_repo(pool, repo, policy.failure_limit),
    );

    let result: Result<ContextEvidence, anyhow::Error> = Ok(ContextEvidence {
        l0_recent: l0_recent?,
        l1_matching: l1_matching?,
        l2_repo: l2_repo?,
        l3_project: l3_project?,
        failures: failures?,
        failure_history: vec![],
        operational_constraints: vec![],
    });
    crate::telemetry::record_db_query("get_context_evidence", started.elapsed(), result.is_ok());
    result
}

async fn get_events_for_repo_by_level(
    pool: &Pool,
    repo: &str,
    level: crate::state::MemoryLevel,
    limit: i64,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    if limit <= 0 {
        return Ok(vec![]);
    }

    let conn = pool.get().await?;
    let level = level.as_i32();
    let rows = conn
        .query(
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
             FROM agent_events
             WHERE repo = $1
               AND summary_level = $2
               AND event_type NOT IN ('failed_attempt', 'remediation')
               AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
             ORDER BY created_at DESC
             LIMIT $3",
            &[&repo, &level, &limit],
        )
        .await?;

    Ok(rows_to_events(rows))
}

async fn get_failure_events_for_repo(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    if limit <= 0 {
        return Ok(vec![]);
    }

    let conn = pool.get().await?;
    let rows = conn
        .query(
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
             FROM agent_events
             WHERE repo = $1
               AND event_type IN ('failed_attempt', 'remediation')
               AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
             ORDER BY created_at DESC
             LIMIT $2",
            &[&repo, &limit],
        )
        .await?;

    Ok(rows_to_events(rows))
}

#[allow(dead_code)]
async fn get_events_for_repo_by_levels(
    pool: &Pool,
    repo: &str,
    levels: &[i32],
    limit: i64,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let conn = pool.get().await?;
    let rows = conn
        .query(
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
             FROM agent_events
             WHERE repo = $1
               AND summarized = false
               AND summary_level = ANY($2)
               AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
             ORDER BY summary_level DESC, created_at DESC
             LIMIT $3",
            &[&repo, &levels, &limit],
        )
        .await?;

    Ok(rows_to_events(rows))
}

fn rows_to_events(rows: Vec<tokio_postgres::Row>) -> Vec<AgentEvent> {
    rows.into_iter()
        .map(|row| AgentEvent {
            id: row.get("id"),
            session_id: row.get("session_id"),
            repo: row.get("repo"),
            actor: row.get("actor"),
            event_type: row.get("event_type"),
            summary: row.get("summary"),
            evidence: row.get("evidence"),
            metadata: row.get("metadata"),
            correlation_id: row.get("correlation_id"),
            parent_event_id: row.get("parent_event_id"),
            trajectory_id: row.get("trajectory_id"),
            attempt_index: row.get("attempt_index"),
            event_role: row.get("event_role"),
            created_at: row.get("created_at"),
            summary_level: row.get("summary_level"),
        })
        .collect()
}

/// Full-text search on agent_events.summary and evidence for a given repo.
/// Returns results ordered by ts_rank DESC.
pub async fn search_events_fts(
    pool: &Pool,
    repo: &str,
    query: &str,
    limit: i64,
) -> Result<Vec<crate::state::SearchHit>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "WITH docs AS (
                     SELECT id, event_type, summary, created_at,
                            to_tsvector('english', coalesce(summary, '') || ' ' || coalesce(evidence, '')) AS tsv
                     FROM agent_events
                     WHERE repo = $1
                       AND summarized = false
                       AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
                 )
                 SELECT id, event_type, summary, created_at
                 FROM docs
                 WHERE tsv @@ plainto_tsquery('english', $2)
                 ORDER BY ts_rank(tsv, plainto_tsquery('english', $2)) DESC
                 LIMIT $3",
                &[&repo, &query, &limit],
            )
            .await?;

        Ok(rows
            .into_iter()
            .map(|row| crate::state::SearchHit {
                event_id: row.get("id"),
                event_type: row.get("event_type"),
                summary: row.get("summary"),
                created_at: Some(row.get("created_at")),
            })
            .collect())
    }
    .await;
    crate::telemetry::record_db_query("search_events_fts", started.elapsed(), result.is_ok());
    result
}

pub async fn hydrate_active_search_hits(
    pool: &Pool,
    repo: &str,
    hits: Vec<crate::state::SearchHit>,
) -> Result<Vec<crate::state::SearchHit>, anyhow::Error> {
    if hits.is_empty() {
        return Ok(hits);
    }

    let started = std::time::Instant::now();
    let result = async {
        let ids: Vec<String> = hits.iter().map(|h| h.event_id.clone()).collect();
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, event_type, summary, created_at
                 FROM agent_events
                 WHERE repo = $1
                   AND summarized = false
                   AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
                   AND id = ANY($2)",
                &[&repo, &ids],
            )
            .await?;

        let by_id: std::collections::HashMap<String, crate::state::SearchHit> = rows
            .into_iter()
            .map(|row| {
                let id: String = row.get("id");
                (
                    id.clone(),
                    crate::state::SearchHit {
                        event_id: id,
                        event_type: row.get("event_type"),
                        summary: row.get("summary"),
                        created_at: Some(row.get("created_at")),
                    },
                )
            })
            .collect();

        Ok(hits
            .into_iter()
            .filter_map(|hit| by_id.get(&hit.event_id).cloned())
            .collect())
    }
    .await;
    crate::telemetry::record_db_query(
        "hydrate_active_search_hits",
        started.elapsed(),
        result.is_ok(),
    );
    result
}

pub async fn get_event_chain_by_event_id(
    pool: &Pool,
    event_id: &str,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let seed = get_event_by_id(pool, event_id).await?;
        let Some(correlation_id) = seed.correlation_id else {
            // Historical rows and non-chain events remain valid. With no chain
            // ID there is nothing to reconstruct, so return the seed by itself.
            return Ok(vec![seed]);
        };
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
                 FROM agent_events
                 WHERE correlation_id = $1
                 ORDER BY created_at ASC, id ASC",
                &[&correlation_id],
            )
            .await?;
        Ok(order_event_chain(rows_to_events(rows), event_id))
    }
    .await;
    crate::telemetry::record_db_query("get_event_chain", started.elapsed(), result.is_ok());
    result
}

async fn get_event_by_id(pool: &Pool, event_id: &str) -> Result<AgentEvent, anyhow::Error> {
    let conn = pool.get().await?;
    let row = conn
        .query_one(
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
             FROM agent_events
             WHERE id = $1",
            &[&event_id],
        )
        .await?;
    Ok(rows_to_events(vec![row]).remove(0))
}

pub async fn get_trajectory(
    pool: &Pool,
    trajectory_id: Uuid,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
                 FROM agent_events
                 WHERE trajectory_id = $1
                 ORDER BY attempt_index ASC NULLS LAST, created_at ASC, id ASC",
                &[&trajectory_id],
            )
            .await?;
        Ok(rows_to_events(rows))
    }
    .await;
    crate::telemetry::record_db_query("get_trajectory", started.elapsed(), result.is_ok());
    result
}

pub async fn get_trajectory_attempts(
    pool: &Pool,
    trajectory_id: Uuid,
) -> Result<Vec<i32>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT DISTINCT attempt_index
                 FROM agent_events
                 WHERE trajectory_id = $1
                   AND attempt_index IS NOT NULL
                 ORDER BY attempt_index ASC",
                &[&trajectory_id],
            )
            .await?;
        Ok(rows
            .into_iter()
            .map(|row| row.get("attempt_index"))
            .collect())
    }
    .await;
    crate::telemetry::record_db_query("get_trajectory_attempts", started.elapsed(), result.is_ok());
    result
}

pub async fn get_trajectory_result(
    pool: &Pool,
    trajectory_id: Uuid,
) -> Result<Option<AgentEvent>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let row = conn
            .query_opt(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
                 FROM agent_events
                 WHERE trajectory_id = $1
                   AND event_role = 'trajectory_result'
                 ORDER BY created_at ASC
                 LIMIT 1",
                &[&trajectory_id],
            )
            .await?;
        Ok(row.map(|row| rows_to_events(vec![row]).remove(0)))
    }
    .await;
    crate::telemetry::record_db_query("get_trajectory_result", started.elapsed(), result.is_ok());
    result
}

pub async fn latest_trajectory_event_for_session(
    pool: &Pool,
    session_id: &str,
) -> Result<Option<AgentEvent>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let row = conn
            .query_opt(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
                 FROM agent_events
                 WHERE session_id = $1
                   AND trajectory_id IS NOT NULL
                 ORDER BY created_at DESC, id DESC
                 LIMIT 1",
                &[&session_id],
            )
            .await?;
        Ok(row.map(|row| rows_to_events(vec![row]).remove(0)))
    }
    .await;
    crate::telemetry::record_db_query(
        "latest_trajectory_event_for_session",
        started.elapsed(),
        result.is_ok(),
    );
    result
}

pub async fn idle_trajectory_ids(
    pool: &Pool,
    idle_timeout_sec: u64,
    limit: i64,
) -> Result<Vec<Uuid>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "WITH latest AS (
                    SELECT trajectory_id, max(created_at) AS last_event_at
                    FROM agent_events
                    WHERE trajectory_id IS NOT NULL
                    GROUP BY trajectory_id
                )
                SELECT latest.trajectory_id
                FROM latest
                WHERE latest.last_event_at < now() - ($1::text || ' seconds')::interval
                  AND NOT EXISTS (
                      SELECT 1
                      FROM agent_events result
                      WHERE result.trajectory_id = latest.trajectory_id
                        AND result.event_role = 'trajectory_result'
                  )
                ORDER BY latest.last_event_at ASC
                LIMIT $2",
                &[&idle_timeout_sec.to_string(), &limit],
            )
            .await?;
        Ok(rows
            .into_iter()
            .map(|row| row.get("trajectory_id"))
            .collect())
    }
    .await;
    crate::telemetry::record_db_query("idle_trajectory_ids", started.elapsed(), result.is_ok());
    result
}

pub async fn emit_trajectory_result_once(
    pool: &Pool,
    embedder: &crate::embedder::Embedder,
    qdrant_url: &str,
    trajectory_id: Uuid,
    boundary_reason: Option<crate::trajectory::BoundaryReason>,
) -> Result<bool, anyhow::Error> {
    let events = get_trajectory(pool, trajectory_id).await?;
    let source_events: Vec<_> = events
        .into_iter()
        .filter(|event| {
            event.event_role.as_deref()
                != Some(crate::trajectory::EventRole::TrajectoryResult.as_str())
        })
        .collect();
    let Some(first_event) = source_events.first() else {
        return Ok(false);
    };
    let summary =
        crate::trajectory::summarize_trajectory(trajectory_id, &source_events, boundary_reason);
    let result_event = crate::trajectory::trajectory_result_event(
        &first_event.session_id,
        &first_event.repo,
        summary.clone(),
    );
    let inserted = insert_trajectory_result_event(pool, &result_event).await?;
    if inserted {
        crate::telemetry::record_trajectory_result(&summary);
        let _ = crate::qdrant::store_event(embedder, qdrant_url, &result_event)
            .await
            .map_err(|e| {
                tracing::warn!(
                    trajectory_id = %trajectory_id,
                    "trajectory result stored in postgres but qdrant indexing failed: {e}"
                );
                e
            });
    }
    Ok(inserted)
}

async fn insert_trajectory_result_event(
    pool: &Pool,
    event: &AgentEvent,
) -> Result<bool, anyhow::Error> {
    crate::trajectory::validate_event_role(event.event_role.as_deref())?;
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .execute(
                "INSERT INTO agent_events
                 (id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, summary_level)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                 ON CONFLICT DO NOTHING",
                &[
                    &event.id,
                    &event.session_id,
                    &event.repo,
                    &event.actor,
                    &event.event_type,
                    &event.summary,
                    &event.evidence,
                    &event.metadata,
                    &event.correlation_id,
                    &event.parent_event_id,
                    &event.trajectory_id,
                    &event.attempt_index,
                    &event.event_role,
                    &event.summary_level,
                ],
            )
            .await?;
        Ok(rows == 1)
    }
    .await;
    crate::telemetry::record_db_query(
        "insert_trajectory_result_event",
        started.elapsed(),
        result.is_ok(),
    );
    result
}

pub fn order_event_chain(events: Vec<AgentEvent>, seed_id: &str) -> Vec<AgentEvent> {
    if events.is_empty() {
        return vec![];
    }
    let by_id: std::collections::HashMap<String, AgentEvent> = events
        .iter()
        .map(|event| (event.id.clone(), event.clone()))
        .collect();
    let Some(seed) = by_id.get(seed_id).cloned() else {
        return events;
    };

    // Walk parent pointers upward to find the root visible in this correlation
    // group. The guard prevents malformed cycles from trapping reconstruction.
    let mut root = seed.clone();
    let mut guard = std::collections::HashSet::new();
    while let Some(parent_id) = root.parent_event_id {
        if !guard.insert(root.id.clone()) {
            break;
        }
        let parent_key = parent_id.to_string();
        let Some(parent) = by_id.get(&parent_key) else {
            break;
        };
        root = parent.clone();
    }

    let grouped = crate::execution_feedback::group_by_parent(&events);
    let mut ordered = Vec::new();
    // Then emit a stable depth-first tree so callers see request -> response ->
    // tools -> validation -> patch/remediation in human-readable order.
    append_chain_tree(&root, &grouped, &mut ordered);
    ordered
}

fn append_chain_tree(
    event: &AgentEvent,
    grouped: &std::collections::BTreeMap<Option<Uuid>, Vec<AgentEvent>>,
    ordered: &mut Vec<AgentEvent>,
) {
    ordered.push(event.clone());
    let Ok(event_uuid) = Uuid::parse_str(&event.id) else {
        return;
    };
    if let Some(children) = grouped.get(&Some(event_uuid)) {
        for child in children {
            append_chain_tree(child, grouped, ordered);
        }
    }
}

pub async fn get_failure_history_for_signatures(
    pool: &Pool,
    repo: &str,
    signatures: &[String],
    limit: i64,
) -> Result<Vec<FailureHistoryItem>, anyhow::Error> {
    if signatures.is_empty() || limit <= 0 {
        return Ok(vec![]);
    }

    let started = std::time::Instant::now();
    let result = async {
        let outcome_event_types = crate::execution_feedback::FAILURE_OUTCOME_EVENT_TYPES.to_vec();
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
                 FROM agent_events
                 WHERE repo = $1
                   AND event_type = ANY($2)
                   AND metadata->>'success' = 'false'
                   AND metadata->'payload'->>'signature' = ANY($3)
                 ORDER BY created_at DESC
                 LIMIT $4",
                &[&repo, &outcome_event_types, &signatures, &limit],
            )
            .await?;

        let failures = rows_to_events(rows);
        let mut items = Vec::new();
        for failure in failures {
            let signature = event_payload_str(&failure, "signature").unwrap_or("unknown");
            let category = event_payload_str(&failure, "signature_category").unwrap_or("unknown");
            // The lookup is exact on canonical signature. Any semantic recall for
            // execution events continues to use the existing hybrid pipeline.
            let remediation = remediation_for_failure(pool, &failure, signature).await?;
            items.push(FailureHistoryItem {
                signature: signature.to_string(),
                category: category.to_string(),
                failure,
                remediation,
            });
        }
        Ok(items)
    }
    .await;
    crate::telemetry::record_db_query("get_failure_history", started.elapsed(), result.is_ok());
    result
}

pub async fn warn_if_legacy_signature_backfill_pending(pool: &Pool) -> Result<(), anyhow::Error> {
    let outcome_event_types = crate::execution_feedback::FAILURE_OUTCOME_EVENT_TYPES.to_vec();
    let conn = pool.get().await?;
    let row = conn
        .query_one(
            "SELECT
                count(*) FILTER (WHERE event_type = 'failure_signature')::BIGINT AS legacy_count,
                count(*) FILTER (
                    WHERE event_type = ANY($1)
                      AND metadata->>'success' = 'false'
                      AND (
                          metadata->'payload'->>'signature' IS NULL
                          OR metadata->'payload'->>'signature_category' IS NULL
                          OR metadata->'payload'->>'fingerprint_version' IS NULL
                      )
                )::BIGINT AS pending_count
             FROM agent_events
             WHERE event_type = 'failure_signature'
                OR event_type = ANY($1)",
            &[&outcome_event_types],
        )
        .await?;
    let legacy_count: i64 = row.get("legacy_count");
    let pending_count: i64 = row.get("pending_count");
    if legacy_count > 0 && pending_count > 0 {
        tracing::warn!(
            target: "execution_feedback",
            legacy_signature_events = legacy_count,
            pending_inline_signature_backfill = pending_count,
            command = "orchestrator-maint backfill-signatures",
            "legacy signature events exist; backfill has not completed; retrieval ignores legacy signature rows; operator should run orchestrator-maint backfill-signatures"
        );
    }
    Ok(())
}

async fn remediation_for_failure(
    pool: &Pool,
    failure: &AgentEvent,
    signature: &str,
) -> Result<Option<AgentEvent>, anyhow::Error> {
    // Prefer a remediation in the same correlation chain: that is the strongest
    // signal that a specific failure was resolved by a specific action.
    let chain = get_event_chain_by_event_id(pool, &failure.id).await?;
    if let Some(remediation) = chain.into_iter().find(|event| {
        event.event_type == crate::execution_feedback::EVENT_TYPE_REMEDIATION
            && (event_payload_str(event, "signature") == Some(signature)
                || event_payload_str(event, "failure_event_id") == Some(failure.id.as_str()))
    }) {
        return Ok(Some(remediation));
    }

    // Some clients may submit remediation events later without a parent pointer.
    // Fall back to the latest same-repo remediation with the same signature.
    let conn = pool.get().await?;
    let row = conn
        .query_opt(
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
             FROM agent_events
             WHERE repo = $1
               AND event_type = 'remediation'
               AND metadata->'payload'->>'signature' = $2
             ORDER BY created_at DESC
             LIMIT 1",
            &[&failure.repo, &signature],
        )
        .await?;
    Ok(row.map(|row| rows_to_events(vec![row]).remove(0)))
}

fn event_payload_str<'a>(event: &'a AgentEvent, key: &str) -> Option<&'a str> {
    event
        .metadata
        .get("payload")
        .and_then(|payload| payload.get(key))
        .and_then(serde_json::Value::as_str)
}

pub async fn insert_error_record(
    pool: &Pool,
    repo: &str,
    task: &str,
    error_type: &str,
    description: &str,
    severity: &str,
) -> Result<(), anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        conn.execute(
            "INSERT INTO error_index (repo, task, error_type, description, severity)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (repo, task, error_type, description)
             DO UPDATE SET
                frequency = error_index.frequency + 1,
                last_seen = now()",
            &[&repo, &task, &error_type, &description, &severity],
        )
        .await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("insert_error_record", started.elapsed(), result.is_ok());
    result
}

pub async fn get_active_errors(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<ErrorRecord>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result: Result<Vec<ErrorRecord>, anyhow::Error> = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, repo, task, error_type, description, severity, frequency, last_seen
             FROM error_index
             WHERE repo = $1
             ORDER BY frequency DESC, last_seen DESC
             LIMIT $2",
                &[&repo, &limit],
            )
            .await?;
        rows.into_iter()
            .map(|row| {
                Ok(ErrorRecord {
                    id: row.get("id"),
                    repo: row.get("repo"),
                    task: row.get("task"),
                    error_type: row.get("error_type"),
                    description: row.get("description"),
                    severity: row.get("severity"),
                    frequency: row.get("frequency"),
                    last_seen: row.get("last_seen"),
                })
            })
            .collect()
    }
    .await;
    crate::telemetry::record_db_query("get_active_errors", started.elapsed(), result.is_ok());
    result
}

// ── Request adapters ──────────────────────────────────────────

pub async fn start_session_from_request(
    pool: &Pool,
    req: &crate::state::StartSessionRequest,
) -> Result<String, anyhow::Error> {
    let id = Uuid::new_v4().to_string();
    let actor = req.actor.as_deref().unwrap_or("agent");
    create_session(pool, &id, &req.repo, &req.task, actor).await?;
    Ok(id)
}

/// Returns `(event_id, qdrant_indexed)`. The Postgres write always succeeds or the
/// whole call errors. Qdrant indexing is best-effort: if the embedding model is
/// unavailable the event is still stored and `qdrant_indexed` is false.
pub async fn append_event_from_request(
    pool: &Pool,
    embedder: &crate::embedder::Embedder,
    qdrant_url: &str,
    req: &crate::state::AppendEventRequest,
) -> Result<(String, bool), anyhow::Error> {
    let event = event_from_append_request(req)?;
    let id = event.id.clone();
    insert_event(pool, &event).await?;
    let qdrant_indexed = match crate::qdrant::store_event(embedder, qdrant_url, &event).await {
        Ok(_) => true,
        Err(e) => {
            tracing::warn!(event_id = %id, "qdrant embedding failed, event stored in postgres only: {e}");
            false
        }
    };

    Ok((id, qdrant_indexed))
}

pub fn event_from_append_request(
    req: &crate::state::AppendEventRequest,
) -> Result<AgentEvent, anyhow::Error> {
    let id = Uuid::new_v4().to_string();
    let actor = req.actor.as_deref().unwrap_or("agent");
    crate::trajectory::validate_event_role(req.event_role.as_deref())?;
    if matches!(req.attempt_index, Some(attempt) if attempt < 1) {
        anyhow::bail!("attempt_index must be positive");
    }
    let lineage_fields = [
        req.trajectory_id.is_some(),
        req.attempt_index.is_some(),
        req.event_role.is_some(),
    ];
    if lineage_fields.iter().any(|present| *present)
        && !lineage_fields.iter().all(|present| *present)
    {
        anyhow::bail!("trajectory_id, attempt_index, and event_role must be written together");
    }
    let metadata = req
        .metadata
        .as_ref()
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let metadata = crate::harness_feedback::annotate_event_metadata(
        &req.event_type,
        &req.summary,
        req.evidence.as_deref(),
        metadata,
    );
    let metadata = crate::feature_extraction::annotate_event_metadata(
        &req.event_type,
        &req.summary,
        req.evidence.as_deref(),
        metadata,
    );

    Ok(AgentEvent {
        id: id.clone(),
        session_id: req.session_id.clone(),
        repo: req.repo.clone(),
        actor: actor.to_string(),
        event_type: req.event_type.clone(),
        summary: req.summary.clone(),
        evidence: req.evidence.clone(),
        metadata,
        correlation_id: req.correlation_id,
        parent_event_id: req.parent_event_id,
        trajectory_id: req.trajectory_id,
        attempt_index: req.attempt_index,
        event_role: req.event_role.clone(),
        created_at: chrono::Utc::now(),
        summary_level: 0,
    })
}

// ── Context pack builder ──────────────────────────────────────

pub async fn check_ready(pool: &deadpool_postgres::Pool) -> Result<(), anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let _ = conn.query_one("SELECT 1", &[]).await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("check_ready", started.elapsed(), result.is_ok());
    result
}

/// BUG-4: Context is reference material only — no directive language that overrides the harness.
/// BUG-11: Merges recency and semantic results, deduplicates by summary, enforces a character budget.
pub fn build_context(
    repo: &str,
    task: &str,
    recent: &[crate::state::EventMemory],
    hybrid_hits: &[crate::state::SearchHit],
    errors: &[crate::state::ErrorRecord],
    char_budget: usize,
) -> String {
    let header = format!("Repository: {repo}\nTask: {task}\n\n");

    // Tiered budget allocations: 60% hybrid, 30% recent, 10% errors
    let hybrid_budget = (char_budget * 60) / 100;
    let recent_budget = (char_budget * 30) / 100;
    let error_budget = char_budget.saturating_sub(hybrid_budget + recent_budget);

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // ── Section 1: Relevant Prior Memory ───────────────────────────────
    let mut hybrid_body = String::new();
    let mut used = 0usize;
    for hit in hybrid_hits {
        if !seen.insert(hit.summary.clone()) {
            continue;
        }
        let line = format!("- [{}] {}\n", hit.event_type, hit.summary);
        if used + line.len() > hybrid_budget {
            hybrid_body.push_str("- [truncated: memory budget exceeded]\n");
            break;
        }
        hybrid_body.push_str(&line);
        used += line.len();
    }
    let hybrid_section = if hybrid_body.is_empty() {
        "== Relevant Prior Memory ==\n- No relevant prior memory found.\n".to_string()
    } else {
        format!("== Relevant Prior Memory ==\n{hybrid_body}")
    };

    // ── Sections 2 + 3 share the recent_budget ─────────────────────────
    let mut recent_used = 0usize;

    // Section 2: Recent Decisions
    let mut decision_body = String::new();
    for m in recent {
        if m.event_type != "decision" {
            continue;
        }
        if !seen.insert(m.summary.clone()) {
            continue;
        }
        let line = match &m.evidence {
            Some(ev) => format!("- [decision] {}\n  Evidence: {}\n", m.summary, ev),
            None => format!("- [decision] {}\n", m.summary),
        };
        if recent_used + line.len() > recent_budget {
            break;
        }
        decision_body.push_str(&line);
        recent_used += line.len();
    }

    // Section 3: Open Questions
    let mut open_body = String::new();
    for m in recent {
        if m.event_type != "checkpoint" {
            continue;
        }
        if let Some(arr) = m.metadata.get("open_questions").and_then(|v| v.as_array()) {
            for q in arr {
                if let Some(text) = q.as_str() {
                    if seen.contains(text) {
                        continue;
                    }
                    let line = format!("- {text}\n");
                    if recent_used + line.len() > recent_budget {
                        break;
                    }
                    seen.insert(text.to_string());
                    open_body.push_str(&line);
                    recent_used += line.len();
                }
            }
        }
    }

    // ── Section 4: Failed Approaches ───────────────────────────────────
    let mut error_body = String::new();
    let mut used = 0usize;
    for err in errors {
        let line = format!(
            "- [{}] {} (seen {} times)\n",
            err.error_type, err.description, err.frequency
        );
        if used + line.len() > error_budget {
            break;
        }
        error_body.push_str(&line);
        used += line.len();
    }

    // ── Assemble — omit empty optional sections ─────────────────────────
    let mut out = format!("{header}{hybrid_section}");
    if !decision_body.is_empty() {
        out.push_str(&format!("\n== Recent Decisions ==\n{decision_body}"));
    }
    if !open_body.is_empty() {
        out.push_str(&format!("\n== Open Questions ==\n{open_body}"));
    }
    if !error_body.is_empty() {
        out.push_str(&format!(
            "\n== Failed Approaches (do not retry) ==\n{error_body}"
        ));
    }
    out
}

pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

#[allow(clippy::too_many_arguments)]
pub fn build_layered_context(
    repo: &str,
    task: &str,
    evidence: &ContextEvidence,
    hybrid_hits: &[crate::state::SearchHit],
    errors: &[crate::state::ErrorRecord],
    policy: &crate::state::ContextPolicy,
    char_budget: usize,
    failure_history_char_budget: usize,
) -> (String, crate::state::ContextPackStats) {
    let header = format!(
        "Repository: {repo}\nTask: {task}\nContext policy: {:?}\n\n",
        policy.task_category
    );
    let mut out = header;
    let mut seen = std::collections::HashSet::new();

    append_event_section(
        &mut out,
        "== Durable Project Memory ==",
        &evidence.l3_project,
        policy.budget_for(policy.l3_project_pct, char_budget),
        &mut seen,
    );
    append_event_section(
        &mut out,
        "== Repo Patterns and Decisions ==",
        &evidence.l2_repo,
        policy.budget_for(policy.l2_repo_pct, char_budget),
        &mut seen,
    );
    append_search_section(
        &mut out,
        "== Relevant Prior Memory ==",
        hybrid_hits,
        policy.budget_for(policy.l1_matching_pct, char_budget) / 2,
        &mut seen,
    );
    append_event_section(
        &mut out,
        "== Relevant Session Summaries ==",
        &evidence.l1_matching,
        policy.budget_for(policy.l1_matching_pct, char_budget),
        &mut seen,
    );
    append_event_section(
        &mut out,
        "== Recent Evidence ==",
        &evidence.l0_recent,
        policy.budget_for(policy.l0_recent_pct, char_budget),
        &mut seen,
    );
    append_failure_section(
        &mut out,
        &evidence.failures,
        errors,
        policy.budget_for(policy.failure_pct, char_budget),
        &mut seen,
    );
    append_operational_constraints_section(&mut out, &evidence.operational_constraints);
    let reused_signatures = append_failure_history_section(
        &mut out,
        &evidence.failure_history,
        failure_history_char_budget,
        &mut seen,
    );
    append_open_questions(&mut out, &evidence.l0_recent, &mut seen);

    let mut stats = evidence.stats();
    stats.failure_history_remediation_signatures = reused_signatures;
    stats.operational_constraints_injected = evidence.operational_constraints.len();
    stats.context_chars = out.len();
    stats.context_tokens_estimate = estimate_tokens(&out);
    stats.retrieval_deduped_hits = hybrid_hits.len();

    (out, stats)
}

fn append_operational_constraints_section(
    out: &mut String,
    constraints: &[crate::feature_extraction::OperationalConstraint],
) {
    if constraints.is_empty() {
        return;
    }

    out.push_str("Operational Constraints:\n");
    for constraint in constraints {
        out.push_str("- ");
        out.push_str(&constraint.text);
        out.push('\n');
    }
    out.push('\n');
}

fn append_failure_history_section(
    out: &mut String,
    items: &[FailureHistoryItem],
    budget: usize,
    seen: &mut std::collections::HashSet<String>,
) -> Vec<String> {
    if items.is_empty() || budget == 0 {
        return vec![];
    }

    let mut body = String::new();
    let mut used = 0usize;
    let mut reused_signatures = Vec::new();
    for item in items {
        let key = format!("failure-history:{}", item.failure.id);
        if !seen.insert(key) {
            continue;
        }
        // This section is deterministic outcome history, separate from the
        // conversational "Failed Attempts and Remediations" memory above it.
        let mut line = format!(
            "- [failure:{}] {}\n  Category: {}\n",
            item.signature, item.failure.summary, item.category
        );
        if let Some(excerpt) = failure_history_excerpt(&item.failure) {
            line.push_str(&format!(
                "  Excerpt: {}\n",
                excerpt.chars().take(300).collect::<String>()
            ));
        }
        if let Some(remediation) = &item.remediation {
            line.push_str(&format!("  Remediation: {}\n", remediation.summary));
            // The metrics layer increments remediation_reuse_total for surfaced
            // remediations, which gives operators a signal that prior fixes are
            // actually helping context assembly.
            reused_signatures.push(item.signature.clone());
        }
        if used + line.len() > budget {
            body.push_str("- [truncated: failure history budget exceeded]\n");
            break;
        }
        used += line.len();
        body.push_str(&line);
    }

    if !body.is_empty() {
        out.push_str("== Failure History ==\n");
        out.push_str(&body);
        out.push('\n');
    }
    reused_signatures
}

fn failure_history_excerpt(event: &AgentEvent) -> Option<String> {
    let payload = event.metadata.get("payload")?;
    for key in [
        "raw_excerpt",
        "failure_reason",
        "stderr_summary",
        "stdout_summary",
    ] {
        if let Some(text) = payload
            .get(key)
            .and_then(serde_json::Value::as_str)
            .filter(|text| !text.trim().is_empty())
        {
            return Some(text.to_string());
        }
    }
    for key in ["failure_summaries", "findings"] {
        if let Some(values) = payload.get(key).and_then(serde_json::Value::as_array) {
            let joined = values
                .iter()
                .filter_map(serde_json::Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            if !joined.is_empty() {
                return Some(joined);
            }
        }
    }
    None
}

fn append_event_section(
    out: &mut String,
    title: &str,
    events: &[AgentEvent],
    budget: usize,
    seen: &mut std::collections::HashSet<String>,
) {
    if events.is_empty() || budget == 0 {
        return;
    }

    let mut body = String::new();
    let mut used = 0usize;
    for event in events {
        if !seen.insert(event.summary.clone()) {
            continue;
        }
        let line = format_event_line(event);
        if used + line.len() > budget {
            body.push_str("- [truncated: section budget exceeded]\n");
            break;
        }
        used += line.len();
        body.push_str(&line);
    }

    if !body.is_empty() {
        out.push_str(title);
        out.push('\n');
        out.push_str(&body);
        out.push('\n');
    }
}

fn append_search_section(
    out: &mut String,
    title: &str,
    hits: &[crate::state::SearchHit],
    budget: usize,
    seen: &mut std::collections::HashSet<String>,
) {
    if hits.is_empty() || budget == 0 {
        return;
    }

    let mut body = String::new();
    let mut used = 0usize;
    for hit in hits {
        if !seen.insert(hit.summary.clone()) {
            continue;
        }
        let line = format!("- [retrieved:{}] {}\n", hit.event_type, hit.summary);
        if used + line.len() > budget {
            body.push_str("- [truncated: retrieval budget exceeded]\n");
            break;
        }
        used += line.len();
        body.push_str(&line);
    }

    if !body.is_empty() {
        out.push_str(title);
        out.push('\n');
        out.push_str(&body);
        out.push('\n');
    }
}

fn append_failure_section(
    out: &mut String,
    failures: &[AgentEvent],
    errors: &[crate::state::ErrorRecord],
    budget: usize,
    seen: &mut std::collections::HashSet<String>,
) {
    if failures.is_empty() && errors.is_empty() {
        return;
    }

    let mut body = String::new();
    let mut used = 0usize;
    for event in failures {
        if !seen.insert(event.summary.clone()) {
            continue;
        }
        let line = format_event_line(event);
        if used + line.len() > budget {
            body.push_str("- [truncated: failure budget exceeded]\n");
            break;
        }
        used += line.len();
        body.push_str(&line);
    }

    for err in errors {
        let text = format!("{}:{}", err.error_type, err.description);
        if !seen.insert(text) {
            continue;
        }
        let line = format!(
            "- [error_index:{}] {} (seen {} times)\n",
            err.error_type, err.description, err.frequency
        );
        if used + line.len() > budget {
            break;
        }
        used += line.len();
        body.push_str(&line);
    }

    if !body.is_empty() {
        out.push_str("== Failed Attempts and Remediations ==\n");
        out.push_str(&body);
        out.push('\n');
    }
}

fn append_open_questions(
    out: &mut String,
    recent: &[AgentEvent],
    seen: &mut std::collections::HashSet<String>,
) {
    const BUDGET: usize = 500;
    let mut body = String::new();
    let mut used = 0usize;
    'outer: for event in recent {
        if event.event_type != "checkpoint" {
            continue;
        }
        if let Some(arr) = event
            .metadata
            .get("open_questions")
            .and_then(|v| v.as_array())
        {
            for q in arr {
                if let Some(text) = q.as_str() {
                    let line = format!("- {text}\n");
                    if used + line.len() > BUDGET {
                        break 'outer;
                    }
                    if seen.insert(text.to_string()) {
                        body.push_str(&line);
                        used += line.len();
                    }
                }
            }
        }
    }

    if !body.is_empty() {
        out.push_str("== Open Questions ==\n");
        out.push_str(&body);
    }
}

fn format_event_line(event: &AgentEvent) -> String {
    let level = format!("L{}", event.summary_level);
    let mut line = format!("- [{level}:{}] {}", event.event_type, event.summary);
    if let Some(evidence) = &event.evidence {
        if !evidence.is_empty() {
            line.push_str(&format!("\n  Evidence: {evidence}"));
        }
    }
    if let Some(source_ids) = event
        .metadata
        .get("source_event_ids")
        .or_else(|| event.metadata.get("summarized_event_ids"))
        .and_then(|v| v.as_array())
    {
        if !source_ids.is_empty() {
            line.push_str(&format!("\n  Source IDs: {}", source_ids.len()));
        }
    }
    if let Some(outcome) = event.metadata.get("outcome").and_then(|v| v.as_str()) {
        line.push_str(&format!("\n  Outcome: {outcome}"));
    }
    line.push('\n');
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ErrorRecord, SearchHit};
    use chrono::Utc;

    fn mem(event_type: &str, summary: &str, evidence: Option<&str>) -> crate::state::EventMemory {
        crate::state::EventMemory {
            event_type: event_type.to_string(),
            summary: summary.to_string(),
            evidence: evidence.map(str::to_string),
            metadata: serde_json::json!({}),
            created_at: Utc::now(),
            summary_level: 0,
        }
    }

    fn mem_with_meta(
        event_type: &str,
        summary: &str,
        metadata: serde_json::Value,
    ) -> crate::state::EventMemory {
        crate::state::EventMemory {
            event_type: event_type.to_string(),
            summary: summary.to_string(),
            evidence: None,
            metadata,
            created_at: Utc::now(),
            summary_level: 0,
        }
    }

    fn hit(event_type: &str, summary: &str) -> SearchHit {
        SearchHit {
            event_id: uuid::Uuid::new_v4().to_string(),
            event_type: event_type.to_string(),
            summary: summary.to_string(),
            created_at: Some(Utc::now()),
        }
    }

    fn err_rec(error_type: &str, description: &str, frequency: i64) -> ErrorRecord {
        ErrorRecord {
            id: "eid".to_string(),
            repo: "r".to_string(),
            task: "t".to_string(),
            error_type: error_type.to_string(),
            description: description.to_string(),
            severity: "medium".to_string(),
            frequency,
            last_seen: Utc::now(),
        }
    }

    fn event(summary: &str) -> AgentEvent {
        AgentEvent {
            id: "tid".to_string(),
            session_id: "tsession".to_string(),
            repo: "trepo".to_string(),
            actor: "tactor".to_string(),
            event_type: "decision".to_string(),
            summary: summary.to_string(),
            evidence: None,
            metadata: serde_json::json!({}),
            correlation_id: None,
            parent_event_id: None,
            trajectory_id: None,
            attempt_index: None,
            event_role: None,
            created_at: Utc::now(),
            summary_level: 0,
        }
    }

    fn event_at_level(event_type: &str, summary: &str, level: i32) -> AgentEvent {
        AgentEvent {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: "session".to_string(),
            repo: "repo".to_string(),
            actor: "actor".to_string(),
            event_type: event_type.to_string(),
            summary: summary.to_string(),
            evidence: None,
            metadata: serde_json::json!({}),
            correlation_id: None,
            parent_event_id: None,
            trajectory_id: None,
            attempt_index: None,
            event_role: None,
            created_at: Utc::now(),
            summary_level: level,
        }
    }

    fn failed_event(summary: &str, evidence: &str, outcome: &str) -> AgentEvent {
        AgentEvent {
            evidence: Some(evidence.to_string()),
            metadata: serde_json::json!({
                "source_event_ids": ["a", "b"],
                "outcome": outcome,
            }),
            ..event_at_level("failed_attempt", summary, 0)
        }
    }

    #[test]
    fn build_context_header_contains_repo_and_task() {
        let out = build_context("my-repo", "my-task", &[], &[], &[], 8000);
        assert!(out.starts_with("Repository: my-repo\nTask: my-task\n"));
    }

    #[test]
    fn build_context_empty_shows_no_prior_memory() {
        let out = build_context("r", "t", &[], &[], &[], 8000);
        assert!(out.contains("No relevant prior memory found"));
    }

    #[test]
    fn build_context_hybrid_hit_appears_in_relevant_section() {
        let hits = vec![hit("edit", "changed auth flow")];
        let out = build_context("r", "t", &[], &hits, &[], 8000);
        assert!(out.contains("== Relevant Prior Memory =="));
        assert!(out.contains("changed auth flow"));
    }

    #[test]
    fn build_context_decision_event_appears_in_decisions_section() {
        let recent = vec![mem(
            "decision",
            "chose retry logic",
            Some("503 intermittent"),
        )];
        let out = build_context("r", "t", &recent, &[], &[], 8000);
        assert!(out.contains("== Recent Decisions =="));
        assert!(out.contains("chose retry logic"));
        assert!(out.contains("Evidence: 503 intermittent"));
    }

    #[test]
    fn build_context_non_decision_recent_not_in_decisions_section() {
        let recent = vec![mem("edit", "changed a file", None)];
        let out = build_context("r", "t", &recent, &[], &[], 8000);
        assert!(!out.contains("== Recent Decisions =="));
    }

    #[test]
    fn build_context_error_appears_in_failed_approaches_section() {
        let errs = vec![err_rec("auth_500", "OAuth callback 500", 3)];
        let out = build_context("r", "t", &[], &[], &errs, 8000);
        assert!(out.contains("== Failed Approaches (do not retry) =="));
        assert!(out.contains("OAuth callback 500"));
        assert!(out.contains("seen 3 times"));
    }

    #[test]
    fn build_context_open_questions_from_checkpoint_metadata() {
        let recent = vec![mem_with_meta(
            "checkpoint",
            "end of sprint",
            serde_json::json!({
                "open_questions": ["Is retry threshold right?", "Do we need caching?"]
            }),
        )];
        let out = build_context("r", "t", &recent, &[], &[], 8000);
        assert!(out.contains("== Open Questions =="));
        assert!(out.contains("Is retry threshold right?"));
    }

    #[test]
    fn build_context_empty_sections_omitted() {
        let hits = vec![hit("edit", "some memory")];
        let out = build_context("r", "t", &[], &hits, &[], 8000);
        assert!(!out.contains("== Recent Decisions =="));
        assert!(!out.contains("== Failed Approaches"));
        assert!(!out.contains("== Open Questions =="));
    }

    #[test]
    fn build_context_deduplicates_same_summary_across_sections() {
        let shared = "shared summary text";
        let hits = vec![hit("edit", shared)];
        let recent = vec![mem("decision", shared, None)];
        let out = build_context("r", "t", &recent, &hits, &[], 8000);
        assert_eq!(out.matches(shared).count(), 1);
    }

    #[test]
    fn build_context_truncates_hybrid_section_at_budget() {
        let hits: Vec<SearchHit> = (0..20)
            .map(|i| {
                hit(
                    "edit",
                    &format!(
                        "event {i} with enough text to consume the hybrid budget allocation here"
                    ),
                )
            })
            .collect();
        let out = build_context("r", "t", &[], &hits, &[], 500);
        assert!(out.contains("truncated"));
    }

    #[test]
    fn context_policy_small_task_includes_l0_and_l1() {
        let mut evidence = ContextEvidence::default();
        evidence.l0_recent = vec![event_at_level("edit", "fresh implementation detail", 0)];
        evidence.l1_matching = vec![event_at_level("summary", "matching prior task summary", 1)];
        evidence.l2_repo = vec![event_at_level("summary", "repo convention", 2)];

        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
        let (out, stats) =
            build_layered_context("r", "fix bug", &evidence, &[], &[], &policy, 8000, 4000);

        assert!(out.contains("== Recent Evidence =="));
        assert!(out.contains("fresh implementation detail"));
        assert!(out.contains("== Relevant Session Summaries =="));
        assert!(out.contains("matching prior task summary"));
        assert_eq!(stats.l0_items_injected, 1);
        assert_eq!(stats.l1_items_injected, 1);
    }

    #[test]
    fn context_policy_broad_task_includes_l1_l2_and_l0_tail() {
        let mut evidence = ContextEvidence::default();
        evidence.l0_recent = vec![event_at_level("edit", "latest local change", 0)];
        evidence.l1_matching = vec![event_at_level("summary", "session summary", 1)];
        evidence.l2_repo = vec![event_at_level("summary", "repo-level pattern", 2)];

        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Broad);
        let (out, _stats) = build_layered_context(
            "r",
            "large refactor",
            &evidence,
            &[],
            &[],
            &policy,
            8000,
            4000,
        );

        assert!(out.contains("latest local change"));
        assert!(out.contains("session summary"));
        assert!(out.contains("repo-level pattern"));
    }

    #[test]
    fn context_policy_architecture_task_prioritizes_l2_l3() {
        let mut evidence = ContextEvidence::default();
        evidence.l2_repo = vec![event_at_level("summary", "repository decision record", 2)];
        evidence.l3_project = vec![event_at_level("summary", "durable architecture truth", 3)];

        let policy =
            crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Architecture);
        let (out, stats) = build_layered_context(
            "r",
            "architecture review",
            &evidence,
            &[],
            &[],
            &policy,
            8000,
            4000,
        );

        let l3_pos = out.find("durable architecture truth").unwrap();
        let l2_pos = out.find("repository decision record").unwrap();
        assert!(l3_pos < l2_pos);
        assert_eq!(stats.l2_items_injected, 1);
        assert_eq!(stats.l3_items_injected, 1);
    }

    #[test]
    fn higher_levels_do_not_eliminate_l0_tail() {
        let mut evidence = ContextEvidence::default();
        evidence.l0_recent = vec![event_at_level("edit", "raw event still included", 0)];
        evidence.l3_project = vec![event_at_level("summary", "durable compressed memory", 3)];

        let policy =
            crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Architecture);
        let (out, _stats) = build_layered_context(
            "r",
            "architecture",
            &evidence,
            &[],
            &[],
            &policy,
            8000,
            4000,
        );

        assert!(out.contains("raw event still included"));
        assert!(out.contains("durable compressed memory"));
    }

    #[test]
    fn failed_attempt_preserves_evidence_and_outcome() {
        let mut evidence = ContextEvidence::default();
        evidence.failures = vec![failed_event(
            "retrying the old cache key caused stale context",
            "observed repeated stale pack after append_event",
            "new key includes event count",
        )];

        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
        let (out, stats) =
            build_layered_context("r", "fix cache", &evidence, &[], &[], &policy, 8000, 4000);

        assert!(out.contains("== Failed Attempts and Remediations =="));
        assert!(out.contains("observed repeated stale pack after append_event"));
        assert!(out.contains("Outcome: new key includes event count"));
        assert!(out.contains("Source IDs: 2"));
        assert_eq!(stats.failed_attempts_injected, 1);
    }

    #[test]
    fn failure_history_section_sits_between_failures_and_open_questions() {
        let mut evidence = ContextEvidence::default();
        evidence.failures = vec![failed_event(
            "old patch broke cache invalidation",
            "cargo test failed",
            "invalidate by repo prefix",
        )];
        evidence.failure_history = vec![FailureHistoryItem {
            signature: "rust:type-mismatch".to_string(),
            category: "type_error".to_string(),
            failure: AgentEvent {
                event_type: crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT.to_string(),
                summary: "failed compile `rust` target `cargo` errors=1 warnings=0".to_string(),
                metadata: serde_json::json!({
                    "payload": {
                        "signature": "rust:type-mismatch",
                        "signature_category": "type_error",
                        "fingerprint_version": 1,
                        "failure_summaries": ["error[E0308]: mismatched types"]
                    }
                }),
                ..event_at_level(
                    crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT,
                    "failed compile `rust` target `cargo` errors=1 warnings=0",
                    0,
                )
            },
            remediation: Some(AgentEvent {
                event_type: crate::execution_feedback::EVENT_TYPE_REMEDIATION.to_string(),
                summary: "changed generic bound to match caller".to_string(),
                metadata: serde_json::json!({
                    "payload": {"signature": "rust:type-mismatch"}
                }),
                ..event_at_level(
                    crate::execution_feedback::EVENT_TYPE_REMEDIATION,
                    "changed generic bound to match caller",
                    0,
                )
            }),
        }];
        evidence.l0_recent = vec![AgentEvent {
            event_type: "checkpoint".to_string(),
            metadata: serde_json::json!({"open_questions": ["Should cache TTL change?"]}),
            ..event_at_level("checkpoint", "checkpoint", 0)
        }];

        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
        let (out, stats) = build_layered_context(
            "r",
            "error[E0308]",
            &evidence,
            &[],
            &[],
            &policy,
            8000,
            4000,
        );

        let failed = out.find("== Failed Attempts and Remediations ==").unwrap();
        let history = out.find("== Failure History ==").unwrap();
        let open = out.find("== Open Questions ==").unwrap();
        assert!(failed < history);
        assert!(history < open);
        assert!(out.contains("changed generic bound to match caller"));
        assert_eq!(stats.failure_history_items_injected, 1);
        assert_eq!(
            stats.failure_history_remediation_signatures,
            vec!["rust:type-mismatch".to_string()]
        );
    }

    #[test]
    fn operational_constraints_section_sits_above_failure_history() {
        let mut evidence = ContextEvidence::default();
        evidence.operational_constraints = vec![crate::feature_extraction::OperationalConstraint {
            constraint_type: "use_known_endpoint".to_string(),
            text: "Do not use `localhost` for host-side orchestrator testing. The correct endpoint for this environment is `http://host.docker.internal:8088`.".to_string(),
        }];
        evidence.failure_history = vec![FailureHistoryItem {
            signature: "process:non-zero-exit".to_string(),
            category: "unknown".to_string(),
            failure: event_at_level(
                crate::execution_feedback::EVENT_TYPE_TOOL_RESULT,
                "failed tool `Bash` exit_code=1",
                0,
            ),
            remediation: None,
        }];

        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
        let (out, stats) =
            build_layered_context("r", "debug", &evidence, &[], &[], &policy, 8000, 4000);

        let constraints = out.find("Operational Constraints:").unwrap();
        let history = out.find("== Failure History ==").unwrap();
        assert!(constraints < history);
        assert!(out.contains("- Do not use `localhost`"));
        assert_eq!(stats.operational_constraints_injected, 1);
    }

    #[test]
    fn failure_history_section_omitted_when_empty() {
        let evidence = ContextEvidence::default();
        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
        let (out, stats) =
            build_layered_context("r", "no failures", &evidence, &[], &[], &policy, 8000, 4000);

        assert!(!out.contains("== Failure History =="));
        assert_eq!(stats.failure_history_items_injected, 0);
    }

    #[test]
    fn failure_history_section_respects_budget() {
        let mut evidence = ContextEvidence::default();
        evidence.failure_history = vec![FailureHistoryItem {
            signature: "unknown".to_string(),
            category: "unknown".to_string(),
            failure: AgentEvent {
                event_type: crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT.to_string(),
                summary: "x".repeat(200),
                metadata: serde_json::json!({
                    "payload": {
                        "signature": "unknown",
                        "signature_category": "unknown",
                        "fingerprint_version": 1,
                        "failure_reason": "y".repeat(200)
                    }
                }),
                ..event_at_level(
                    crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT,
                    "long failure",
                    0,
                )
            },
            remediation: None,
        }];
        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
        let (out, _stats) =
            build_layered_context("r", "unknown", &evidence, &[], &[], &policy, 8000, 50);
        assert!(out.contains("truncated: failure history budget exceeded"));
    }

    #[test]
    fn event_chain_orders_from_root_to_leaf() {
        let correlation_id = uuid::Uuid::new_v4();
        let root_id = uuid::Uuid::new_v4();
        let response_id = uuid::Uuid::new_v4();
        let tool_id = uuid::Uuid::new_v4();
        let validation_id = uuid::Uuid::new_v4();

        let mut root = event_at_level("user_message", "request", 0);
        root.id = root_id.to_string();
        root.correlation_id = Some(correlation_id);
        root.parent_event_id = None;

        let mut response = event_at_level("assistant_message", "patch", 0);
        response.id = response_id.to_string();
        response.correlation_id = Some(correlation_id);
        response.parent_event_id = Some(root_id);

        let mut tool = event_at_level(
            crate::execution_feedback::EVENT_TYPE_TOOL_RESULT,
            "apply",
            0,
        );
        tool.id = tool_id.to_string();
        tool.correlation_id = Some(correlation_id);
        tool.parent_event_id = Some(response_id);

        let mut validation = event_at_level(
            crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT,
            "schema ok",
            0,
        );
        validation.id = validation_id.to_string();
        validation.correlation_id = Some(correlation_id);
        validation.parent_event_id = Some(tool_id);

        let ordered = order_event_chain(
            vec![
                validation.clone(),
                tool.clone(),
                root.clone(),
                response.clone(),
            ],
            &validation.id,
        );
        let ids: Vec<String> = ordered.into_iter().map(|event| event.id).collect();
        assert_eq!(
            ids,
            vec![
                root_id.to_string(),
                response_id.to_string(),
                tool_id.to_string(),
                validation_id.to_string()
            ]
        );
    }

    #[test]
    fn failure_history_context_build_p99_under_five_ms() {
        let mut durations = Vec::new();
        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
        let mut evidence = ContextEvidence::default();
        evidence.failure_history = (0..50)
            .map(|idx| FailureHistoryItem {
                signature: format!("typescript:TS{idx:04}"),
                category: "typescript".to_string(),
                failure: event_at_level(
                    crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT,
                    &format!("failure {idx}"),
                    0,
                ),
                remediation: Some(event_at_level(
                    crate::execution_feedback::EVENT_TYPE_REMEDIATION,
                    &format!("remediation {idx}"),
                    0,
                )),
            })
            .collect();

        for _ in 0..100 {
            let started = std::time::Instant::now();
            let _ = build_layered_context(
                "r",
                "typescript:TS2322",
                &evidence,
                &[],
                &[],
                &policy,
                8000,
                4000,
            );
            durations.push(started.elapsed());
        }
        durations.sort();
        let p99 = durations[durations.len() - 1];
        let threshold = if cfg!(debug_assertions) {
            std::time::Duration::from_millis(20)
        } else {
            std::time::Duration::from_millis(5)
        };
        assert!(p99 < threshold, "p99 was {p99:?}");
    }

    #[test]
    fn promoted_memory_requires_source_ids() {
        assert!(!crate::summarizer::has_source_ids(&serde_json::json!({})));
        assert!(!crate::summarizer::has_source_ids(
            &serde_json::json!({"summarized_event_ids": []})
        ));
        assert!(crate::summarizer::has_source_ids(
            &serde_json::json!({"summarized_event_ids": ["e1"]})
        ));
        assert!(crate::summarizer::has_source_ids(
            &serde_json::json!({"source_event_ids": ["e1"]})
        ));
    }

    #[test]
    fn vector_text_contains_all_fields() {
        let mut e = event("summary text");
        e.evidence = Some("evidence text".to_string());
        let t = e.vector_text();
        assert!(t.contains("trepo"));
        assert!(t.contains("decision"));
        assert!(t.contains("summary text"));
        assert!(t.contains("evidence text"));
    }

    #[test]
    fn vector_text_with_no_evidence_does_not_panic() {
        let e = event("summary text");
        let t = e.vector_text();
        assert!(t.contains("summary text"));
    }

    #[test]
    fn payload_contains_required_fields() {
        let e = event("payload summary");
        let p = e.payload();
        assert_eq!(p["event_id"], "tid");
        assert_eq!(p["repo"], "trepo");
        assert_eq!(p["event_type"], "decision");
        assert_eq!(p["summary"], "payload summary");
        assert!(p["created_at"].is_string());
    }

    #[test]
    fn to_memory_copies_fields_correctly() {
        let mut e = event("mem summary");
        e.evidence = Some("proof".to_string());
        let m = e.to_memory();
        assert_eq!(m.event_type, "decision");
        assert_eq!(m.summary, "mem summary");
        assert_eq!(m.evidence, Some("proof".to_string()));
    }

    #[test]
    fn error_record_frequency_field_accessible() {
        let rec = crate::state::ErrorRecord {
            id: "id".to_string(),
            repo: "r".to_string(),
            task: "t".to_string(),
            error_type: "e".to_string(),
            description: "d".to_string(),
            severity: "medium".to_string(),
            frequency: 3,
            last_seen: Utc::now(),
        };
        assert_eq!(rec.frequency, 3);
    }

    #[test]
    fn summary_level_filters_correctly() {
        assert_eq!(preferred_summary_levels(0), vec![0]);
        assert_eq!(preferred_summary_levels(19), vec![0]);
        assert_eq!(preferred_summary_levels(20), vec![1]);
        assert_eq!(preferred_summary_levels(199), vec![1]);
        assert_eq!(preferred_summary_levels(200), vec![2]);
        assert_eq!(preferred_summary_levels(1999), vec![2]);
        assert_eq!(preferred_summary_levels(2000), vec![3, 2]);
    }
}
