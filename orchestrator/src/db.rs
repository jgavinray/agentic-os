use crate::state::ErrorRecord;
use deadpool_postgres::{Config, Pool, PoolConfig};
use tokio_postgres::NoTls;
use uuid::Uuid;

const SINGLE_WRITER_LOCK_KEY: i64 = 0x4167_656e_7469_634f;

// ── Reusable event type ──────────────────────────────────────

pub struct AgentEvent {
    pub id: String,
    pub session_id: String,
    pub repo: String,
    pub actor: String,
    pub event_type: String,
    pub summary: String,
    pub evidence: Option<String>,
    pub metadata: serde_json::Value,
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
}

impl ContextEvidence {
    pub fn memories(&self) -> Vec<crate::state::EventMemory> {
        self.l0_recent
            .iter()
            .chain(self.l1_matching.iter())
            .chain(self.l2_repo.iter())
            .chain(self.l3_project.iter())
            .chain(self.failures.iter())
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
            ..Default::default()
        }
    }
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
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        conn.execute(
            "INSERT INTO agent_events
             (id, session_id, repo, actor, event_type, summary, evidence, metadata, summary_level)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            &[
                &event.id,
                &event.session_id,
                &event.repo,
                &event.actor,
                &event.event_type,
                &event.summary,
                &event.evidence,
                &event.metadata,
                &event.summary_level,
            ],
        )
        .await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("insert_event", started.elapsed(), result.is_ok());
    result
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
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, created_at, summary_level
             FROM agent_events
             WHERE repo = $1
               AND summarized = false
               AND summary_level = 0
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
                "SELECT count(*)::BIGINT AS count FROM agent_events WHERE repo = $1",
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
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, created_at, summary_level
             FROM agent_events
             WHERE repo = $1
               AND summary_level = $2
               AND event_type NOT IN ('failed_attempt', 'remediation')
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
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, created_at, summary_level
             FROM agent_events
             WHERE repo = $1
               AND event_type IN ('failed_attempt', 'remediation')
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
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, created_at, summary_level
             FROM agent_events
             WHERE repo = $1
               AND summarized = false
               AND summary_level = ANY($2)
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
    let id = Uuid::new_v4().to_string();
    let actor = req.actor.as_deref().unwrap_or("agent");
    let metadata = req
        .metadata
        .as_ref()
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    let event = AgentEvent {
        id: id.clone(),
        session_id: req.session_id.clone(),
        repo: req.repo.clone(),
        actor: actor.to_string(),
        event_type: req.event_type.clone(),
        summary: req.summary.clone(),
        evidence: req.evidence.clone(),
        metadata,
        created_at: chrono::Utc::now(),
        summary_level: 0,
    };

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

pub fn build_layered_context(
    repo: &str,
    task: &str,
    evidence: &ContextEvidence,
    hybrid_hits: &[crate::state::SearchHit],
    errors: &[crate::state::ErrorRecord],
    policy: &crate::state::ContextPolicy,
    char_budget: usize,
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
    append_open_questions(&mut out, &evidence.l0_recent, &mut seen);

    let mut stats = evidence.stats();
    stats.context_chars = out.len();
    stats.context_tokens_estimate = estimate_tokens(&out);
    stats.retrieval_deduped_hits = hybrid_hits.len();

    (out, stats)
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
            build_layered_context("r", "fix bug", &evidence, &[], &[], &policy, 8000);

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
        let (out, _stats) =
            build_layered_context("r", "large refactor", &evidence, &[], &[], &policy, 8000);

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
        let (out, _stats) =
            build_layered_context("r", "architecture", &evidence, &[], &[], &policy, 8000);

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
            build_layered_context("r", "fix cache", &evidence, &[], &[], &policy, 8000);

        assert!(out.contains("== Failed Attempts and Remediations =="));
        assert!(out.contains("observed repeated stale pack after append_event"));
        assert!(out.contains("Outcome: new key includes event count"));
        assert!(out.contains("Source IDs: 2"));
        assert_eq!(stats.failed_attempts_injected, 1);
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
