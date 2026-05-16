use crate::state::ErrorRecord;
use deadpool_postgres::{Config, Pool, PoolConfig};
use tokio_postgres::NoTls;
use uuid::Uuid;

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
}

impl AgentEvent {
    pub fn to_memory(&self) -> crate::state::EventMemory {
        crate::state::EventMemory {
            event_type: self.event_type.clone(),
            summary: self.summary.clone(),
            evidence: self.evidence.clone(),
            metadata: self.metadata.clone(),
            created_at: self.created_at,
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
            "created_at": self.created_at
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

pub async fn init_schema(pool: &Pool) -> Result<(), anyhow::Error> {
    let conn = pool.get().await?;
    conn.batch_execute(
        r#"
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS agent_sessions (
    id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    task TEXT NOT NULL,
    actor TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    repo TEXT NOT NULL,
    actor TEXT NOT NULL,
    event_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    evidence TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT fk_session
        FOREIGN KEY (session_id)
        REFERENCES agent_sessions(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS error_index (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    repo TEXT NOT NULL,
    task TEXT NOT NULL,
    error_type TEXT NOT NULL,
    description TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'medium',
    frequency BIGINT NOT NULL DEFAULT 1,
    last_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_error_index UNIQUE (repo, task, error_type, description)
);

CREATE INDEX IF NOT EXISTS idx_agent_events_repo_created
    ON agent_events(repo, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_events_session
    ON agent_events(session_id);

CREATE INDEX IF NOT EXISTS idx_error_index_repo
    ON error_index(repo);
CREATE INDEX IF NOT EXISTS idx_error_index_type
    ON error_index(error_type);
CREATE INDEX IF NOT EXISTS idx_error_index_freq
    ON error_index(frequency DESC);

CREATE INDEX IF NOT EXISTS idx_agent_events_fts
    ON agent_events USING gin(
        to_tsvector('english',
            coalesce(summary, '') || ' ' || coalesce(evidence, ''))
    );

ALTER TABLE agent_events ADD COLUMN IF NOT EXISTS summarized BOOLEAN NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS idx_agent_events_type
    ON agent_events(event_type, repo);
        "#,
    )
    .await?;

    Ok(())
}

// ── DB query functions ────────────────────────────────────────

pub async fn create_session(
    pool: &Pool,
    session_id: &str,
    repo: &str,
    task: &str,
    actor: &str,
) -> Result<(), anyhow::Error> {
    let conn = pool.get().await?;
    conn.execute(
        "INSERT INTO agent_sessions (id, repo, task, actor) VALUES ($1, $2, $3, $4)",
        &[&session_id, &repo, &task, &actor],
    )
    .await?;
    Ok(())
}

/// BUG-2: Find an existing session for (repo, task) from the last 4 hours, or create one.
/// Prevents callers from needing a separate /sessions/start call for every completion.
pub async fn find_or_create_session(
    pool: &Pool,
    repo: &str,
    task: &str,
    actor: &str,
) -> Result<String, anyhow::Error> {
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

pub async fn insert_event(pool: &Pool, event: &AgentEvent) -> Result<(), anyhow::Error> {
    let conn = pool.get().await?;
    conn.execute(
        "INSERT INTO agent_events
         (id, session_id, repo, actor, event_type, summary, evidence, metadata)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        &[
            &event.id,
            &event.session_id,
            &event.repo,
            &event.actor,
            &event.event_type,
            &event.summary,
            &event.evidence,
            &event.metadata,
        ],
    )
    .await?;
    Ok(())
}

pub async fn get_events_for_repo(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let conn = pool.get().await?;
    let rows = conn
        .query(
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, created_at
             FROM agent_events
             WHERE repo = $1
               AND summarized = false
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
        })
        .collect();

    Ok(events)
}

/// Full-text search on agent_events.summary and evidence for a given repo.
/// Returns results ordered by ts_rank DESC.
pub async fn search_events_fts(
    pool: &Pool,
    repo: &str,
    query: &str,
    limit: i64,
) -> Result<Vec<crate::state::SearchHit>, anyhow::Error> {
    let conn = pool.get().await?;
    let rows = conn
        .query(
            "WITH docs AS (
                 SELECT id, event_type, summary,
                        to_tsvector('english', coalesce(summary, '') || ' ' || coalesce(evidence, '')) AS tsv
                 FROM agent_events
                 WHERE repo = $1
                   AND summarized = false
             )
             SELECT id, event_type, summary
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
        })
        .collect())
}

pub async fn insert_error_record(
    pool: &Pool,
    repo: &str,
    task: &str,
    error_type: &str,
    description: &str,
    severity: &str,
) -> Result<(), anyhow::Error> {
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

pub async fn get_active_errors(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<ErrorRecord>, anyhow::Error> {
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
    http: &reqwest::Client,
    embedding_url: &str,
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
    };

    insert_event(pool, &event).await?;
    let qdrant_indexed = match crate::qdrant::store_event(http, embedding_url, qdrant_url, &event)
        .await
    {
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
    let conn = pool.get().await?;
    let _ = conn.query_one("SELECT 1", &[]).await?;
    Ok(())
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
        }
    }

    fn hit(event_type: &str, summary: &str) -> SearchHit {
        SearchHit {
            event_id: uuid::Uuid::new_v4().to_string(),
            event_type: event_type.to_string(),
            summary: summary.to_string(),
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
}
