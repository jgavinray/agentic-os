use deadpool_postgres::{Config, Pool};
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
            self.repo, self.event_type, self.summary,
            self.evidence.as_deref().unwrap_or(""),
            self.metadata
        )
    }
}

// ── Connection pool ────────────────────────────────────────────

pub fn create_pool(database_url: &str) -> Result<Pool, anyhow::Error> {
    let mut cfg = Config::new();
    cfg.url = Some(database_url.to_string());
    if let Some(pool) = &mut cfg.pool {
        pool.max_size = 16;
    }

    let pool = cfg.create_pool(None, NoTls)?;
    Ok(pool)
}

pub async fn init_schema(pool: &Pool) -> Result<(), anyhow::Error> {
    let conn = pool.get().await?;
    conn.batch_execute(
        r#"
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

CREATE INDEX IF NOT EXISTS idx_agent_events_repo_created
ON agent_events(repo, created_at DESC);
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
        "INSERT INTO agent_sessions (id, repo, task, actor)
         VALUES ($1, $2, $3, $4)",
        &[&session_id, &repo, &task, &actor],
    )
    .await?;
    Ok(())
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
    let rows = conn.query(
        "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, created_at
         FROM agent_events
         WHERE repo = $1
         ORDER BY created_at DESC
         LIMIT $2",
        &[&repo, &limit],
    )
    .await?;

    let events: Vec<AgentEvent> = rows
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

pub async fn append_event_from_request(
    pool: &Pool,
    qdrant_url: &str,
    req: &crate::state::AppendEventRequest,
) -> Result<String, anyhow::Error> {
    let id = Uuid::new_v4().to_string();
    let actor = req.actor.as_deref().unwrap_or("agent");
    let metadata = req.metadata.as_ref().cloned().unwrap_or_else(|| serde_json::json!({}));

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
    crate::qdrant::store_event(qdrant_url, &event).await?;

    Ok(id)
}

// ── Context pack builder ──────────────────────────────────────

pub async fn check_ready(pool: &deadpool_postgres::Pool) -> Result<(), anyhow::Error> {
    let conn = pool.get().await?;
    let _ = conn.query_one("SELECT 1", &[]).await?;
    Ok(())
}

pub fn build_context(
    repo: &str,
    task: &str,
    memories: &[crate::state::EventMemory],
) -> String {
    let memory_text = if memories.is_empty() {
        "- No prior memory found.".to_string()
    } else {
        memories
            .iter()
            .map(|m| {
                if let Some(evidence) = &m.evidence {
                    format!("- [{}] {} Evidence: {}", m.event_type, m.summary, evidence)
                } else {
                    format!("- [{}] {}", m.event_type, m.summary)
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "Repository: {}\nTask: {}\n\nRelevant prior memory:\n{}\n\nOperating rules:\n- Use prior failures and decisions as constraints.\n- Do not repeat known-bad attempts.\n- Prefer small, verifiable changes.\n- Record meaningful actions back to the orchestrator.\n",
        repo, task, memory_text
    )
}
