use deadpool_postgres::Pool;
use uuid::Uuid;

use crate::db_types::AgentEvent;

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

/// Find an existing session for (repo, task) from the last 4 hours, or create one.
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
    crate::db::insert_event(pool, &event).await?;
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
