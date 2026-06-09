use crate::db::AgentEvent;
use crate::state::AppState;

pub(crate) struct PersistedSummary {
    pub(crate) summary_id: String,
    pub(crate) summarized_count: usize,
}

pub(crate) async fn persist_summary_event(
    state: &AppState,
    conn: &deadpool_postgres::Object,
    session_id: &str,
    target_level: i32,
    source_level: i32,
    event_ids: &[String],
    summary_text: String,
) -> Result<PersistedSummary, anyhow::Error> {
    let session_row = conn
        .query_one(
            "SELECT repo FROM agent_sessions WHERE id = $1",
            &[&session_id],
        )
        .await?;
    let repo: String = session_row.get("repo");

    let summary_id = uuid::Uuid::new_v4().to_string();
    let metadata = serde_json::json!({
        "summarized_event_ids": event_ids,
        "summary_version": target_level,
        "source_summary_level": source_level,
    });

    conn.execute(
        "INSERT INTO agent_events (id, session_id, repo, actor, event_type, summary, metadata, summary_level)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        &[
            &summary_id,
            &session_id,
            &repo,
            &"summarizer",
            &"summary",
            &summary_text,
            &metadata,
            &target_level,
        ],
    )
    .await?;

    let summary_event = AgentEvent {
        id: summary_id.clone(),
        session_id: session_id.to_string(),
        repo: repo.clone(),
        actor: "summarizer".to_string(),
        event_type: "summary".to_string(),
        summary: summary_text,
        evidence: None,
        metadata,
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: None,
        created_at: chrono::Utc::now(),
        summary_level: target_level,
    };

    if let Err(e) =
        crate::qdrant::store_event(&state.embedder, &state.qdrant_url, &summary_event).await
    {
        tracing::warn!(
            target: "summarizer",
            session_id = %session_id,
            "qdrant embed failed for summary (event still stored): {e}"
        );
    }

    conn.execute(
        "UPDATE agent_events SET summarized = true WHERE id = ANY($1)",
        &[&event_ids],
    )
    .await?;

    Ok(PersistedSummary {
        summary_id,
        summarized_count: event_ids.len(),
    })
}
