use crate::db::AgentEvent;
use crate::state::AppState;
use crate::summarizer_candidates::{
    candidate_sessions, promotable_source_rows, source_event_ids, source_messages_text,
};
use crate::summarizer_failures::record_summarization_failure;
use crate::summarizer_levels::{source_level_for_target, summary_prompt_for_level};
use crate::summarizer_upstream::request_summary;
use crate::telemetry;
use std::sync::Arc;
use tokio::time::{interval, Duration};

pub async fn run(state: Arc<AppState>) {
    let mut tick = interval(Duration::from_secs(60));
    loop {
        tick.tick().await;
        telemetry::record_summarizer_tick();
        if let Err(e) = check_sessions(Arc::clone(&state)).await {
            tracing::warn!(target: "summarizer", "check_sessions error: {e}");
        }
    }
}

async fn check_sessions(state: Arc<AppState>) -> Result<(), anyhow::Error> {
    let conn = state.pool.get().await?;
    let candidates = candidate_sessions(&conn).await?;
    drop(conn);

    for target_level in [1, 2, 3] {
        let count = candidates
            .iter()
            .filter(|(_, level)| *level == target_level)
            .count();
        telemetry::record_summarizer_candidate(target_level, count);
    }

    for (session_id, target_level) in candidates {
        summarize_session(Arc::clone(&state), session_id, target_level).await;
    }
    Ok(())
}

async fn summarize_session(state: Arc<AppState>, session_id: String, target_level: i32) {
    let conn = match state.pool.get().await {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(target: "summarizer", session_id = %session_id, "pool error: {e}");
            return;
        }
    };

    let lock_key: i64 = match conn
        .query_one(
            "SELECT ('x' || substr(md5($1), 1, 16))::bit(64)::bigint AS k",
            &[&format!("{session_id}:{target_level}")],
        )
        .await
    {
        Ok(r) => r.get("k"),
        Err(e) => {
            tracing::warn!(target: "summarizer", "hashtext failed: {e}");
            return;
        }
    };

    let acquired: bool = conn
        .query_one("SELECT pg_try_advisory_lock($1)", &[&lock_key])
        .await
        .map(|r| r.get(0))
        .unwrap_or(false);

    if !acquired {
        return;
    }

    if let Err(e) = do_summarize(&state, &conn, &session_id, target_level).await {
        telemetry::record_summarizer_written(target_level, false);
        let reason = e.to_string();
        match record_summarization_failure(&state, &conn, &session_id, target_level, &reason).await
        {
            Ok(true) => {
                tracing::debug!(
                    target: "summarizer",
                    session_id = %session_id,
                    target_level,
                    "summarization still failing; duplicate failure event suppressed: {reason}"
                );
            }
            Ok(false) => {
                tracing::warn!(
                    target: "summarizer",
                    session_id = %session_id,
                    target_level,
                    "summarization failed: {reason}"
                );
            }
            Err(record_err) => {
                tracing::warn!(
                    target: "summarizer",
                    session_id = %session_id,
                    target_level,
                    "summarization failed: {reason}; failed to record summarization failure event: {record_err}"
                );
            }
        }
    }

    let _ = conn
        .execute("SELECT pg_advisory_unlock($1)", &[&lock_key])
        .await;
}

async fn do_summarize(
    state: &AppState,
    conn: &deadpool_postgres::Object,
    session_id: &str,
    target_level: i32,
) -> Result<(), anyhow::Error> {
    let started = std::time::Instant::now();
    let source_level = source_level_for_target(target_level)?;
    let prompt_template = summary_prompt_for_level(target_level)?;

    let rows = promotable_source_rows(state, conn, session_id, target_level, source_level).await?;

    if rows.is_empty() {
        return Ok(());
    }

    let event_ids = source_event_ids(&rows);
    let messages_text = source_messages_text(&rows);

    let prompt = prompt_template.replace("{messages}", &messages_text);

    let summary_text = request_summary(state, prompt).await?;

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
    telemetry::record_summarizer_written(target_level, true);
    telemetry::record_summarizer_duration(target_level, started.elapsed());

    tracing::info!(
        target: "summarizer",
        session_id = %session_id,
        summarized_count = event_ids.len(),
        summary_id = %summary_id,
        target_level,
        "summarized session messages"
    );

    Ok(())
}
