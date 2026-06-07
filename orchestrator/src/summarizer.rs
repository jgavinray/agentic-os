use crate::db::AgentEvent;
use crate::state::AppState;
use crate::summarizer_failures::record_summarization_failure;
use crate::summarizer_levels::{source_level_for_target, summary_prompt_for_level};
pub(crate) use crate::summarizer_promotion::{has_source_ids, should_promote_to_level};
use crate::telemetry;
use serde_json::Value;
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
    let rows = conn
        .query(
            "WITH candidates AS (
                 SELECT session_id, 1 AS target_level
                 FROM agent_events
                 WHERE event_type IN ('user_message', 'assistant_message')
                   AND summary_level = 0
                   AND summarized = false
                 GROUP BY session_id
                 HAVING count(*) > 20
                 UNION ALL
                 SELECT session_id, 2 AS target_level
                 FROM agent_events
                 WHERE event_type = 'summary'
                   AND summary_level = 1
                   AND summarized = false
                 GROUP BY session_id
                 HAVING count(*) > 20
                 UNION ALL
                 SELECT session_id, 3 AS target_level
                 FROM agent_events
                 WHERE event_type = 'summary'
                   AND summary_level = 2
                   AND summarized = false
                 GROUP BY session_id
                 HAVING count(*) > 200
             )
             SELECT session_id, target_level FROM candidates",
            &[],
        )
        .await?;
    let candidates: Vec<(String, i32)> = rows
        .iter()
        .map(|r| (r.get("session_id"), r.get("target_level")))
        .collect();
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

    let rows = conn
        .query(
            "SELECT id, event_type, summary, metadata
             FROM agent_events
             WHERE session_id = $1
               AND summary_level = $2
               AND summarized = false
               AND (
                   ($2 = 0 AND event_type IN ('user_message', 'assistant_message'))
                   OR ($2 > 0 AND event_type = 'summary')
               )
             ORDER BY created_at ASC
             LIMIT $3",
            &[
                &session_id,
                &source_level,
                &crate::state::MAX_SUMMARIZER_EVENTS,
            ],
        )
        .await?;
    let source_count = rows.len();
    let mut promotable_rows = Vec::new();
    for row in rows {
        let event_type: String = row.get("event_type");
        let summary: String = row.get("summary");
        let metadata: Value = row.get("metadata");
        let accepted =
            should_promote_to_level(target_level, &event_type, &summary, &metadata, source_count);
        if target_level > 1 {
            telemetry::record_promotion(&state.metrics, accepted, has_source_ids(&metadata));
        }
        if accepted {
            promotable_rows.push(row);
        }
    }
    let rows = promotable_rows;

    if rows.is_empty() {
        return Ok(());
    }

    let event_ids: Vec<String> = rows.iter().map(|r| r.get("id")).collect();

    let messages_text = rows
        .iter()
        .map(|r| {
            let role = if r.get::<_, String>("event_type") == "user_message" {
                "User"
            } else {
                "Assistant"
            };
            format!("{role}: {}", r.get::<_, String>("summary"))
        })
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = prompt_template.replace("{messages}", &messages_text);

    let request_body = serde_json::json!({
        "model": state.summarizer_model.clone(),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": state.summarizer_max_tokens,
        "temperature": 0.1,
    });

    let url = format!("{}/chat/completions", state.summarizer_url);
    let upstream_started = std::time::Instant::now();
    let mut request = state.http.post(&url).json(&request_body);
    if let Some(key) = &state.summarizer_key {
        request = request.bearer_auth(key);
    }
    let resp = request.send().await?;
    let status = resp.status();
    telemetry::record_upstream_summarizer(
        "chat_completions",
        upstream_started.elapsed(),
        &status.as_u16().to_string(),
    );
    if !status.is_success() {
        telemetry::record_upstream_summarizer_error(
            "chat_completions",
            telemetry::upstream_error_kind(status),
        );
    }
    let val: serde_json::Value = resp.json().await.inspect_err(|_| {
        telemetry::record_upstream_summarizer_error("chat_completions", "parse");
    })?;
    telemetry::record_tokens(
        &state.metrics,
        &crate::state::TokenUsage::from_openai_value(&val),
        &state.summarizer_model,
    );

    let content_val = &val["choices"][0]["message"]["content"];
    let summary_text = match content_val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => anyhow::bail!("no content in summarization response"),
    };

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
