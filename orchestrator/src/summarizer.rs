use crate::db::AgentEvent;
use crate::state::AppState;
use std::sync::Arc;
use tokio::time::{interval, Duration};

const SUMMARIZE_PROMPT: &str = "\
You are a precise technical summarizer. Extract information from conversation messages \
into these exact sections. Include only what is explicitly stated. \
Output nothing else — no preamble, no explanation.

DECISIONS:
(one decision per line, or the word \"none\")
OPEN_QUESTIONS:
(one question per line, or the word \"none\")
FAILED_APPROACHES:
(one failed approach per line, or the word \"none\")
KEY_FACTS:
(one key fact per line, or the word \"none\")

Messages:
{messages}";

pub async fn run(state: Arc<AppState>) {
    let mut tick = interval(Duration::from_secs(60));
    loop {
        tick.tick().await;
        if let Err(e) = check_sessions(Arc::clone(&state)).await {
            tracing::warn!(target: "summarizer", "check_sessions error: {e}");
        }
    }
}

async fn check_sessions(state: Arc<AppState>) -> Result<(), anyhow::Error> {
    let conn = state.pool.get().await?;
    let rows = conn
        .query(
            "SELECT session_id
             FROM agent_events
             WHERE event_type IN ('user_message', 'assistant_message')
               AND summarized = false
             GROUP BY session_id
             HAVING count(*) > 20",
            &[],
        )
        .await?;
    let session_ids: Vec<String> = rows.iter().map(|r| r.get("session_id")).collect();
    drop(conn);

    for session_id in session_ids {
        tokio::spawn(summarize_session(Arc::clone(&state), session_id));
    }
    Ok(())
}

async fn summarize_session(state: Arc<AppState>, session_id: String) {
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
            &[&session_id],
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

    if let Err(e) = do_summarize(&state, &conn, &session_id).await {
        tracing::warn!(target: "summarizer", session_id = %session_id, "summarization failed: {e}");
    }

    let _ = conn
        .execute("SELECT pg_advisory_unlock($1)", &[&lock_key])
        .await;
}

async fn do_summarize(
    state: &AppState,
    conn: &deadpool_postgres::Object,
    session_id: &str,
) -> Result<(), anyhow::Error> {
    let rows = conn
        .query(
            "SELECT id, event_type, summary
             FROM agent_events
             WHERE session_id = $1
               AND event_type IN ('user_message', 'assistant_message')
               AND summarized = false
             ORDER BY created_at ASC
             LIMIT 10",
            &[&session_id],
        )
        .await?;

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

    let prompt = SUMMARIZE_PROMPT.replace("{messages}", &messages_text);

    let request_body = serde_json::json!({
        "model": "qwen36-35b-heretic",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.1,
    });

    let url = format!("{}/chat/completions", state.litellm_url);
    let val: serde_json::Value = state
        .http
        .post(&url)
        .bearer_auth(&state.litellm_key)
        .json(&request_body)
        .send()
        .await?
        .json()
        .await?;

    let summary_text = val["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("no content in summarization response"))?
        .to_string();

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
        "summary_version": 1,
    });

    conn.execute(
        "INSERT INTO agent_events (id, session_id, repo, actor, event_type, summary, metadata)
         VALUES ($1, $2, $3, $4, $5, $6, $7)",
        &[
            &summary_id,
            &session_id,
            &repo,
            &"summarizer",
            &"summary",
            &summary_text,
            &metadata,
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
        created_at: chrono::Utc::now(),
    };

    if let Err(e) = crate::qdrant::store_event(
        &state.http,
        &state.embedding_url,
        &state.qdrant_url,
        &summary_event,
    )
    .await
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

    tracing::info!(
        target: "summarizer",
        session_id = %session_id,
        summarized_count = event_ids.len(),
        summary_id = %summary_id,
        "summarized session messages"
    );

    Ok(())
}
