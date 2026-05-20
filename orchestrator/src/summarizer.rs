use crate::db::AgentEvent;
use crate::state::AppState;
use crate::telemetry;
use serde_json::json;
use serde_json::Value;
use std::sync::Arc;
use tokio::time::{interval, Duration};

const SUMMARIZER_FAILURE_COOLDOWN_SEC: i64 = 1800;

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

async fn record_summarization_failure(
    state: &AppState,
    conn: &deadpool_postgres::Object,
    session_id: &str,
    target_level: i32,
    reason: &str,
) -> Result<bool, anyhow::Error> {
    let reason = crate::execution_feedback::summarize_text(reason, 500);
    let target_level_text = target_level.to_string();
    let recent_duplicate: bool = conn
        .query_one(
            "SELECT EXISTS (
                 SELECT 1
                 FROM agent_events
                 WHERE session_id = $1
                   AND actor = 'summarizer'
                   AND event_type = 'failed_attempt'
                   AND metadata->'payload'->>'failure_class' = 'summarization_failure'
                   AND metadata->'payload'->>'target_level' = $2
                   AND metadata->'payload'->>'failure_reason' = $3
                   AND created_at > now() - ($4::text || ' seconds')::interval
             ) AS exists",
            &[
                &session_id,
                &target_level_text,
                &reason,
                &SUMMARIZER_FAILURE_COOLDOWN_SEC.to_string(),
            ],
        )
        .await?
        .get("exists");
    if recent_duplicate {
        return Ok(true);
    }

    let session_row = conn
        .query_one(
            "SELECT repo FROM agent_sessions WHERE id = $1",
            &[&session_id],
        )
        .await?;
    let repo: String = session_row.get("repo");
    let summary = format!("summarization failed: {reason}");
    let metadata = crate::feature_extraction::annotate_event_metadata(
        "failed_attempt",
        &summary,
        None,
        json!({
            "producer_signals": {
                "summarizer": [{"type": "summarization_failure"}]
            },
            "success": false,
            "payload": {
                "failure_class": "summarization_failure",
                "failure_reason": reason,
                "target_level": target_level,
            }
        }),
    );
    let event = AgentEvent {
        id: uuid::Uuid::new_v4().to_string(),
        session_id: session_id.to_string(),
        repo: repo.clone(),
        actor: "summarizer".to_string(),
        event_type: "failed_attempt".to_string(),
        summary,
        evidence: None,
        metadata,
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: None,
        created_at: chrono::Utc::now(),
        summary_level: 0,
    };
    crate::db::insert_event(&state.pool, &event).await?;
    if state.feature_extraction_enabled {
        crate::feature_extraction::run_inline_extraction_best_effort(
            state.pool.clone(),
            state.metrics.clone(),
            repo,
            session_id.to_string(),
            None,
        )
        .await;
    }

    Ok(false)
}

pub(crate) fn should_promote_to_level(
    target_level: i32,
    event_type: &str,
    summary: &str,
    metadata: &Value,
    source_count: usize,
) -> bool {
    match target_level {
        1 => true,
        2 => {
            has_source_ids(metadata)
                && (source_count >= 2
                    || has_l2_signal(event_type, summary, metadata)
                    || has_failed_or_remediated_content(summary))
        }
        3 => {
            has_source_ids(metadata)
                && (metadata_flag(metadata, "stable")
                    || metadata_flag(metadata, "manually_confirmed")
                    || metadata_flag(metadata, "architecture_truth")
                    || (source_count >= 2 && is_architecture_related(summary)))
        }
        _ => false,
    }
}

pub(crate) fn has_source_ids(metadata: &Value) -> bool {
    metadata
        .get("source_event_ids")
        .or_else(|| metadata.get("summarized_event_ids"))
        .and_then(|v| v.as_array())
        .map(|ids| !ids.is_empty())
        .unwrap_or(false)
}

fn has_l2_signal(event_type: &str, summary: &str, metadata: &Value) -> bool {
    metadata_flag(metadata, "important")
        || metadata_flag(metadata, "referenced")
        || metadata.get("last_referenced_at").is_some()
        || metadata
            .get("referenced_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
            > 0
        || matches!(event_type, "failed_attempt" | "remediation")
        || is_architecture_related(summary)
}

fn metadata_flag(metadata: &Value, key: &str) -> bool {
    metadata.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
}

fn has_failed_or_remediated_content(summary: &str) -> bool {
    let lower = summary.to_lowercase();
    lower.contains("failed")
        || lower.contains("failure")
        || lower.contains("remediation")
        || lower.contains("do not retry")
}

fn is_architecture_related(summary: &str) -> bool {
    let lower = summary.to_lowercase();
    crate::state::TaskCategory::ARCHITECTURE_KEYWORDS
        .iter()
        .any(|kw| lower.contains(kw))
}

pub(crate) fn source_level_for_target(target_level: i32) -> Result<i32, anyhow::Error> {
    match target_level {
        1 => Ok(0),
        2 => Ok(1),
        3 => Ok(2),
        _ => anyhow::bail!("invalid summary target level: {target_level}"),
    }
}

pub(crate) fn summary_prompt_for_level(target_level: i32) -> Result<&'static str, anyhow::Error> {
    let idx = usize::try_from(target_level - 1).unwrap_or(usize::MAX);
    crate::state::SUMMARY_PROMPTS
        .get(idx)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("invalid summary target level: {target_level}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn promotion_sequence() {
        assert_eq!(source_level_for_target(1).unwrap(), 0);
        assert_eq!(source_level_for_target(2).unwrap(), 1);
        assert_eq!(source_level_for_target(3).unwrap(), 2);
        assert!(source_level_for_target(4).is_err());

        assert_ne!(
            summary_prompt_for_level(1).unwrap(),
            summary_prompt_for_level(2).unwrap()
        );
        assert_ne!(
            summary_prompt_for_level(2).unwrap(),
            summary_prompt_for_level(3).unwrap()
        );
    }

    #[test]
    fn l1_to_l2_requires_promotion_signal() {
        let no_sources = json!({});
        assert!(!should_promote_to_level(
            2,
            "summary",
            "ordinary session summary",
            &no_sources,
            10
        ));

        let with_sources = json!({"summarized_event_ids": ["e1"]});
        assert!(!should_promote_to_level(
            2,
            "summary",
            "ordinary session summary",
            &with_sources,
            1
        ));

        let important = json!({"summarized_event_ids": ["e1"], "important": true});
        assert!(should_promote_to_level(
            2,
            "summary",
            "repo convention should be reused",
            &important,
            1
        ));

        let repeated = json!({"summarized_event_ids": ["e1", "e2"]});
        assert!(should_promote_to_level(
            2,
            "summary",
            "same retry mistake happened again",
            &repeated,
            2
        ));
    }

    #[test]
    fn l2_to_l3_requires_stability_or_confirmation() {
        let with_sources = json!({"summarized_event_ids": ["s1"]});
        assert!(!should_promote_to_level(
            3,
            "summary",
            "repo-level cache convention",
            &with_sources,
            5
        ));

        let stable = json!({"summarized_event_ids": ["s1"], "stable": true});
        assert!(should_promote_to_level(
            3,
            "summary",
            "repo-level cache convention",
            &stable,
            1
        ));

        let confirmed = json!({"summarized_event_ids": ["s1"], "manually_confirmed": true});
        assert!(should_promote_to_level(
            3,
            "summary",
            "durable project memory",
            &confirmed,
            1
        ));

        let no_sources = json!({"stable": true});
        assert!(!should_promote_to_level(
            3,
            "summary",
            "stable but unverifiable",
            &no_sources,
            5
        ));
    }
}
