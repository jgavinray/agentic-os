use crate::db::AgentEvent;
use crate::state::AppState;
use serde_json::json;

const SUMMARIZER_FAILURE_COOLDOWN_SEC: i64 = 1800;

pub(crate) async fn record_summarization_failure(
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
