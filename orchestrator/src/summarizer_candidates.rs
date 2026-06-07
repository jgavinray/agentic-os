use crate::state::AppState;
use crate::summarizer_promotion::{has_source_ids, should_promote_to_level};
use crate::telemetry;

pub(crate) type SummaryCandidate = (String, i32);

pub(crate) async fn candidate_sessions(
    conn: &deadpool_postgres::Object,
) -> Result<Vec<SummaryCandidate>, anyhow::Error> {
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

    Ok(rows
        .iter()
        .map(|r| (r.get("session_id"), r.get("target_level")))
        .collect())
}

pub(crate) async fn promotable_source_rows(
    state: &AppState,
    conn: &deadpool_postgres::Object,
    session_id: &str,
    target_level: i32,
    source_level: i32,
) -> Result<Vec<tokio_postgres::Row>, anyhow::Error> {
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
        let metadata: serde_json::Value = row.get("metadata");
        let accepted =
            should_promote_to_level(target_level, &event_type, &summary, &metadata, source_count);
        if target_level > 1 {
            telemetry::record_promotion(&state.metrics, accepted, has_source_ids(&metadata));
        }
        if accepted {
            promotable_rows.push(row);
        }
    }

    Ok(promotable_rows)
}

pub(crate) fn source_event_ids(rows: &[tokio_postgres::Row]) -> Vec<String> {
    rows.iter().map(|r| r.get("id")).collect()
}

pub(crate) fn source_messages_text(rows: &[tokio_postgres::Row]) -> String {
    rows.iter()
        .map(|r| {
            let role = if r.get::<_, String>("event_type") == "user_message" {
                "User"
            } else {
                "Assistant"
            };
            format!("{role}: {}", r.get::<_, String>("summary"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}
