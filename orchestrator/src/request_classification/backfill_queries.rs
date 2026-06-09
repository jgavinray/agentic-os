use chrono::{DateTime, Utc};
use deadpool_postgres::Pool;

use crate::request_classification_types::{
    BackfillOptions, CLASSIFICATION_SCHEMA_VERSION, ROUTING_POLICY_VERSION,
};

pub(crate) struct ClassificationBatchRow {
    pub event: crate::db::AgentEvent,
    pub already_classified: bool,
    pub needs_stale_repair: bool,
}

pub(crate) async fn load_classification_batch(
    pool: &Pool,
    opts: &BackfillOptions,
    batch_size: i64,
    last_created_at: Option<DateTime<Utc>>,
    last_id: Option<&str>,
) -> Result<Vec<ClassificationBatchRow>, anyhow::Error> {
    let conn = pool.get().await?;
    let last_id = last_id.map(str::to_string);
    let rows = conn
        .query(
            "SELECT
                e.id,
                e.session_id,
                e.repo,
                e.actor,
                e.event_type,
                e.summary,
                e.evidence,
                e.metadata,
                e.correlation_id,
                e.parent_event_id,
                e.trajectory_id,
                e.attempt_index,
                e.event_role,
                e.created_at,
                e.summary_level,
                c.event_id IS NOT NULL AS already_classified,
                (
                    c.event_id IS NOT NULL
                    AND coalesce(
                        CASE
                            WHEN (c.features->>'char_count') ~ '^[0-9]+$'
                            THEN (c.features->>'char_count')::INTEGER
                            ELSE NULL
                        END,
                        -1
                    ) = 0
                ) AS needs_stale_repair
             FROM agent_events e
             LEFT JOIN agent_request_classifications c
               ON c.event_id = e.id
              AND c.classification_schema_version = $1
              AND c.routing_policy_version = $2
             WHERE ($3::TEXT IS NULL OR e.repo = $3)
               AND ($4::TEXT IS NULL OR e.session_id = $4)
               AND ($5::TIMESTAMPTZ IS NULL OR e.created_at >= $5)
               AND length(btrim(coalesce(e.summary, '') || coalesce(e.evidence, ''), E' \t\n\r')) > 0
               AND (
                   e.event_type = 'user_message'
                   OR e.event_role = 'request'
                   OR (
                       e.event_type = 'context_pack'
                       AND e.event_role = 'context_pack'
                       AND e.metadata ? 'request'
                   )
               )
               AND (
                   $6::TIMESTAMPTZ IS NULL
                   OR e.created_at > $6
                   OR (e.created_at = $6 AND e.id > $7)
               )
             ORDER BY e.created_at ASC, e.id ASC
             LIMIT $8",
            &[
                &CLASSIFICATION_SCHEMA_VERSION,
                &ROUTING_POLICY_VERSION,
                &opts.repo,
                &opts.session_id,
                &opts.since,
                &last_created_at,
                &last_id,
                &batch_size,
            ],
        )
        .await?;

    Ok(rows
        .into_iter()
        .map(|row| {
            let event = crate::db::AgentEvent {
                id: row.get("id"),
                session_id: row.get("session_id"),
                repo: row.get("repo"),
                actor: row.get("actor"),
                event_type: row.get("event_type"),
                summary: row.get("summary"),
                evidence: row.get("evidence"),
                metadata: row.get("metadata"),
                correlation_id: row.get("correlation_id"),
                parent_event_id: row.get("parent_event_id"),
                trajectory_id: row.get("trajectory_id"),
                attempt_index: row.get("attempt_index"),
                event_role: row.get("event_role"),
                created_at: row.get("created_at"),
                summary_level: row.get("summary_level"),
            };
            ClassificationBatchRow {
                event,
                already_classified: row.get("already_classified"),
                needs_stale_repair: row.get("needs_stale_repair"),
            }
        })
        .collect())
}
