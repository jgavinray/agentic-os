use orchestrator::execution_feedback;
use std::ops::DerefMut;

use super::options::BackfillOptions;

pub(super) struct BackfillReport {
    pub(super) pending: i64,
    pub(super) updated: i64,
    pub(super) dry_run: bool,
    pub(super) batch_size: i64,
}

pub(super) async fn backfill_signatures(
    pool: &deadpool_postgres::Pool,
    opts: BackfillOptions,
) -> Result<BackfillReport, anyhow::Error> {
    let pending = count_pending_backfill(pool).await?;
    if opts.dry_run || pending == 0 {
        return Ok(BackfillReport {
            pending,
            updated: 0,
            dry_run: opts.dry_run,
            batch_size: opts.batch_size,
        });
    }

    let mut updated = 0i64;
    loop {
        let batch = load_backfill_batch(pool, opts.batch_size).await?;
        if batch.is_empty() {
            break;
        }

        let mut conn = pool.get().await?;
        let tx = conn.deref_mut().deref_mut().transaction().await?;
        for row in batch {
            let Some(updated_metadata) = execution_feedback::backfill_inline_signature_metadata(
                &row.metadata,
                row.legacy_raw_excerpt.as_deref(),
            ) else {
                continue;
            };
            let changed = tx
                .execute(
                    "UPDATE agent_events
                     SET metadata = $2
                     WHERE id = $1
                       AND (
                           metadata->'payload'->>'signature' IS NULL
                           OR metadata->'payload'->>'signature_category' IS NULL
                           OR metadata->'payload'->>'fingerprint_version' IS NULL
                       )",
                    &[&row.id, &updated_metadata],
                )
                .await?;
            updated += changed as i64;
        }
        tx.commit().await?;
    }

    Ok(BackfillReport {
        pending,
        updated,
        dry_run: opts.dry_run,
        batch_size: opts.batch_size,
    })
}

async fn count_pending_backfill(pool: &deadpool_postgres::Pool) -> Result<i64, anyhow::Error> {
    let conn = pool.get().await?;
    let outcome_event_types = execution_feedback::FAILURE_OUTCOME_EVENT_TYPES.to_vec();
    let row = conn
        .query_one(
            "SELECT count(*)::BIGINT AS count
             FROM agent_events
             WHERE event_type = ANY($1)
               AND metadata->>'success' = 'false'
               AND (
                   metadata->'payload'->>'signature' IS NULL
                   OR metadata->'payload'->>'signature_category' IS NULL
                   OR metadata->'payload'->>'fingerprint_version' IS NULL
               )",
            &[&outcome_event_types],
        )
        .await?;
    Ok(row.get("count"))
}

struct BackfillRow {
    id: String,
    metadata: serde_json::Value,
    legacy_raw_excerpt: Option<String>,
}

async fn load_backfill_batch(
    pool: &deadpool_postgres::Pool,
    batch_size: i64,
) -> Result<Vec<BackfillRow>, anyhow::Error> {
    let conn = pool.get().await?;
    let outcome_event_types = execution_feedback::FAILURE_OUTCOME_EVENT_TYPES.to_vec();
    let rows = conn
        .query(
            "SELECT e.id,
                    e.metadata,
                    fs.metadata->'payload'->>'raw_excerpt' AS legacy_raw_excerpt
             FROM agent_events e
             LEFT JOIN LATERAL (
                 SELECT metadata
                 FROM agent_events fs
                 WHERE fs.event_type = 'failure_signature'
                   AND fs.parent_event_id::text = e.id
                 ORDER BY fs.created_at DESC
                 LIMIT 1
             ) fs ON true
             WHERE e.event_type = ANY($1)
               AND e.metadata->>'success' = 'false'
               AND (
                   e.metadata->'payload'->>'signature' IS NULL
                   OR e.metadata->'payload'->>'signature_category' IS NULL
                   OR e.metadata->'payload'->>'fingerprint_version' IS NULL
               )
             ORDER BY e.created_at ASC, e.id ASC
             LIMIT $2",
            &[&outcome_event_types, &batch_size],
        )
        .await?;

    Ok(rows
        .into_iter()
        .map(|row| BackfillRow {
            id: row.get("id"),
            metadata: row.get("metadata"),
            legacy_raw_excerpt: row.get("legacy_raw_excerpt"),
        })
        .collect())
}
