use crate::harness_feedback::{annotate_event_metadata, is_quarantined, record_metadata_metrics};
use serde_json::Value;

#[derive(Debug, Clone, Default)]
pub struct BackfillOptions {
    pub repo: Option<String>,
    pub session_id: Option<String>,
    pub since: Option<chrono::DateTime<chrono::Utc>>,
    pub dry_run: bool,
    pub batch_size: i64,
}

#[derive(Debug, Clone, Default)]
pub struct BackfillReport {
    pub events_scanned: i64,
    pub updated: i64,
    pub quarantined: i64,
    pub dry_run: bool,
    pub batch_size: i64,
}

pub async fn run_backfill(
    pool: &deadpool_postgres::Pool,
    opts: &BackfillOptions,
) -> Result<BackfillReport, anyhow::Error> {
    let batch_size = opts.batch_size.max(1);
    let mut report = BackfillReport {
        dry_run: opts.dry_run,
        batch_size,
        ..BackfillReport::default()
    };
    let mut last_created_at: Option<chrono::DateTime<chrono::Utc>> = None;
    let mut last_id: Option<String> = None;

    loop {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, event_type, summary, evidence, metadata, created_at
                 FROM agent_events
                 WHERE ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TEXT IS NULL OR session_id = $2)
                   AND ($3::TIMESTAMPTZ IS NULL OR created_at >= $3)
                   AND (
                       $4::TIMESTAMPTZ IS NULL
                       OR created_at > $4
                       OR (created_at = $4 AND id > $5)
                   )
                 ORDER BY created_at ASC, id ASC
                 LIMIT $6",
                &[
                    &opts.repo,
                    &opts.session_id,
                    &opts.since,
                    &last_created_at,
                    &last_id,
                    &batch_size,
                ],
            )
            .await?;
        if rows.is_empty() {
            break;
        }

        for row in rows {
            let id: String = row.get("id");
            let event_type: String = row.get("event_type");
            let summary: String = row.get("summary");
            let evidence: Option<String> = row.get("evidence");
            let metadata: Value = row.get("metadata");
            let created_at: chrono::DateTime<chrono::Utc> = row.get("created_at");

            report.events_scanned += 1;
            last_created_at = Some(created_at);
            last_id = Some(id.clone());

            let updated = annotate_event_metadata(
                &event_type,
                &summary,
                evidence.as_deref(),
                metadata.clone(),
            );
            if updated == metadata {
                continue;
            }
            report.updated += 1;
            if is_quarantined(&updated) {
                report.quarantined += 1;
            }
            if opts.dry_run {
                continue;
            }
            conn.execute(
                "UPDATE agent_events
                 SET metadata = $2
                 WHERE id = $1 AND metadata IS DISTINCT FROM $2",
                &[&id, &updated],
            )
            .await?;
            record_metadata_metrics(&updated);
        }
    }

    Ok(report)
}
