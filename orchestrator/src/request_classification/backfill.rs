use chrono::{DateTime, Utc};
use deadpool_postgres::Pool;

use crate::request_classification_backfill_queries::load_classification_batch;
use crate::request_classification_types::{BackfillOptions, BackfillReport, PersistOutcome};

pub async fn run_backfill(
    pool: &Pool,
    opts: &BackfillOptions,
) -> Result<BackfillReport, anyhow::Error> {
    let result = run_backfill_inner(pool, opts).await;
    crate::telemetry::record_request_classification_backfill_run(if result.is_ok() {
        "success"
    } else {
        "failure"
    });
    result
}

async fn run_backfill_inner(
    pool: &Pool,
    opts: &BackfillOptions,
) -> Result<BackfillReport, anyhow::Error> {
    let batch_size = opts.batch_size.max(1);
    let mut report = BackfillReport {
        dry_run: opts.dry_run,
        batch_size,
        ..BackfillReport::default()
    };
    let mut last_created_at: Option<DateTime<Utc>> = None;
    let mut last_id: Option<String> = None;

    loop {
        let rows =
            load_classification_batch(pool, opts, batch_size, last_created_at, last_id.as_deref())
                .await?;
        if rows.is_empty() {
            break;
        }

        for row in rows {
            report.events_scanned += 1;
            last_created_at = Some(row.event.created_at);
            last_id = Some(row.event.id.clone());

            let should_repair = opts.repair || (opts.repair_stale && row.needs_stale_repair);
            if row.already_classified && !should_repair {
                report.skipped += 1;
                continue;
            }

            let classification = crate::request_classification::classify_request_event(&row.event);
            if opts.dry_run {
                if row.already_classified {
                    report.updated += 1;
                } else {
                    report.inserted += 1;
                }
                crate::telemetry::record_request_classification_write("dry_run");
                continue;
            }

            let outcome = if row.already_classified {
                crate::request_classification_store::update_classification_if_changed(
                    pool,
                    &classification,
                )
                .await?
            } else {
                crate::request_classification_store::persist_classification(pool, &classification)
                    .await?
            };
            match outcome {
                PersistOutcome::Inserted => report.inserted += 1,
                PersistOutcome::Updated => report.updated += 1,
                PersistOutcome::Skipped => report.skipped += 1,
            }
        }
    }

    Ok(report)
}
