use crate::feature_detection_tags::bootstrap_annotate_event_metadata;
use crate::feature_extraction::extract_records;
use crate::feature_extraction_persistence::persist_feature_record;
use crate::feature_extraction_types::{
    BackfillOptions, BackfillReport, ExtractionConfig, ExtractionReport, ExtractionScope,
};
use deadpool_postgres::Pool;
use std::ops::DerefMut;

pub use crate::feature_extraction_queries::{
    load_events_for_scope, operational_constraints_for_context,
};

pub async fn extract_and_persist_scope(
    pool: &Pool,
    scope: &ExtractionScope,
    config: &ExtractionConfig,
    dry_run: bool,
) -> Result<ExtractionReport, anyhow::Error> {
    let events = load_events_for_scope(pool, scope, None).await?;
    let report = extract_records(&events, config);
    if !dry_run {
        for record in &report.records {
            persist_feature_record(pool, record).await?;
        }
    }
    Ok(report)
}

pub async fn bootstrap_tag_events(
    pool: &Pool,
    scope: &ExtractionScope,
    batch_size: i64,
    dry_run: bool,
) -> Result<i64, anyhow::Error> {
    let events = load_events_for_scope(pool, scope, None).await?;
    let mut updates = Vec::new();
    for event in events {
        let updated = bootstrap_annotate_event_metadata(&event);
        if updated != event.metadata {
            updates.push((event.id, updated));
        }
    }
    if dry_run || updates.is_empty() {
        return Ok(updates.len() as i64);
    }

    let mut conn = pool.get().await?;
    let mut changed = 0i64;
    for chunk in updates.chunks(batch_size.max(1) as usize) {
        let tx = conn.deref_mut().deref_mut().transaction().await?;
        for (id, metadata) in chunk {
            changed += tx
                .execute(
                    "UPDATE agent_events
                     SET metadata = $2
                     WHERE id = $1 AND metadata IS DISTINCT FROM $2",
                    &[id, metadata],
                )
                .await? as i64;
        }
        tx.commit().await?;
    }
    Ok(changed)
}

pub async fn run_backfill(
    pool: &Pool,
    opts: &BackfillOptions,
    config: &ExtractionConfig,
) -> Result<BackfillReport, anyhow::Error> {
    let scope = ExtractionScope {
        repo: opts.repo.clone(),
        session_id: opts.session_id.clone(),
        trajectory_id: opts.trajectory_id,
        since: opts.since,
        until: None,
    };
    let mut report = BackfillReport {
        dry_run: opts.dry_run,
        ..Default::default()
    };
    if !opts.skip_bootstrap_tagging {
        report.bootstrap_updated =
            bootstrap_tag_events(pool, &scope, opts.batch_size, opts.dry_run).await?;
    }

    let mut events = load_events_for_scope(pool, &scope, None).await?;
    if opts.dry_run && !opts.skip_bootstrap_tagging {
        for event in &mut events {
            event.metadata = bootstrap_annotate_event_metadata(event);
        }
    }
    report.events_scanned = events.len();
    let extraction = extract_records(&events, config);
    report.feature_records = extraction.records.len();
    if !opts.dry_run {
        for record in extraction.records {
            persist_feature_record(pool, &record).await?;
            report.persisted_records += 1;
        }
    }
    Ok(report)
}
