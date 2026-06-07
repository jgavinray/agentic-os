use crate::db::AgentEvent;
use crate::feature_constraints::enforce_constraint_token_budget;
use crate::feature_detection_tags::bootstrap_annotate_event_metadata;
use crate::feature_extraction::extract_records;
use crate::feature_extraction_persistence::persist_feature_record;
use crate::feature_extraction_types::{
    constraint_freshness_window_sec_from_env, feature_window_sec_from_env,
    max_operational_constraints_from_env, BackfillOptions, BackfillReport, ExtractionConfig,
    ExtractionReport, ExtractionScope, OperationalConstraint, SuppressedConstraint,
};
use chrono::Utc;
use deadpool_postgres::Pool;
use serde_json::Value;
use std::ops::DerefMut;
use uuid::Uuid;

pub async fn load_events_for_scope(
    pool: &Pool,
    scope: &ExtractionScope,
    limit: Option<i64>,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata,
                        correlation_id, parent_event_id, trajectory_id, attempt_index, event_role,
                        created_at, summary_level
                 FROM agent_events
                 WHERE ($1::text IS NULL OR repo = $1)
                   AND ($2::text IS NULL OR session_id = $2)
                   AND ($3::uuid IS NULL OR trajectory_id = $3)
                   AND ($4::timestamptz IS NULL OR created_at >= $4)
                   AND ($5::timestamptz IS NULL OR created_at < $5)
                 ORDER BY created_at ASC, id ASC
                 LIMIT COALESCE($6, 9223372036854775807)",
                &[
                    &scope.repo,
                    &scope.session_id,
                    &scope.trajectory_id,
                    &scope.since,
                    &scope.until,
                    &limit,
                ],
            )
            .await?;
        Ok(rows
            .into_iter()
            .map(|row| AgentEvent {
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
            })
            .collect())
    }
    .await;
    crate::telemetry::record_db_query("feature_load_events", started.elapsed(), result.is_ok());
    result
}

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

pub async fn operational_constraints_for_context(
    pool: &Pool,
    repo: &str,
    session_id: Option<&str>,
    trajectory_id: Option<Uuid>,
    token_budget: usize,
) -> Result<(Vec<OperationalConstraint>, Vec<SuppressedConstraint>), anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let row = if let Some(trajectory_id) = trajectory_id {
            conn.query_opt(
                "SELECT recommended_constraints
                 FROM agent_feature_records
                 WHERE repo = $1 AND trajectory_id = $2
                 ORDER BY updated_at DESC
                 LIMIT 1",
                &[&repo, &trajectory_id],
            )
            .await?
        } else if let Some(session_id) = session_id {
            conn.query_opt(
                "SELECT recommended_constraints
                 FROM agent_feature_records
                 WHERE repo = $1 AND session_id = $2
                 ORDER BY window_end DESC, updated_at DESC
                 LIMIT 1",
                &[&repo, &session_id],
            )
            .await?
        } else {
            conn.query_opt(
                "SELECT recommended_constraints
                 FROM agent_feature_records
                 WHERE repo = $1
                 ORDER BY updated_at DESC
                 LIMIT 1",
                &[&repo],
            )
            .await?
        };

        let constraints = row
            .map(|row| row.get::<_, Value>("recommended_constraints"))
            .and_then(|value| serde_json::from_value::<Vec<OperationalConstraint>>(value).ok())
            .unwrap_or_default();
        Ok(enforce_constraint_token_budget(&constraints, token_budget))
    }
    .await;
    crate::telemetry::record_db_query(
        "feature_context_constraints",
        started.elapsed(),
        result.is_ok(),
    );
    result
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

pub async fn run_inline_extraction_best_effort(
    pool: Pool,
    metrics: crate::telemetry::MetricsRegistry,
    repo: String,
    session_id: String,
    trajectory_id: Option<Uuid>,
) {
    let started = std::time::Instant::now();
    let config = ExtractionConfig {
        feature_window_sec: feature_window_sec_from_env(),
        constraint_freshness_window_sec: constraint_freshness_window_sec_from_env(),
        max_operational_constraints: max_operational_constraints_from_env(),
        evaluation_time: Utc::now(),
    };
    let scope = ExtractionScope {
        repo: Some(repo.clone()),
        session_id: Some(session_id),
        trajectory_id,
        since: None,
        until: None,
    };

    let events = match load_events_for_scope(&pool, &scope, None).await {
        Ok(events) => events,
        Err(e) => {
            tracing::warn!(
                target: "feature_extraction",
                repo = %repo,
                stage = "extraction",
                "inline feature extraction failed to load events: {e}"
            );
            crate::telemetry::record_feature_extraction_failure("extraction");
            return;
        }
    };
    let report = extract_records(&events, &config);
    for _ in 0..report.unknown_tag_schema_versions {
        crate::telemetry::record_unknown_tag_schema_version();
    }
    for record in &report.records {
        for class in &record.failure_classes {
            crate::telemetry::record_feature_failure_class(class);
        }
        if let Err(e) = persist_feature_record(&pool, record).await {
            tracing::warn!(
                target: "feature_extraction",
                repo = %repo,
                stage = "persistence",
                "inline feature extraction failed to persist record: {e}"
            );
            crate::telemetry::record_feature_extraction_failure("persistence");
            continue;
        }
        crate::telemetry::record_trajectory_feature_written(&metrics);
    }
    crate::telemetry::record_feature_extraction_duration(started.elapsed());
    drop(metrics);
}
