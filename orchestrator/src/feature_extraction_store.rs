use crate::db::AgentEvent;
use crate::feature_constraints::enforce_constraint_token_budget;
use crate::feature_detection_tags::bootstrap_annotate_event_metadata;
use crate::feature_extraction::extract_records;
use crate::feature_extraction_types::{
    constraint_freshness_window_sec_from_env, feature_window_sec_from_env,
    max_operational_constraints_from_env, BackfillOptions, BackfillReport, ExtractionConfig,
    ExtractionReport, ExtractionScope, FeatureRecord, OperationalConstraint, SuppressedConstraint,
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

pub async fn persist_feature_record(
    pool: &Pool,
    record: &FeatureRecord,
) -> Result<(), anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let failure_classes = serde_json::to_value(&record.failure_classes)?;
        let recommended_constraints = serde_json::to_value(&record.recommended_constraints)?;
        let suppressed_constraints = serde_json::to_value(&record.suppressed_constraints)?;
        conn.execute(
            "INSERT INTO agent_feature_records (
                feature_id, trajectory_id, session_id, repo, window_start, window_end,
                event_count, user_message_count, assistant_message_count, failed_attempt_count,
                summary_count, tool_loop_count, repeated_read_loop_count, repeated_bash_loop_count,
                user_interruption_count, missing_auth_count, wrong_endpoint_count,
                summarization_failure_count, migration_failure_count, other_failure_count,
                context_pack_empty_count, context_pack_truncated_count, high_input_token_count,
                slow_upstream_model_count, empty_tool_use_message_count,
                abandoned_before_model_count, single_model_abandoned_no_tools_count,
                summarizer_shared_upstream_count,
                known_endpoint, known_auth_header, known_migration_fix, loop_detected,
                user_interrupted, failure_classes, recommended_constraints, suppressed_constraints,
                created_at, updated_at
             )
             VALUES (
                $1, $2, $3, $4, $5, $6,
                $7, $8, $9, $10,
                $11, $12, $13, $14,
                $15, $16, $17,
                $18, $19, $20,
                $21, $22, $23, $24,
                $25, $26, $27, $28,
                $29, $30, $31, $32,
                $33, $34, $35, $36,
                now(), now()
             )
             ON CONFLICT (feature_id)
             DO UPDATE SET
                trajectory_id = EXCLUDED.trajectory_id,
                session_id = EXCLUDED.session_id,
                repo = EXCLUDED.repo,
                window_start = EXCLUDED.window_start,
                window_end = EXCLUDED.window_end,
                event_count = EXCLUDED.event_count,
                user_message_count = EXCLUDED.user_message_count,
                assistant_message_count = EXCLUDED.assistant_message_count,
                failed_attempt_count = EXCLUDED.failed_attempt_count,
                summary_count = EXCLUDED.summary_count,
                tool_loop_count = EXCLUDED.tool_loop_count,
                repeated_read_loop_count = EXCLUDED.repeated_read_loop_count,
                repeated_bash_loop_count = EXCLUDED.repeated_bash_loop_count,
                user_interruption_count = EXCLUDED.user_interruption_count,
                missing_auth_count = EXCLUDED.missing_auth_count,
                wrong_endpoint_count = EXCLUDED.wrong_endpoint_count,
                summarization_failure_count = EXCLUDED.summarization_failure_count,
                migration_failure_count = EXCLUDED.migration_failure_count,
                other_failure_count = EXCLUDED.other_failure_count,
                context_pack_empty_count = EXCLUDED.context_pack_empty_count,
                context_pack_truncated_count = EXCLUDED.context_pack_truncated_count,
                high_input_token_count = EXCLUDED.high_input_token_count,
                slow_upstream_model_count = EXCLUDED.slow_upstream_model_count,
                empty_tool_use_message_count = EXCLUDED.empty_tool_use_message_count,
                abandoned_before_model_count = EXCLUDED.abandoned_before_model_count,
                single_model_abandoned_no_tools_count = EXCLUDED.single_model_abandoned_no_tools_count,
                summarizer_shared_upstream_count = EXCLUDED.summarizer_shared_upstream_count,
                known_endpoint = EXCLUDED.known_endpoint,
                known_auth_header = EXCLUDED.known_auth_header,
                known_migration_fix = EXCLUDED.known_migration_fix,
                loop_detected = EXCLUDED.loop_detected,
                user_interrupted = EXCLUDED.user_interrupted,
                failure_classes = EXCLUDED.failure_classes,
                recommended_constraints = EXCLUDED.recommended_constraints,
                suppressed_constraints = EXCLUDED.suppressed_constraints,
                updated_at = CASE
                    WHEN (
                        agent_feature_records.trajectory_id,
                        agent_feature_records.session_id,
                        agent_feature_records.repo,
                        agent_feature_records.window_start,
                        agent_feature_records.window_end,
                        agent_feature_records.event_count,
                        agent_feature_records.user_message_count,
                        agent_feature_records.assistant_message_count,
                        agent_feature_records.failed_attempt_count,
                        agent_feature_records.summary_count,
                        agent_feature_records.tool_loop_count,
                        agent_feature_records.repeated_read_loop_count,
                        agent_feature_records.repeated_bash_loop_count,
                        agent_feature_records.user_interruption_count,
                        agent_feature_records.missing_auth_count,
                        agent_feature_records.wrong_endpoint_count,
                        agent_feature_records.summarization_failure_count,
                        agent_feature_records.migration_failure_count,
                        agent_feature_records.other_failure_count,
                        agent_feature_records.context_pack_empty_count,
                        agent_feature_records.context_pack_truncated_count,
                        agent_feature_records.high_input_token_count,
                        agent_feature_records.slow_upstream_model_count,
                        agent_feature_records.empty_tool_use_message_count,
                        agent_feature_records.abandoned_before_model_count,
                        agent_feature_records.single_model_abandoned_no_tools_count,
                        agent_feature_records.summarizer_shared_upstream_count,
                        agent_feature_records.known_endpoint,
                        agent_feature_records.known_auth_header,
                        agent_feature_records.known_migration_fix,
                        agent_feature_records.loop_detected,
                        agent_feature_records.user_interrupted,
                        agent_feature_records.failure_classes,
                        agent_feature_records.recommended_constraints,
                        agent_feature_records.suppressed_constraints
                    ) IS DISTINCT FROM (
                        EXCLUDED.trajectory_id,
                        EXCLUDED.session_id,
                        EXCLUDED.repo,
                        EXCLUDED.window_start,
                        EXCLUDED.window_end,
                        EXCLUDED.event_count,
                        EXCLUDED.user_message_count,
                        EXCLUDED.assistant_message_count,
                        EXCLUDED.failed_attempt_count,
                        EXCLUDED.summary_count,
                        EXCLUDED.tool_loop_count,
                        EXCLUDED.repeated_read_loop_count,
                        EXCLUDED.repeated_bash_loop_count,
                        EXCLUDED.user_interruption_count,
                        EXCLUDED.missing_auth_count,
                        EXCLUDED.wrong_endpoint_count,
                        EXCLUDED.summarization_failure_count,
                        EXCLUDED.migration_failure_count,
                        EXCLUDED.other_failure_count,
                        EXCLUDED.context_pack_empty_count,
                        EXCLUDED.context_pack_truncated_count,
                        EXCLUDED.high_input_token_count,
                        EXCLUDED.slow_upstream_model_count,
                        EXCLUDED.empty_tool_use_message_count,
                        EXCLUDED.abandoned_before_model_count,
                        EXCLUDED.single_model_abandoned_no_tools_count,
                        EXCLUDED.summarizer_shared_upstream_count,
                        EXCLUDED.known_endpoint,
                        EXCLUDED.known_auth_header,
                        EXCLUDED.known_migration_fix,
                        EXCLUDED.loop_detected,
                        EXCLUDED.user_interrupted,
                        EXCLUDED.failure_classes,
                        EXCLUDED.recommended_constraints,
                        EXCLUDED.suppressed_constraints
                    )
                    THEN now()
                    ELSE agent_feature_records.updated_at
                END",
            &[
                &record.feature_id,
                &record.trajectory_id,
                &record.session_id,
                &record.repo,
                &record.window_start,
                &record.window_end,
                &record.event_count,
                &record.user_message_count,
                &record.assistant_message_count,
                &record.failed_attempt_count,
                &record.summary_count,
                &record.tool_loop_count,
                &record.repeated_read_loop_count,
                &record.repeated_bash_loop_count,
                &record.user_interruption_count,
                &record.missing_auth_count,
                &record.wrong_endpoint_count,
                &record.summarization_failure_count,
                &record.migration_failure_count,
                &record.other_failure_count,
                &record.context_pack_empty_count,
                &record.context_pack_truncated_count,
                &record.high_input_token_count,
                &record.slow_upstream_model_count,
                &record.empty_tool_use_message_count,
                &record.abandoned_before_model_count,
                &record.single_model_abandoned_no_tools_count,
                &record.summarizer_shared_upstream_count,
                &record.known_endpoint,
                &record.known_auth_header,
                &record.known_migration_fix,
                &record.loop_detected,
                &record.user_interrupted,
                &failure_classes,
                &recommended_constraints,
                &suppressed_constraints,
            ],
        )
        .await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("feature_persist_record", started.elapsed(), result.is_ok());
    result
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
