use chrono::{DateTime, Utc};
use deadpool_postgres::Pool;

use crate::request_classification_types::{
    BackfillOptions, BackfillReport, LabelCount, PersistOutcome, ReportOptions,
    RequestClassification, RequestClassificationReport, SessionRouteCount,
    CLASSIFICATION_SCHEMA_VERSION, ROUTING_POLICY_VERSION,
};

pub async fn classify_and_persist_event(
    pool: &Pool,
    event: &crate::db::AgentEvent,
) -> Result<Option<PersistOutcome>, anyhow::Error> {
    if !crate::request_classification::is_classifiable_request_event(event) {
        return Ok(None);
    }
    let classification = crate::request_classification::classify_request_event(event);
    let outcome = persist_classification(pool, &classification).await?;
    Ok(Some(outcome))
}

pub async fn persist_classification(
    pool: &Pool,
    classification: &RequestClassification,
) -> Result<PersistOutcome, anyhow::Error> {
    let result = async {
        let conn = pool.get().await?;
        let secondary_domains = classification
            .secondary_domains
            .iter()
            .map(|domain| domain.as_str().to_string())
            .collect::<Vec<_>>();
        let risk = classification
            .risk
            .iter()
            .map(|risk| risk.as_str().to_string())
            .collect::<Vec<_>>();
        let affected = conn
            .execute(
                "INSERT INTO agent_request_classifications (
                    event_id,
                    repo,
                    session_id,
                    trajectory_id,
                    event_created_at,
                    classified_at,
                    classification_schema_version,
                    routing_policy_version,
                    classifier_source,
                    intent,
                    domain,
                    secondary_domains,
                    artifact_type,
                    risk,
                    complexity,
                    recommended_route,
                    response_contract,
                    features
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9,
                    $10, $11, $12, $13, $14, $15, $16, $17, $18
                )
                ON CONFLICT (
                    event_id,
                    classification_schema_version,
                    routing_policy_version
                )
                DO NOTHING",
                &[
                    &classification.event_id,
                    &classification.repo,
                    &classification.session_id,
                    &classification.trajectory_id,
                    &classification.event_created_at,
                    &classification.classified_at,
                    &classification.classification_schema_version,
                    &classification.routing_policy_version,
                    &classification.classifier_source,
                    &classification.intent.as_str(),
                    &classification.domain.as_str(),
                    &secondary_domains,
                    &classification.artifact_type.as_str(),
                    &risk,
                    &classification.complexity.as_str(),
                    &classification.recommended_route.as_str(),
                    &classification.response_contract.as_str(),
                    &classification.features,
                ],
            )
            .await?;
        Ok::<PersistOutcome, anyhow::Error>(if affected == 1 {
            PersistOutcome::Inserted
        } else {
            PersistOutcome::Skipped
        })
    }
    .await;

    match &result {
        Ok(outcome) => {
            crate::telemetry::record_request_classification_write(outcome.as_str());
            if matches!(outcome, PersistOutcome::Inserted) {
                crate::request_classification::record_classification_metrics(classification);
            }
        }
        Err(_) => crate::telemetry::record_request_classification_write("error"),
    }
    result
}

pub async fn update_classification_if_changed(
    pool: &Pool,
    classification: &RequestClassification,
) -> Result<PersistOutcome, anyhow::Error> {
    let result = async {
        let conn = pool.get().await?;
        let secondary_domains = classification
            .secondary_domains
            .iter()
            .map(|domain| domain.as_str().to_string())
            .collect::<Vec<_>>();
        let risk = classification
            .risk
            .iter()
            .map(|risk| risk.as_str().to_string())
            .collect::<Vec<_>>();
        let affected = conn
            .execute(
                "UPDATE agent_request_classifications
                 SET
                    repo = $2,
                    session_id = $3,
                    trajectory_id = $4,
                    event_created_at = $5,
                    classified_at = $6,
                    classifier_source = $9,
                    intent = $10,
                    domain = $11,
                    secondary_domains = $12,
                    artifact_type = $13,
                    risk = $14,
                    complexity = $15,
                    recommended_route = $16,
                    response_contract = $17,
                    features = $18
                 WHERE event_id = $1
                   AND classification_schema_version = $7
                   AND routing_policy_version = $8
                   AND (
                       repo IS DISTINCT FROM $2
                       OR session_id IS DISTINCT FROM $3
                       OR trajectory_id IS DISTINCT FROM $4
                       OR event_created_at IS DISTINCT FROM $5
                       OR classified_at IS DISTINCT FROM $6
                       OR classifier_source IS DISTINCT FROM $9
                       OR intent IS DISTINCT FROM $10
                       OR domain IS DISTINCT FROM $11
                       OR secondary_domains IS DISTINCT FROM $12
                       OR artifact_type IS DISTINCT FROM $13
                       OR risk IS DISTINCT FROM $14
                       OR complexity IS DISTINCT FROM $15
                       OR recommended_route IS DISTINCT FROM $16
                       OR response_contract IS DISTINCT FROM $17
                       OR features IS DISTINCT FROM $18
                   )",
                &[
                    &classification.event_id,
                    &classification.repo,
                    &classification.session_id,
                    &classification.trajectory_id,
                    &classification.event_created_at,
                    &classification.classified_at,
                    &classification.classification_schema_version,
                    &classification.routing_policy_version,
                    &classification.classifier_source,
                    &classification.intent.as_str(),
                    &classification.domain.as_str(),
                    &secondary_domains,
                    &classification.artifact_type.as_str(),
                    &risk,
                    &classification.complexity.as_str(),
                    &classification.recommended_route.as_str(),
                    &classification.response_contract.as_str(),
                    &classification.features,
                ],
            )
            .await?;
        Ok::<PersistOutcome, anyhow::Error>(if affected == 1 {
            PersistOutcome::Updated
        } else {
            PersistOutcome::Skipped
        })
    }
    .await;

    match &result {
        Ok(outcome) => {
            crate::telemetry::record_request_classification_write(outcome.as_str());
            if matches!(outcome, PersistOutcome::Updated) {
                crate::request_classification::record_classification_metrics(classification);
            }
        }
        Err(_) => crate::telemetry::record_request_classification_write("error"),
    }
    result
}

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
                update_classification_if_changed(pool, &classification).await?
            } else {
                persist_classification(pool, &classification).await?
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

async fn load_classification_batch(
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

struct ClassificationBatchRow {
    event: crate::db::AgentEvent,
    already_classified: bool,
    needs_stale_repair: bool,
}

pub async fn request_classification_report(
    pool: &Pool,
    opts: &ReportOptions,
) -> Result<RequestClassificationReport, anyhow::Error> {
    let conn = pool.get().await?;
    let by_route = count_grouped(
        &conn,
        "recommended_route",
        opts.repo.as_deref(),
        opts.since,
        20,
    )
    .await?;
    let unknown_label_counts =
        count_unknown_labels(&conn, opts.repo.as_deref(), opts.since).await?;
    let repeated_guardrail_sessions =
        count_repeated_guardrail_sessions(&conn, opts.repo.as_deref(), opts.since).await?;
    let top_risk_flags = count_risk_flags(&conn, opts.repo.as_deref(), opts.since, 20).await?;

    Ok(RequestClassificationReport {
        by_route,
        top_risk_flags,
        unknown_label_counts,
        repeated_guardrail_sessions,
    })
}

async fn count_grouped(
    conn: &deadpool_postgres::Object,
    column: &str,
    repo: Option<&str>,
    since: Option<DateTime<Utc>>,
    limit: i64,
) -> Result<Vec<LabelCount>, anyhow::Error> {
    let column = match column {
        "recommended_route" => "recommended_route",
        "intent" => "intent",
        "domain" => "domain",
        "complexity" => "complexity",
        _ => anyhow::bail!("unsupported report column"),
    };
    let sql = format!(
        "SELECT {column} AS label, count(*)::BIGINT AS count
         FROM agent_request_classifications
         WHERE ($1::TEXT IS NULL OR repo = $1)
           AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
         GROUP BY {column}
         ORDER BY count DESC, {column} ASC
         LIMIT $3"
    );
    let rows = conn.query(&sql, &[&repo, &since, &limit]).await?;
    Ok(rows
        .into_iter()
        .map(|row| LabelCount {
            label: row.get("label"),
            count: row.get("count"),
        })
        .collect())
}

async fn count_risk_flags(
    conn: &deadpool_postgres::Object,
    repo: Option<&str>,
    since: Option<DateTime<Utc>>,
    limit: i64,
) -> Result<Vec<LabelCount>, anyhow::Error> {
    let rows = conn
        .query(
            "SELECT risk_label AS label, count(*)::BIGINT AS count
             FROM agent_request_classifications,
                  unnest(risk) AS risk_label
             WHERE ($1::TEXT IS NULL OR repo = $1)
               AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
             GROUP BY risk_label
             ORDER BY count DESC, risk_label ASC
             LIMIT $3",
            &[&repo, &since, &limit],
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| LabelCount {
            label: row.get("label"),
            count: row.get("count"),
        })
        .collect())
}

async fn count_unknown_labels(
    conn: &deadpool_postgres::Object,
    repo: Option<&str>,
    since: Option<DateTime<Utc>>,
) -> Result<Vec<LabelCount>, anyhow::Error> {
    let rows = conn
        .query(
            "SELECT field, sum(count)::BIGINT AS count
             FROM (
                 SELECT 'intent' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE intent = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
                 UNION ALL
                 SELECT 'domain' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE domain = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
                 UNION ALL
                 SELECT 'artifact_type' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE artifact_type = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
                 UNION ALL
                 SELECT 'complexity' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE complexity = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
                 UNION ALL
                 SELECT 'recommended_route' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE recommended_route = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
                 UNION ALL
                 SELECT 'response_contract' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE response_contract = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
             ) unknowns
             GROUP BY field
             ORDER BY field ASC",
            &[&repo, &since],
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| LabelCount {
            label: row.get("field"),
            count: row.get("count"),
        })
        .collect())
}

async fn count_repeated_guardrail_sessions(
    conn: &deadpool_postgres::Object,
    repo: Option<&str>,
    since: Option<DateTime<Utc>>,
) -> Result<Vec<SessionRouteCount>, anyhow::Error> {
    let rows = conn
        .query(
            "SELECT session_id, count(*)::BIGINT AS count
             FROM agent_request_classifications
             WHERE recommended_route = 'refuse_or_guardrail'
               AND ($1::TEXT IS NULL OR repo = $1)
               AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
             GROUP BY session_id
             HAVING count(*) > 1
             ORDER BY count DESC, session_id ASC
             LIMIT 20",
            &[&repo, &since],
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| SessionRouteCount {
            session_id: row.get("session_id"),
            count: row.get("count"),
        })
        .collect())
}
