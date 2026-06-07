use crate::db_event_rows::{row_to_event, rows_to_events};
use crate::db_trajectory::get_event_chain_by_event_id;
use crate::db_types::{AgentEvent, FailureHistoryItem};
use crate::state::ErrorRecord;
use deadpool_postgres::Pool;

pub async fn get_failure_history_for_signatures(
    pool: &Pool,
    repo: &str,
    signatures: &[String],
    limit: i64,
) -> Result<Vec<FailureHistoryItem>, anyhow::Error> {
    if signatures.is_empty() || limit <= 0 {
        return Ok(vec![]);
    }

    let started = std::time::Instant::now();
    let result = async {
        let outcome_event_types = crate::execution_feedback::FAILURE_OUTCOME_EVENT_TYPES.to_vec();
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
                 FROM agent_events
                 WHERE repo = $1
                   AND event_type = ANY($2)
                   AND metadata->>'success' = 'false'
                   AND metadata->'payload'->>'signature' = ANY($3)
                 ORDER BY created_at DESC
                 LIMIT $4",
                &[&repo, &outcome_event_types, &signatures, &limit],
            )
            .await?;

        let failures = rows_to_events(rows);
        let mut items = Vec::new();
        for failure in failures {
            let signature = event_payload_str(&failure, "signature").unwrap_or("unknown");
            let category = event_payload_str(&failure, "signature_category").unwrap_or("unknown");
            // The lookup is exact on canonical signature. Any semantic recall for
            // execution events continues to use the existing hybrid pipeline.
            let remediation = remediation_for_failure(pool, &failure, signature).await?;
            items.push(FailureHistoryItem {
                signature: signature.to_string(),
                category: category.to_string(),
                failure,
                remediation,
            });
        }
        Ok(items)
    }
    .await;
    crate::telemetry::record_db_query("get_failure_history", started.elapsed(), result.is_ok());
    result
}

pub async fn get_recent_failure_history(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<FailureHistoryItem>, anyhow::Error> {
    if limit <= 0 {
        return Ok(vec![]);
    }

    let started = std::time::Instant::now();
    let result = async {
        let outcome_event_types = crate::execution_feedback::FAILURE_OUTCOME_EVENT_TYPES.to_vec();
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata,
                        correlation_id, parent_event_id, trajectory_id, attempt_index,
                        event_role, created_at, summary_level
                 FROM agent_events
                 WHERE repo = $1
                   AND event_type = ANY($2)
                   AND metadata->>'success' = 'false'
                 ORDER BY created_at DESC
                 LIMIT $3",
                &[&repo, &outcome_event_types, &limit],
            )
            .await?;

        let failures = rows_to_events(rows);
        let mut items = Vec::new();
        for failure in failures {
            let signature = event_payload_str(&failure, "signature").unwrap_or("unknown");
            let category = event_payload_str(&failure, "signature_category").unwrap_or("unknown");
            let remediation = remediation_for_failure(pool, &failure, signature).await?;
            items.push(FailureHistoryItem {
                signature: signature.to_string(),
                category: category.to_string(),
                failure,
                remediation,
            });
        }
        Ok(items)
    }
    .await;
    crate::telemetry::record_db_query(
        "get_recent_failure_history",
        started.elapsed(),
        result.is_ok(),
    );
    result
}

pub async fn warn_if_legacy_signature_backfill_pending(pool: &Pool) -> Result<(), anyhow::Error> {
    let outcome_event_types = crate::execution_feedback::FAILURE_OUTCOME_EVENT_TYPES.to_vec();
    let conn = pool.get().await?;
    let row = conn
        .query_one(
            "SELECT
                count(*) FILTER (WHERE event_type = 'failure_signature')::BIGINT AS legacy_count,
                count(*) FILTER (
                    WHERE event_type = ANY($1)
                      AND metadata->>'success' = 'false'
                      AND (
                          metadata->'payload'->>'signature' IS NULL
                          OR metadata->'payload'->>'signature_category' IS NULL
                          OR metadata->'payload'->>'fingerprint_version' IS NULL
                      )
                )::BIGINT AS pending_count
             FROM agent_events
             WHERE event_type = 'failure_signature'
                OR event_type = ANY($1)",
            &[&outcome_event_types],
        )
        .await?;
    let legacy_count: i64 = row.get("legacy_count");
    let pending_count: i64 = row.get("pending_count");
    if legacy_count > 0 && pending_count > 0 {
        tracing::warn!(
            target: "execution_feedback",
            legacy_signature_events = legacy_count,
            pending_inline_signature_backfill = pending_count,
            command = "orchestrator-maint backfill-signatures",
            "legacy signature events exist; backfill has not completed; retrieval ignores legacy signature rows; operator should run orchestrator-maint backfill-signatures"
        );
    }
    Ok(())
}

async fn remediation_for_failure(
    pool: &Pool,
    failure: &AgentEvent,
    signature: &str,
) -> Result<Option<AgentEvent>, anyhow::Error> {
    // Prefer a remediation in the same correlation chain: that is the strongest
    // signal that a specific failure was resolved by a specific action.
    let chain = get_event_chain_by_event_id(pool, &failure.id).await?;
    if let Some(remediation) = chain.into_iter().find(|event| {
        event.event_type == crate::execution_feedback::EVENT_TYPE_REMEDIATION
            && (event_payload_str(event, "signature") == Some(signature)
                || event_payload_str(event, "failure_event_id") == Some(failure.id.as_str()))
    }) {
        return Ok(Some(remediation));
    }

    // Some clients may submit remediation events later without a parent pointer.
    // Fall back to the latest same-repo remediation with the same signature.
    let conn = pool.get().await?;
    let row = conn
        .query_opt(
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
             FROM agent_events
             WHERE repo = $1
               AND event_type = 'remediation'
               AND metadata->'payload'->>'signature' = $2
             ORDER BY created_at DESC
             LIMIT 1",
            &[&failure.repo, &signature],
        )
        .await?;
    Ok(row.map(row_to_event))
}

fn event_payload_str<'a>(event: &'a AgentEvent, key: &str) -> Option<&'a str> {
    event
        .metadata
        .get("payload")
        .and_then(|payload| payload.get(key))
        .and_then(serde_json::Value::as_str)
}

pub async fn insert_error_record(
    pool: &Pool,
    repo: &str,
    task: &str,
    error_type: &str,
    description: &str,
    severity: &str,
) -> Result<(), anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        conn.execute(
            "INSERT INTO error_index (repo, task, error_type, description, severity)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (repo, task, error_type, description)
             DO UPDATE SET
                frequency = error_index.frequency + 1,
                last_seen = now()",
            &[&repo, &task, &error_type, &description, &severity],
        )
        .await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("insert_error_record", started.elapsed(), result.is_ok());
    result
}

pub async fn get_active_errors(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<ErrorRecord>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result: Result<Vec<ErrorRecord>, anyhow::Error> = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, repo, task, error_type, description, severity, frequency, last_seen
             FROM error_index
             WHERE repo = $1
             ORDER BY frequency DESC, last_seen DESC
             LIMIT $2",
                &[&repo, &limit],
            )
            .await?;
        rows.into_iter()
            .map(|row| {
                Ok(ErrorRecord {
                    id: row.get("id"),
                    repo: row.get("repo"),
                    task: row.get("task"),
                    error_type: row.get("error_type"),
                    description: row.get("description"),
                    severity: row.get("severity"),
                    frequency: row.get("frequency"),
                    last_seen: row.get("last_seen"),
                })
            })
            .collect()
    }
    .await;
    crate::telemetry::record_db_query("get_active_errors", started.elapsed(), result.is_ok());
    result
}
