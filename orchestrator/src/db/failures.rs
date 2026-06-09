use crate::db_event_rows::rows_to_events;
use crate::db_failure_remediation::{event_payload_str, remediation_for_failure};
use crate::db_types::FailureHistoryItem;
use deadpool_postgres::Pool;

pub use crate::db_error_index::{get_active_errors, insert_error_record};

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
