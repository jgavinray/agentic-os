use crate::db_types::ContextCompilerLedgerEntry;
use deadpool_postgres::Pool;
use uuid::Uuid;

#[allow(clippy::too_many_arguments)]
pub async fn insert_context_compiler_ledger_entry(
    pool: &Pool,
    repo: &str,
    artifact_type: &str,
    candidate_source: &str,
    candidate_id: Option<&str>,
    decision: &str,
    reason: &str,
    artifact_id: Option<Uuid>,
    metadata: serde_json::Value,
) -> Result<(), anyhow::Error> {
    let started = std::time::Instant::now();
    let id = Uuid::new_v4();
    let result = async {
        let conn = pool.get().await?;
        conn.execute(
            "INSERT INTO context_compiler_ledger
             (id, repo, artifact_type, candidate_source, candidate_id, decision,
              reason, artifact_id, metadata)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            &[
                &id,
                &repo,
                &artifact_type,
                &candidate_source,
                &candidate_id,
                &decision,
                &reason,
                &artifact_id,
                &metadata,
            ],
        )
        .await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query(
        "insert_context_compiler_ledger_entry",
        started.elapsed(),
        result.is_ok(),
    );
    result
}

pub async fn get_context_compiler_ledger(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<ContextCompilerLedgerEntry>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, repo, artifact_type, candidate_source, candidate_id, decision,
                        reason, artifact_id, metadata, created_at
                 FROM context_compiler_ledger
                 WHERE repo = $1
                 ORDER BY created_at DESC
                 LIMIT $2",
                &[&repo, &limit],
            )
            .await?;
        Ok(rows
            .into_iter()
            .map(|row| ContextCompilerLedgerEntry {
                id: row.get("id"),
                repo: row.get("repo"),
                artifact_type: row.get("artifact_type"),
                candidate_source: row.get("candidate_source"),
                candidate_id: row.get("candidate_id"),
                decision: row.get("decision"),
                reason: row.get("reason"),
                artifact_id: row.get("artifact_id"),
                metadata: row.get("metadata"),
                created_at: row.get("created_at"),
            })
            .collect())
    }
    .await;
    crate::telemetry::record_db_query(
        "get_context_compiler_ledger",
        started.elapsed(),
        result.is_ok(),
    );
    result
}
