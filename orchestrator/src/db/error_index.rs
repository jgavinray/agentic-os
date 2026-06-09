use crate::state::ErrorRecord;
use deadpool_postgres::Pool;

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
