//! Per-trajectory edit-target tracking for stateful single-file enforcement.
//!
//! `single_file_edit` means one file per trajectory, but a single authorize
//! call cannot see what the trajectory already edited. This capture-side table
//! records the target path of every allowed file edit keyed by trajectory id;
//! the authorize route consults the first recorded target and denies edits to
//! a different file for the rest of the trajectory.

use deadpool_postgres::Pool;
use uuid::Uuid;

/// Initialize capture-side storage for trajectory edit targets.
pub async fn init(pool: &Pool) -> Result<(), anyhow::Error> {
    let conn = pool.get().await?;
    conn.batch_execute(
        "CREATE TABLE IF NOT EXISTS trajectory_edit_targets (
            id UUID PRIMARY KEY,
            trajectory_id UUID NOT NULL,
            target_path TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS trajectory_edit_targets_trajectory_idx
            ON trajectory_edit_targets(trajectory_id, created_at ASC);",
    )
    .await?;
    Ok(())
}

/// First edit target recorded for the trajectory, if any.
pub async fn first_target(
    pool: &Pool,
    trajectory_id: Uuid,
) -> Result<Option<String>, anyhow::Error> {
    let conn = pool.get().await?;
    let row = conn
        .query_opt(
            "SELECT target_path FROM trajectory_edit_targets
             WHERE trajectory_id = $1 ORDER BY created_at ASC LIMIT 1",
            &[&trajectory_id],
        )
        .await?;
    Ok(row.map(|row| row.get::<_, String>(0)))
}

/// Record an allowed edit target. Best-effort: storage failure must not break
/// the authorization path.
pub fn record_target_best_effort(pool: Option<&Pool>, trajectory_id: Uuid, target_path: String) {
    let Some(pool) = pool else {
        return;
    };
    let pool = pool.clone();
    tokio::spawn(async move {
        let result: Result<(), anyhow::Error> = async {
            let conn = pool.get().await?;
            conn.execute(
                "INSERT INTO trajectory_edit_targets (id, trajectory_id, target_path)
                 VALUES ($1, $2, $3)",
                &[&Uuid::new_v4(), &trajectory_id, &target_path],
            )
            .await?;
            Ok(())
        }
        .await;
        if let Err(e) = result {
            tracing::warn!("trajectory_edit_targets insert failed: {e}");
        }
    });
}
