use deadpool_postgres::Pool;
use uuid::Uuid;

use crate::db_event_rows::{row_to_event, rows_to_events};
use crate::db_types::AgentEvent;

pub async fn get_trajectory(
    pool: &Pool,
    trajectory_id: Uuid,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
                 FROM agent_events
                 WHERE trajectory_id = $1
                 ORDER BY attempt_index ASC NULLS LAST, created_at ASC, id ASC",
                &[&trajectory_id],
            )
            .await?;
        Ok(rows_to_events(rows))
    }
    .await;
    crate::telemetry::record_db_query("get_trajectory", started.elapsed(), result.is_ok());
    result
}

pub async fn get_trajectory_attempts(
    pool: &Pool,
    trajectory_id: Uuid,
) -> Result<Vec<i32>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT DISTINCT attempt_index
                 FROM agent_events
                 WHERE trajectory_id = $1
                   AND attempt_index IS NOT NULL
                 ORDER BY attempt_index ASC",
                &[&trajectory_id],
            )
            .await?;
        Ok(rows
            .into_iter()
            .map(|row| row.get("attempt_index"))
            .collect())
    }
    .await;
    crate::telemetry::record_db_query("get_trajectory_attempts", started.elapsed(), result.is_ok());
    result
}

pub async fn get_trajectory_result(
    pool: &Pool,
    trajectory_id: Uuid,
) -> Result<Option<AgentEvent>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let row = conn
            .query_opt(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
                 FROM agent_events
                 WHERE trajectory_id = $1
                   AND event_role = 'trajectory_result'
                 ORDER BY created_at ASC
                 LIMIT 1",
                &[&trajectory_id],
            )
            .await?;
        Ok(row.map(row_to_event))
    }
    .await;
    crate::telemetry::record_db_query("get_trajectory_result", started.elapsed(), result.is_ok());
    result
}

pub async fn latest_trajectory_event_for_session(
    pool: &Pool,
    session_id: &str,
) -> Result<Option<AgentEvent>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let row = conn
            .query_opt(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
                 FROM agent_events
                 WHERE session_id = $1
                   AND trajectory_id IS NOT NULL
                 ORDER BY created_at DESC, id DESC
                 LIMIT 1",
                &[&session_id],
            )
            .await?;
        Ok(row.map(row_to_event))
    }
    .await;
    crate::telemetry::record_db_query(
        "latest_trajectory_event_for_session",
        started.elapsed(),
        result.is_ok(),
    );
    result
}

pub async fn idle_trajectory_ids(
    pool: &Pool,
    idle_timeout_sec: u64,
    limit: i64,
) -> Result<Vec<Uuid>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "WITH latest AS (
                    SELECT trajectory_id, max(created_at) AS last_event_at
                    FROM agent_events
                    WHERE trajectory_id IS NOT NULL
                    GROUP BY trajectory_id
                )
                SELECT latest.trajectory_id
                FROM latest
                WHERE latest.last_event_at < now() - ($1::text || ' seconds')::interval
                  AND NOT EXISTS (
                      SELECT 1
                      FROM agent_events result
                      WHERE result.trajectory_id = latest.trajectory_id
                        AND result.event_role = 'trajectory_result'
                  )
                ORDER BY latest.last_event_at ASC
                LIMIT $2",
                &[&idle_timeout_sec.to_string(), &limit],
            )
            .await?;
        Ok(rows
            .into_iter()
            .map(|row| row.get("trajectory_id"))
            .collect())
    }
    .await;
    crate::telemetry::record_db_query("idle_trajectory_ids", started.elapsed(), result.is_ok());
    result
}
