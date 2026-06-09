use deadpool_postgres::Pool;
use uuid::Uuid;

use crate::db_types::AgentEvent;

pub async fn emit_trajectory_result_once(
    pool: &Pool,
    embedder: &crate::embedder::Embedder,
    qdrant_url: &str,
    trajectory_id: Uuid,
    boundary_reason: Option<crate::trajectory::BoundaryReason>,
) -> Result<bool, anyhow::Error> {
    let events = crate::db_trajectory::get_trajectory(pool, trajectory_id).await?;
    let source_events: Vec<_> = events
        .into_iter()
        .filter(|event| {
            event.event_role.as_deref()
                != Some(crate::trajectory::EventRole::TrajectoryResult.as_str())
        })
        .collect();
    let Some(first_event) = source_events.first() else {
        return Ok(false);
    };
    let summary =
        crate::trajectory::summarize_trajectory(trajectory_id, &source_events, boundary_reason);
    let result_event = crate::trajectory::trajectory_result_event(
        &first_event.session_id,
        &first_event.repo,
        summary.clone(),
    );
    let inserted = insert_trajectory_result_event(pool, &result_event).await?;
    if inserted {
        crate::telemetry::record_trajectory_result(&summary);
        let _ = crate::qdrant::store_event(embedder, qdrant_url, &result_event)
            .await
            .map_err(|e| {
                tracing::warn!(
                    trajectory_id = %trajectory_id,
                    "trajectory result stored in postgres but qdrant indexing failed: {e}"
                );
                e
            });
    }
    Ok(inserted)
}

async fn insert_trajectory_result_event(
    pool: &Pool,
    event: &AgentEvent,
) -> Result<bool, anyhow::Error> {
    crate::trajectory::validate_event_role(event.event_role.as_deref())?;
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .execute(
                "INSERT INTO agent_events
                 (id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, summary_level)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                 ON CONFLICT DO NOTHING",
                &[
                    &event.id,
                    &event.session_id,
                    &event.repo,
                    &event.actor,
                    &event.event_type,
                    &event.summary,
                    &event.evidence,
                    &event.metadata,
                    &event.correlation_id,
                    &event.parent_event_id,
                    &event.trajectory_id,
                    &event.attempt_index,
                    &event.event_role,
                    &event.summary_level,
                ],
            )
            .await?;
        Ok(rows == 1)
    }
    .await;
    crate::telemetry::record_db_query(
        "insert_trajectory_result_event",
        started.elapsed(),
        result.is_ok(),
    );
    result
}
