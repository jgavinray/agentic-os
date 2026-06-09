use crate::db_event_rows::rows_to_events;
use crate::db_types::AgentEvent;
use deadpool_postgres::Pool;

pub(crate) async fn get_events_for_repo_by_level(
    pool: &Pool,
    repo: &str,
    level: crate::state::MemoryLevel,
    limit: i64,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    if limit <= 0 {
        return Ok(vec![]);
    }

    let conn = pool.get().await?;
    let level = level.as_i32();
    let rows = conn
        .query(
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
             FROM agent_events
             WHERE repo = $1
               AND summary_level = $2
               AND event_type NOT IN ('failed_attempt', 'remediation')
               AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
             ORDER BY created_at DESC
             LIMIT $3",
            &[&repo, &level, &limit],
        )
        .await?;

    Ok(rows_to_events(rows))
}

pub(crate) async fn get_failure_events_for_repo(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    if limit <= 0 {
        return Ok(vec![]);
    }

    let conn = pool.get().await?;
    let rows = conn
        .query(
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
             FROM agent_events
             WHERE repo = $1
               AND event_type IN ('failed_attempt', 'remediation')
               AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
             ORDER BY created_at DESC
             LIMIT $2",
            &[&repo, &limit],
        )
        .await?;

    Ok(rows_to_events(rows))
}

#[allow(dead_code)]
pub(crate) async fn get_events_for_repo_by_levels(
    pool: &Pool,
    repo: &str,
    levels: &[i32],
    limit: i64,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let conn = pool.get().await?;
    let rows = conn
        .query(
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
             FROM agent_events
             WHERE repo = $1
               AND summarized = false
               AND summary_level = ANY($2)
               AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
             ORDER BY summary_level DESC, created_at DESC
             LIMIT $3",
            &[&repo, &levels, &limit],
        )
        .await?;

    Ok(rows_to_events(rows))
}
