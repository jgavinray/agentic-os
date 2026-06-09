use crate::db_context_event_queries::{
    get_events_for_repo_by_level, get_events_for_repo_by_levels, get_failure_events_for_repo,
};
use crate::db_event_rows::rows_to_events;
use crate::db_types::{AgentEvent, ContextEvidence};
use deadpool_postgres::Pool;

pub use crate::db_context_search::{hydrate_active_search_hits, search_events_fts};

#[allow(dead_code)]
pub async fn get_events_for_repo(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let conn = pool.get().await?;
    let rows = conn
        .query(
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
             FROM agent_events
             WHERE repo = $1
               AND summarized = false
               AND summary_level = 0
               AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
             ORDER BY created_at DESC
             LIMIT $2",
            &[&repo, &limit],
        )
        .await?;

    Ok(rows_to_events(rows))
}

pub async fn count_events_for_repo(pool: &Pool, repo: &str) -> Result<i64, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let row = conn
            .query_one(
                "SELECT count(*)::BIGINT AS count
                 FROM agent_events
                 WHERE repo = $1
                   AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'",
                &[&repo],
            )
            .await?;
        Ok(row.get("count"))
    }
    .await;
    crate::telemetry::record_db_query("count_events_for_repo", started.elapsed(), result.is_ok());
    result
}

pub fn preferred_summary_levels(event_count: i64) -> Vec<i32> {
    match event_count {
        n if n < 20 => vec![0],
        n if n < 200 => vec![1],
        n if n < 2000 => vec![2],
        _ => vec![3, 2],
    }
}

#[allow(dead_code)]
pub async fn get_context_events_for_repo(
    pool: &Pool,
    repo: &str,
    event_count: i64,
    limit: i64,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let levels = preferred_summary_levels(event_count);
    let mut events = get_events_for_repo_by_levels(pool, repo, &levels, limit).await?;

    // Fresh or partially migrated repositories may not yet have promoted summaries.
    // Fall back through lower levels so context remains available while the
    // background summarizer catches up.
    if events.is_empty() && !levels.contains(&0) {
        events = get_events_for_repo_by_levels(pool, repo, &[1, 0], limit).await?;
    }
    if events.is_empty() {
        events = get_events_for_repo(pool, repo, limit).await?;
    }

    Ok(events)
}

pub async fn get_context_evidence_for_policy(
    pool: &Pool,
    repo: &str,
    policy: &crate::state::ContextPolicy,
) -> Result<ContextEvidence, anyhow::Error> {
    let started = std::time::Instant::now();
    let (l0_recent, l1_matching, l2_repo, l3_project, failures) = tokio::join!(
        get_events_for_repo_by_level(
            pool,
            repo,
            crate::state::MemoryLevel::L0,
            policy.l0_recent_limit
        ),
        get_events_for_repo_by_level(pool, repo, crate::state::MemoryLevel::L1, policy.l1_limit),
        get_events_for_repo_by_level(pool, repo, crate::state::MemoryLevel::L2, policy.l2_limit),
        get_events_for_repo_by_level(pool, repo, crate::state::MemoryLevel::L3, policy.l3_limit),
        get_failure_events_for_repo(pool, repo, policy.failure_limit),
    );

    let result: Result<ContextEvidence, anyhow::Error> = Ok(ContextEvidence {
        l0_recent: l0_recent?,
        l1_matching: l1_matching?,
        l2_repo: l2_repo?,
        l3_project: l3_project?,
        failures: failures?,
        failure_history: vec![],
        operational_constraints: vec![],
    });
    crate::telemetry::record_db_query("get_context_evidence", started.elapsed(), result.is_ok());
    result
}
