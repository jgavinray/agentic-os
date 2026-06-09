use crate::db_event_rows::rows_to_events;
use crate::db_types::AgentEvent;
use deadpool_postgres::Pool;

pub async fn get_recent_instruction_candidates(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let instruction_patterns: &[&str] = &[
            "%explicit%",
            "%instruction%",
            "%must%",
            "%need%",
            "%should%",
            "%do not%",
            "%don't%",
            "%never%",
            "%stop%",
            "%default%",
            "%config%",
        ];
        let rows = conn
            .query(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata,
                        correlation_id, parent_event_id, trajectory_id, attempt_index,
                        event_role, created_at, summary_level
                 FROM agent_events
                 WHERE repo = $1
                   AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
                   AND (
                     event_role = 'user'
                     OR actor = 'user'
                     OR event_type IN ('user_message', 'instruction', 'preference')
                   )
                   AND (
                     summary ILIKE ANY($2)
                     OR coalesce(evidence, '') ILIKE ANY($2)
                   )
                 ORDER BY created_at DESC
                 LIMIT $3",
                &[&repo, &instruction_patterns, &limit],
            )
            .await?;

        Ok(rows_to_events(rows))
    }
    .await;
    crate::telemetry::record_db_query(
        "get_recent_instruction_candidates",
        started.elapsed(),
        result.is_ok(),
    );
    result
}

pub async fn get_recent_repo_decision_candidates(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata,
                        correlation_id, parent_event_id, trajectory_id, attempt_index,
                        event_role, created_at, summary_level
                 FROM agent_events
                 WHERE repo = $1
                   AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
                   AND (
                     event_type IN ('decision', 'checkpoint')
                     OR summary ILIKE ANY($2)
                     OR coalesce(evidence, '') ILIKE ANY($2)
                   )
                 ORDER BY created_at DESC
                 LIMIT $3",
                &[
                    &repo,
                    &[
                        "%decision%",
                        "%decided%",
                        "%should use%",
                        "%must use%",
                        "%architecture%",
                        "%default%",
                    ]
                    .as_slice(),
                    &limit,
                ],
            )
            .await?;
        Ok(rows_to_events(rows))
    }
    .await;
    crate::telemetry::record_db_query(
        "get_recent_repo_decision_candidates",
        started.elapsed(),
        result.is_ok(),
    );
    result
}

pub async fn get_recent_session_events(
    pool: &Pool,
    session_id: &str,
    limit: i64,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata,
                        correlation_id, parent_event_id, trajectory_id, attempt_index,
                        event_role, created_at, summary_level
                 FROM agent_events
                 WHERE session_id = $1
                   AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
                   AND event_type NOT IN ('context_pack')
                 ORDER BY created_at DESC
                 LIMIT $2",
                &[&session_id, &limit],
            )
            .await?;
        Ok(rows_to_events(rows))
    }
    .await;
    crate::telemetry::record_db_query(
        "get_recent_session_events",
        started.elapsed(),
        result.is_ok(),
    );
    result
}
