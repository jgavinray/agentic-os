use crate::db::AgentEvent;
use crate::feature_constraints::enforce_constraint_token_budget;
use crate::feature_extraction_types::{
    ExtractionScope, OperationalConstraint, SuppressedConstraint,
};
use deadpool_postgres::Pool;
use serde_json::Value;
use uuid::Uuid;

pub async fn load_events_for_scope(
    pool: &Pool,
    scope: &ExtractionScope,
    limit: Option<i64>,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata,
                        correlation_id, parent_event_id, trajectory_id, attempt_index, event_role,
                        created_at, summary_level
                 FROM agent_events
                 WHERE ($1::text IS NULL OR repo = $1)
                   AND ($2::text IS NULL OR session_id = $2)
                   AND ($3::uuid IS NULL OR trajectory_id = $3)
                   AND ($4::timestamptz IS NULL OR created_at >= $4)
                   AND ($5::timestamptz IS NULL OR created_at < $5)
                 ORDER BY created_at ASC, id ASC
                 LIMIT COALESCE($6, 9223372036854775807)",
                &[
                    &scope.repo,
                    &scope.session_id,
                    &scope.trajectory_id,
                    &scope.since,
                    &scope.until,
                    &limit,
                ],
            )
            .await?;
        Ok(rows
            .into_iter()
            .map(|row| AgentEvent {
                id: row.get("id"),
                session_id: row.get("session_id"),
                repo: row.get("repo"),
                actor: row.get("actor"),
                event_type: row.get("event_type"),
                summary: row.get("summary"),
                evidence: row.get("evidence"),
                metadata: row.get("metadata"),
                correlation_id: row.get("correlation_id"),
                parent_event_id: row.get("parent_event_id"),
                trajectory_id: row.get("trajectory_id"),
                attempt_index: row.get("attempt_index"),
                event_role: row.get("event_role"),
                created_at: row.get("created_at"),
                summary_level: row.get("summary_level"),
            })
            .collect())
    }
    .await;
    crate::telemetry::record_db_query("feature_load_events", started.elapsed(), result.is_ok());
    result
}

pub async fn operational_constraints_for_context(
    pool: &Pool,
    repo: &str,
    session_id: Option<&str>,
    trajectory_id: Option<Uuid>,
    token_budget: usize,
) -> Result<(Vec<OperationalConstraint>, Vec<SuppressedConstraint>), anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let row = if let Some(trajectory_id) = trajectory_id {
            conn.query_opt(
                "SELECT recommended_constraints
                 FROM agent_feature_records
                 WHERE repo = $1 AND trajectory_id = $2
                 ORDER BY updated_at DESC
                 LIMIT 1",
                &[&repo, &trajectory_id],
            )
            .await?
        } else if let Some(session_id) = session_id {
            conn.query_opt(
                "SELECT recommended_constraints
                 FROM agent_feature_records
                 WHERE repo = $1 AND session_id = $2
                 ORDER BY window_end DESC, updated_at DESC
                 LIMIT 1",
                &[&repo, &session_id],
            )
            .await?
        } else {
            conn.query_opt(
                "SELECT recommended_constraints
                 FROM agent_feature_records
                 WHERE repo = $1
                 ORDER BY updated_at DESC
                 LIMIT 1",
                &[&repo],
            )
            .await?
        };

        let constraints = row
            .map(|row| row.get::<_, Value>("recommended_constraints"))
            .and_then(|value| serde_json::from_value::<Vec<OperationalConstraint>>(value).ok())
            .unwrap_or_default();
        Ok(enforce_constraint_token_budget(&constraints, token_budget))
    }
    .await;
    crate::telemetry::record_db_query(
        "feature_context_constraints",
        started.elapsed(),
        result.is_ok(),
    );
    result
}
