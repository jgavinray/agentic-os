use crate::db_event_rows::row_to_event;
use crate::db_trajectory::get_event_chain_by_event_id;
use crate::db_types::AgentEvent;
use deadpool_postgres::Pool;

pub(crate) async fn remediation_for_failure(
    pool: &Pool,
    failure: &AgentEvent,
    signature: &str,
) -> Result<Option<AgentEvent>, anyhow::Error> {
    // Prefer a remediation in the same correlation chain: that is the strongest
    // signal that a specific failure was resolved by a specific action.
    let chain = get_event_chain_by_event_id(pool, &failure.id).await?;
    if let Some(remediation) = chain.into_iter().find(|event| {
        event.event_type == crate::execution_feedback::EVENT_TYPE_REMEDIATION
            && (event_payload_str(event, "signature") == Some(signature)
                || event_payload_str(event, "failure_event_id") == Some(failure.id.as_str()))
    }) {
        return Ok(Some(remediation));
    }

    // Some clients may submit remediation events later without a parent pointer.
    // Fall back to the latest same-repo remediation with the same signature.
    let conn = pool.get().await?;
    let row = conn
        .query_opt(
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
             FROM agent_events
             WHERE repo = $1
               AND event_type = 'remediation'
               AND metadata->'payload'->>'signature' = $2
             ORDER BY created_at DESC
             LIMIT 1",
            &[&failure.repo, &signature],
        )
        .await?;
    Ok(row.map(row_to_event))
}

pub(crate) fn event_payload_str<'a>(event: &'a AgentEvent, key: &str) -> Option<&'a str> {
    event
        .metadata
        .get("payload")
        .and_then(|payload| payload.get(key))
        .and_then(serde_json::Value::as_str)
}
