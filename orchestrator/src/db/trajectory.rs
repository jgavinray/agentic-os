use deadpool_postgres::Pool;
use uuid::Uuid;

use crate::db_event_rows::{row_to_event, rows_to_events};
use crate::db_types::AgentEvent;

pub use crate::db_trajectory_queries::{
    get_trajectory, get_trajectory_attempts, get_trajectory_result, idle_trajectory_ids,
    latest_trajectory_event_for_session,
};

pub async fn get_event_chain_by_event_id(
    pool: &Pool,
    event_id: &str,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let seed = get_event_by_id(pool, event_id).await?;
        let Some(correlation_id) = seed.correlation_id else {
            // Historical rows and non-chain events remain valid. With no chain
            // ID there is nothing to reconstruct, so return the seed by itself.
            return Ok(vec![seed]);
        };
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
                 FROM agent_events
                 WHERE correlation_id = $1
                 ORDER BY created_at ASC, id ASC",
                &[&correlation_id],
            )
            .await?;
        Ok(order_event_chain(rows_to_events(rows), event_id))
    }
    .await;
    crate::telemetry::record_db_query("get_event_chain", started.elapsed(), result.is_ok());
    result
}

async fn get_event_by_id(pool: &Pool, event_id: &str) -> Result<AgentEvent, anyhow::Error> {
    let conn = pool.get().await?;
    let row = conn
        .query_one(
            "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, created_at, summary_level
             FROM agent_events
             WHERE id = $1",
            &[&event_id],
        )
        .await?;
    Ok(row_to_event(row))
}

pub fn order_event_chain(events: Vec<AgentEvent>, seed_id: &str) -> Vec<AgentEvent> {
    if events.is_empty() {
        return vec![];
    }
    let by_id: std::collections::HashMap<String, AgentEvent> = events
        .iter()
        .map(|event| (event.id.clone(), event.clone()))
        .collect();
    let Some(seed) = by_id.get(seed_id).cloned() else {
        return events;
    };

    // Walk parent pointers upward to find the root visible in this correlation
    // group. The guard prevents malformed cycles from trapping reconstruction.
    let mut root = seed.clone();
    let mut guard = std::collections::HashSet::new();
    while let Some(parent_id) = root.parent_event_id {
        if !guard.insert(root.id.clone()) {
            break;
        }
        let parent_key = parent_id.to_string();
        let Some(parent) = by_id.get(&parent_key) else {
            break;
        };
        root = parent.clone();
    }

    let grouped = crate::execution_feedback::group_by_parent(&events);
    let mut ordered = Vec::new();
    // Then emit a stable depth-first tree so callers see request -> response ->
    // tools -> validation -> patch/remediation in human-readable order.
    append_chain_tree(&root, &grouped, &mut ordered);
    ordered
}

fn append_chain_tree(
    event: &AgentEvent,
    grouped: &std::collections::BTreeMap<Option<Uuid>, Vec<AgentEvent>>,
    ordered: &mut Vec<AgentEvent>,
) {
    ordered.push(event.clone());
    let Ok(event_uuid) = Uuid::parse_str(&event.id) else {
        return;
    };
    if let Some(children) = grouped.get(&Some(event_uuid)) {
        for child in children {
            append_chain_tree(child, grouped, ordered);
        }
    }
}
