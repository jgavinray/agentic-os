use crate::db_types::AgentEvent;

pub(crate) fn rows_to_events(rows: Vec<tokio_postgres::Row>) -> Vec<AgentEvent> {
    rows.into_iter().map(row_to_event).collect()
}

pub(crate) fn row_to_event(row: tokio_postgres::Row) -> AgentEvent {
    AgentEvent {
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
    }
}
