use crate::context_artifacts::{stable_hash, ContextArtifact};

pub fn durable_project_memory_artifact(
    repo: String,
    notes: &[crate::total_recall::MemoryNote],
) -> Option<ContextArtifact> {
    let notes = notes
        .iter()
        .filter(|note| !note.archived && !note.content.trim().is_empty())
        .take(5)
        .collect::<Vec<_>>();
    if notes.is_empty() {
        return None;
    }

    let rendered = notes
        .iter()
        .map(|note| {
            let title = note.title.as_deref().unwrap_or(&note.date);
            format!(
                "- {}: {}\n",
                truncate_line(title, 120),
                truncate_line(&note.content, 260)
            )
        })
        .collect::<String>();
    let compact = notes
        .iter()
        .map(|note| truncate_line(&note.content, 160))
        .collect::<Vec<_>>()
        .join(" ");
    let raw = serde_json::json!({
        "source": "total_recall",
        "notes": notes
            .iter()
            .map(|note| serde_json::json!({
                "id": note.id,
                "date": note.date,
                "title": note.title,
                "content": note.content,
                "updated_at": note.updated_at,
            }))
            .collect::<Vec<_>>(),
    });
    let invalidation_key = stable_hash(&serde_json::json!({
        "artifact_type": "durable_project_memory",
        "notes": raw["notes"],
    }));

    Some(ContextArtifact::new(
        repo,
        "repo",
        "durable_project_memory",
        Some(raw.to_string()),
        compact,
        rendered,
        invalidation_key,
        serde_json::json!([]),
    ))
}

pub fn repo_decisions_artifact(
    repo: String,
    events: &[crate::db::AgentEvent],
) -> Option<ContextArtifact> {
    let events = events.iter().take(6).collect::<Vec<_>>();
    if events.is_empty() {
        return None;
    }
    let rendered = events
        .iter()
        .map(|event| {
            let evidence = event
                .evidence
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .map(|value| format!(" Evidence: {}", truncate_line(value, 160)))
                .unwrap_or_default();
            format!("- {}{}\n", truncate_line(&event.summary, 220), evidence)
        })
        .collect::<String>();
    let raw = serde_json::json!({
        "source": "agent_events",
        "events": events.iter().map(|event| event.payload()).collect::<Vec<_>>(),
    });
    let source_event_ids = serde_json::json!(events
        .iter()
        .map(|event| event.id.clone())
        .collect::<Vec<_>>());
    let invalidation_key = stable_hash(&serde_json::json!({
        "artifact_type": "repo_decisions",
        "source_event_ids": source_event_ids,
        "events": raw["events"],
    }));
    Some(
        ContextArtifact::new(
            repo,
            "repo",
            "repo_decisions",
            Some(raw.to_string()),
            events
                .iter()
                .map(|event| truncate_line(&event.summary, 120))
                .collect::<Vec<_>>()
                .join(" "),
            rendered,
            invalidation_key,
            serde_json::json!([]),
        )
        .with_source_event_ids(source_event_ids),
    )
}

pub fn session_state_artifact(
    repo: String,
    session_id: &str,
    events: &[crate::db::AgentEvent],
) -> Option<ContextArtifact> {
    let events = events.iter().take(5).collect::<Vec<_>>();
    if events.is_empty() {
        return None;
    }
    let rendered = events
        .iter()
        .map(|event| {
            format!(
                "- [{}] {}\n",
                event.event_type,
                truncate_line(&event.summary, 220)
            )
        })
        .collect::<String>();
    let raw = serde_json::json!({
        "source": "agent_events",
        "session_id": session_id,
        "events": events.iter().map(|event| event.payload()).collect::<Vec<_>>(),
    });
    let source_event_ids = serde_json::json!(events
        .iter()
        .map(|event| event.id.clone())
        .collect::<Vec<_>>());
    let invalidation_key = stable_hash(&serde_json::json!({
        "artifact_type": "session_state",
        "session_id": session_id,
        "source_event_ids": source_event_ids,
    }));
    Some(
        ContextArtifact::new(
            repo,
            "session",
            "session_state",
            Some(raw.to_string()),
            events
                .iter()
                .map(|event| truncate_line(&event.summary, 120))
                .collect::<Vec<_>>()
                .join(" "),
            rendered,
            invalidation_key,
            serde_json::json!([]),
        )
        .with_source_event_ids(source_event_ids),
    )
}

pub(super) fn truncate_line(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim().replace('\n', " ");
    trimmed.chars().take(max_chars).collect()
}
