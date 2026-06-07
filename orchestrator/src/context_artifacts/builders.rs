use crate::context_artifacts::{stable_hash, ContextArtifact};

pub fn active_instruction_artifact(
    repo: String,
    events: &[crate::db::AgentEvent],
) -> Option<ContextArtifact> {
    let instructions = active_instruction_lines(events);
    if instructions.is_empty() {
        return None;
    }

    let rendered = instructions
        .iter()
        .map(|line| format!("- {line}\n"))
        .collect::<String>();
    let compact = instructions.join(" ");
    let raw = serde_json::json!({
        "source": "agent_events",
        "instructions": instructions,
        "events": events.iter().map(crate::db::AgentEvent::payload).collect::<Vec<_>>(),
    });
    let source_event_ids = serde_json::json!(events
        .iter()
        .map(|event| event.id.clone())
        .collect::<Vec<_>>());
    let invalidation_key = stable_hash(&serde_json::json!({
        "artifact_type": "active_instruction",
        "source_event_ids": source_event_ids,
        "instructions": raw["instructions"],
    }));

    Some(
        ContextArtifact::new(
            repo,
            "repo",
            "active_instruction",
            Some(raw.to_string()),
            compact,
            rendered,
            invalidation_key,
            serde_json::json!([]),
        )
        .with_source_event_ids(source_event_ids),
    )
}

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

fn active_instruction_lines(events: &[crate::db::AgentEvent]) -> Vec<String> {
    let mut lines = Vec::new();
    let mut subjects = std::collections::HashSet::new();
    for event in events {
        let text = event
            .evidence
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(&event.summary)
            .trim();
        let subject = instruction_subject(text);
        if !subjects.insert(subject) {
            continue;
        }
        let normalized = normalize_instruction(text);
        if normalized.is_empty() || lines.iter().any(|line| line == &normalized) {
            continue;
        }
        lines.push(normalized);
        if lines.len() >= 5 {
            break;
        }
    }
    lines
}

pub fn instruction_subject(text: &str) -> &'static str {
    let lower = text.to_ascii_lowercase();
    if lower.contains("model") || lower.contains("snowflake") || lower.contains("embedding") {
        "model"
    } else if lower.contains("config") || lower.contains("default") || lower.contains("explicit") {
        "configuration"
    } else if lower.contains("tool")
        || lower.contains("python")
        || lower.contains("perl")
        || lower.contains("git apply")
        || lower.contains("apply_patch")
    {
        "tooling"
    } else if lower.contains("total recall") || lower.contains("memory") {
        "memory"
    } else if lower.contains("commit") || lower.contains("push") {
        "git"
    } else {
        "general"
    }
}

fn normalize_instruction(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(3)
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(320)
        .collect()
}

pub(super) fn truncate_line(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim().replace('\n', " ");
    trimmed.chars().take(max_chars).collect()
}
