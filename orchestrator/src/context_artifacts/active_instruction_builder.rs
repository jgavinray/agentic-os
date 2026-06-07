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
