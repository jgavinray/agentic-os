use crate::context_artifacts::builders::truncate_line;
use crate::context_artifacts::{stable_hash, ContextArtifact};

pub fn failure_history_artifact(
    repo: String,
    items: &[crate::db::FailureHistoryItem],
) -> Option<ContextArtifact> {
    let items = items
        .iter()
        .filter(|item| item.remediation.is_some())
        .take(5)
        .collect::<Vec<_>>();
    if items.is_empty() {
        return None;
    }

    let mut rendered = String::new();
    let mut compact_lines = Vec::new();
    let mut source_event_ids = Vec::new();
    for item in &items {
        let remediation = item.remediation.as_ref()?;
        rendered.push_str(&format!(
            "- resolved failure [{}]: {}\n  remediation: {}\n",
            item.category,
            truncate_line(&item.failure.summary, 220),
            truncate_line(&remediation.summary, 220),
        ));
        compact_lines.push(format!(
            "Resolved {} failure with remediation: {}",
            item.category,
            truncate_line(&remediation.summary, 160)
        ));
        source_event_ids.push(item.failure.id.clone());
        source_event_ids.push(remediation.id.clone());
    }

    let raw = serde_json::json!({
        "source": "agent_events",
        "items": items
            .iter()
            .map(|item| serde_json::json!({
                "signature": item.signature,
                "category": item.category,
                "failure": item.failure.payload(),
                "remediation": item.remediation.as_ref().map(crate::db::AgentEvent::payload),
            }))
            .collect::<Vec<_>>(),
    });
    source_event_ids.sort();
    source_event_ids.dedup();
    let source_event_ids_json = serde_json::json!(source_event_ids);
    let invalidation_key = stable_hash(&serde_json::json!({
        "artifact_type": "failure_history",
        "source_event_ids": source_event_ids_json,
        "items": raw["items"],
    }));

    Some(
        ContextArtifact::new(
            repo,
            "repo",
            "failure_history",
            Some(raw.to_string()),
            compact_lines.join(" "),
            rendered,
            invalidation_key,
            serde_json::json!([]),
        )
        .with_source_event_ids(source_event_ids_json),
    )
}
