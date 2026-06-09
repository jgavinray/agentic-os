use crate::db::AgentEvent;

pub(crate) fn format_event_line(event: &AgentEvent) -> String {
    let level = format!("L{}", event.summary_level);
    let mut line = format!("- [{level}:{}] {}", event.event_type, event.summary);
    if let Some(evidence) = &event.evidence {
        if !evidence.is_empty() {
            line.push_str(&format!("\n  Evidence: {evidence}"));
        }
    }
    if let Some(source_ids) = event
        .metadata
        .get("source_event_ids")
        .or_else(|| event.metadata.get("summarized_event_ids"))
        .and_then(|v| v.as_array())
    {
        if !source_ids.is_empty() {
            line.push_str(&format!("\n  Source IDs: {}", source_ids.len()));
        }
    }
    if let Some(outcome) = event.metadata.get("outcome").and_then(|v| v.as_str()) {
        line.push_str(&format!("\n  Outcome: {outcome}"));
    }
    line.push('\n');
    line
}
