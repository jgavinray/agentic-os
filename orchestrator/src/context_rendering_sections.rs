use crate::db::{AgentEvent, FailureHistoryItem};

pub(crate) fn append_operational_constraints_section(
    out: &mut String,
    constraints: &[crate::feature_extraction::OperationalConstraint],
) {
    if constraints.is_empty() {
        return;
    }

    out.push_str("Operational Constraints:\n");
    for constraint in constraints {
        out.push_str("- ");
        out.push_str(&constraint.text);
        out.push('\n');
    }
    out.push('\n');
}

pub(crate) fn append_failure_history_section(
    out: &mut String,
    items: &[FailureHistoryItem],
    budget: usize,
    seen: &mut std::collections::HashSet<String>,
) -> Vec<String> {
    if items.is_empty() || budget == 0 {
        return vec![];
    }

    let mut body = String::new();
    let mut used = 0usize;
    let mut reused_signatures = Vec::new();
    for item in items {
        let key = format!("failure-history:{}", item.failure.id);
        if !seen.insert(key) {
            continue;
        }
        let mut line = format!(
            "- [failure:{}] {}\n  Category: {}\n",
            item.signature, item.failure.summary, item.category
        );
        if let Some(excerpt) = failure_history_excerpt(&item.failure) {
            line.push_str(&format!(
                "  Excerpt: {}\n",
                excerpt.chars().take(300).collect::<String>()
            ));
        }
        if let Some(remediation) = &item.remediation {
            line.push_str(&format!("  Remediation: {}\n", remediation.summary));
            reused_signatures.push(item.signature.clone());
        }
        if used + line.len() > budget {
            body.push_str("- [truncated: failure history budget exceeded]\n");
            break;
        }
        used += line.len();
        body.push_str(&line);
    }

    if !body.is_empty() {
        out.push_str("== Failure History ==\n");
        out.push_str(&body);
        out.push('\n');
    }
    reused_signatures
}

fn failure_history_excerpt(event: &AgentEvent) -> Option<String> {
    let payload = event.metadata.get("payload")?;
    for key in [
        "raw_excerpt",
        "failure_reason",
        "stderr_summary",
        "stdout_summary",
    ] {
        if let Some(text) = payload
            .get(key)
            .and_then(serde_json::Value::as_str)
            .filter(|text| !text.trim().is_empty())
        {
            return Some(text.to_string());
        }
    }
    for key in ["failure_summaries", "findings"] {
        if let Some(values) = payload.get(key).and_then(serde_json::Value::as_array) {
            let joined = values
                .iter()
                .filter_map(serde_json::Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            if !joined.is_empty() {
                return Some(joined);
            }
        }
    }
    None
}

pub(crate) fn append_event_section(
    out: &mut String,
    title: &str,
    events: &[AgentEvent],
    budget: usize,
    seen: &mut std::collections::HashSet<String>,
) {
    if events.is_empty() || budget == 0 {
        return;
    }

    let mut body = String::new();
    let mut used = 0usize;
    for event in events {
        if !seen.insert(event.summary.clone()) {
            continue;
        }
        let line = format_event_line(event);
        if used + line.len() > budget {
            body.push_str("- [truncated: section budget exceeded]\n");
            break;
        }
        used += line.len();
        body.push_str(&line);
    }

    if !body.is_empty() {
        out.push_str(title);
        out.push('\n');
        out.push_str(&body);
        out.push('\n');
    }
}

pub(crate) fn append_search_section(
    out: &mut String,
    title: &str,
    hits: &[crate::state::SearchHit],
    budget: usize,
    seen: &mut std::collections::HashSet<String>,
) {
    if hits.is_empty() || budget == 0 {
        return;
    }

    let mut body = String::new();
    let mut used = 0usize;
    for hit in hits {
        if !seen.insert(hit.summary.clone()) {
            continue;
        }
        let line = format!("- [retrieved:{}] {}\n", hit.event_type, hit.summary);
        if used + line.len() > budget {
            body.push_str("- [truncated: retrieval budget exceeded]\n");
            break;
        }
        used += line.len();
        body.push_str(&line);
    }

    if !body.is_empty() {
        out.push_str(title);
        out.push('\n');
        out.push_str(&body);
        out.push('\n');
    }
}

pub(crate) fn append_failure_section(
    out: &mut String,
    failures: &[AgentEvent],
    errors: &[crate::state::ErrorRecord],
    budget: usize,
    seen: &mut std::collections::HashSet<String>,
) {
    if failures.is_empty() && errors.is_empty() {
        return;
    }

    let mut body = String::new();
    let mut used = 0usize;
    for event in failures {
        if !seen.insert(event.summary.clone()) {
            continue;
        }
        let line = format_event_line(event);
        if used + line.len() > budget {
            body.push_str("- [truncated: failure budget exceeded]\n");
            break;
        }
        used += line.len();
        body.push_str(&line);
    }

    for err in errors {
        let text = format!("{}:{}", err.error_type, err.description);
        if !seen.insert(text) {
            continue;
        }
        let line = format!(
            "- [error_index:{}] {} (seen {} times)\n",
            err.error_type, err.description, err.frequency
        );
        if used + line.len() > budget {
            break;
        }
        used += line.len();
        body.push_str(&line);
    }

    if !body.is_empty() {
        out.push_str("== Failed Attempts and Remediations ==\n");
        out.push_str(&body);
        out.push('\n');
    }
}

pub(crate) fn append_open_questions(
    out: &mut String,
    recent: &[AgentEvent],
    seen: &mut std::collections::HashSet<String>,
) {
    const BUDGET: usize = 500;
    let mut body = String::new();
    let mut used = 0usize;
    'outer: for event in recent {
        if event.event_type != "checkpoint" {
            continue;
        }
        if let Some(arr) = event
            .metadata
            .get("open_questions")
            .and_then(|v| v.as_array())
        {
            for q in arr {
                if let Some(text) = q.as_str() {
                    let line = format!("- {text}\n");
                    if used + line.len() > BUDGET {
                        break 'outer;
                    }
                    if seen.insert(text.to_string()) {
                        body.push_str(&line);
                        used += line.len();
                    }
                }
            }
        }
    }

    if !body.is_empty() {
        out.push_str("== Open Questions ==\n");
        out.push_str(&body);
    }
}

fn format_event_line(event: &AgentEvent) -> String {
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
