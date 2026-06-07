use crate::context_rendering_event_lines::format_event_line;
use crate::db::{AgentEvent, FailureHistoryItem};

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
