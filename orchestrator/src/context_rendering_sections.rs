use crate::context_rendering_event_lines::format_event_line;
use crate::db::AgentEvent;

pub(crate) use crate::context_rendering_failure_sections::{
    append_failure_history_section, append_failure_section,
};

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
