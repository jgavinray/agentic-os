use crate::state::{ErrorRecord, EventMemory, SearchHit};
use std::collections::HashSet;

/// BUG-4: Context is reference material only - no directive language that overrides the harness.
/// BUG-11: Merges recency and semantic results, deduplicates by summary, enforces a character budget.
pub fn build_context(
    repo: &str,
    task: &str,
    recent: &[EventMemory],
    hybrid_hits: &[SearchHit],
    errors: &[ErrorRecord],
    char_budget: usize,
) -> String {
    let header = format!("Repository: {repo}\nTask: {task}\n\n");

    // Tiered budget allocations: 60% hybrid, 30% recent, 10% errors.
    let hybrid_budget = (char_budget * 60) / 100;
    let recent_budget = (char_budget * 30) / 100;
    let error_budget = char_budget.saturating_sub(hybrid_budget + recent_budget);

    let mut seen: HashSet<String> = HashSet::new();

    let hybrid_section = render_hybrid_memory(hybrid_hits, hybrid_budget, &mut seen);
    let (decision_body, open_body) = render_recent_memory(recent, recent_budget, &mut seen);
    let error_body = render_error_memory(errors, error_budget);

    let mut out = format!("{header}{hybrid_section}");
    if !decision_body.is_empty() {
        out.push_str(&format!("\n== Recent Decisions ==\n{decision_body}"));
    }
    if !open_body.is_empty() {
        out.push_str(&format!("\n== Open Questions ==\n{open_body}"));
    }
    if !error_body.is_empty() {
        out.push_str(&format!(
            "\n== Failed Approaches (do not retry) ==\n{error_body}"
        ));
    }
    out
}

fn render_hybrid_memory(
    hybrid_hits: &[SearchHit],
    hybrid_budget: usize,
    seen: &mut HashSet<String>,
) -> String {
    let mut body = String::new();
    let mut used = 0usize;
    for hit in hybrid_hits {
        if !seen.insert(hit.summary.clone()) {
            continue;
        }
        let line = format!("- [{}] {}\n", hit.event_type, hit.summary);
        if used + line.len() > hybrid_budget {
            body.push_str("- [truncated: memory budget exceeded]\n");
            break;
        }
        body.push_str(&line);
        used += line.len();
    }
    if body.is_empty() {
        "== Relevant Prior Memory ==\n- No relevant prior memory found.\n".to_string()
    } else {
        format!("== Relevant Prior Memory ==\n{body}")
    }
}

fn render_recent_memory(
    recent: &[EventMemory],
    recent_budget: usize,
    seen: &mut HashSet<String>,
) -> (String, String) {
    let mut used = 0usize;
    let mut decision_body = String::new();
    for m in recent {
        if m.event_type != "decision" {
            continue;
        }
        if !seen.insert(m.summary.clone()) {
            continue;
        }
        let line = match &m.evidence {
            Some(ev) => format!("- [decision] {}\n  Evidence: {}\n", m.summary, ev),
            None => format!("- [decision] {}\n", m.summary),
        };
        if used + line.len() > recent_budget {
            break;
        }
        decision_body.push_str(&line);
        used += line.len();
    }

    let mut open_body = String::new();
    for m in recent {
        if m.event_type != "checkpoint" {
            continue;
        }
        if let Some(arr) = m.metadata.get("open_questions").and_then(|v| v.as_array()) {
            for q in arr {
                if let Some(text) = q.as_str() {
                    if seen.contains(text) {
                        continue;
                    }
                    let line = format!("- {text}\n");
                    if used + line.len() > recent_budget {
                        break;
                    }
                    seen.insert(text.to_string());
                    open_body.push_str(&line);
                    used += line.len();
                }
            }
        }
    }

    (decision_body, open_body)
}

fn render_error_memory(errors: &[ErrorRecord], error_budget: usize) -> String {
    let mut body = String::new();
    let mut used = 0usize;
    for err in errors {
        let line = format!(
            "- [{}] {} (seen {} times)\n",
            err.error_type, err.description, err.frequency
        );
        if used + line.len() > error_budget {
            break;
        }
        body.push_str(&line);
        used += line.len();
    }
    body
}
