use crate::context_rendering_sections::{
    append_event_section, append_failure_history_section, append_failure_section,
    append_open_questions, append_operational_constraints_section, append_search_section,
};
use crate::db::ContextEvidence;

/// BUG-4: Context is reference material only - no directive language that overrides the harness.
/// BUG-11: Merges recency and semantic results, deduplicates by summary, enforces a character budget.
pub fn build_context(
    repo: &str,
    task: &str,
    recent: &[crate::state::EventMemory],
    hybrid_hits: &[crate::state::SearchHit],
    errors: &[crate::state::ErrorRecord],
    char_budget: usize,
) -> String {
    let header = format!("Repository: {repo}\nTask: {task}\n\n");

    // Tiered budget allocations: 60% hybrid, 30% recent, 10% errors.
    let hybrid_budget = (char_budget * 60) / 100;
    let recent_budget = (char_budget * 30) / 100;
    let error_budget = char_budget.saturating_sub(hybrid_budget + recent_budget);

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut hybrid_body = String::new();
    let mut used = 0usize;
    for hit in hybrid_hits {
        if !seen.insert(hit.summary.clone()) {
            continue;
        }
        let line = format!("- [{}] {}\n", hit.event_type, hit.summary);
        if used + line.len() > hybrid_budget {
            hybrid_body.push_str("- [truncated: memory budget exceeded]\n");
            break;
        }
        hybrid_body.push_str(&line);
        used += line.len();
    }
    let hybrid_section = if hybrid_body.is_empty() {
        "== Relevant Prior Memory ==\n- No relevant prior memory found.\n".to_string()
    } else {
        format!("== Relevant Prior Memory ==\n{hybrid_body}")
    };

    let mut recent_used = 0usize;
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
        if recent_used + line.len() > recent_budget {
            break;
        }
        decision_body.push_str(&line);
        recent_used += line.len();
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
                    if recent_used + line.len() > recent_budget {
                        break;
                    }
                    seen.insert(text.to_string());
                    open_body.push_str(&line);
                    recent_used += line.len();
                }
            }
        }
    }

    let mut error_body = String::new();
    let mut used = 0usize;
    for err in errors {
        let line = format!(
            "- [{}] {} (seen {} times)\n",
            err.error_type, err.description, err.frequency
        );
        if used + line.len() > error_budget {
            break;
        }
        error_body.push_str(&line);
        used += line.len();
    }

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

pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

#[allow(clippy::too_many_arguments)]
pub fn build_layered_context(
    repo: &str,
    task: &str,
    evidence: &ContextEvidence,
    hybrid_hits: &[crate::state::SearchHit],
    errors: &[crate::state::ErrorRecord],
    policy: &crate::state::ContextPolicy,
    char_budget: usize,
    failure_history_char_budget: usize,
) -> (String, crate::state::ContextPackStats) {
    let header = format!(
        "Repository: {repo}\nTask: {task}\nContext policy: {:?}\n\n",
        policy.task_category
    );
    let mut out = header;
    let mut seen = std::collections::HashSet::new();

    append_event_section(
        &mut out,
        "== Durable Project Memory ==",
        &evidence.l3_project,
        policy.budget_for(policy.l3_project_pct, char_budget),
        &mut seen,
    );
    append_event_section(
        &mut out,
        "== Repo Patterns and Decisions ==",
        &evidence.l2_repo,
        policy.budget_for(policy.l2_repo_pct, char_budget),
        &mut seen,
    );
    append_search_section(
        &mut out,
        "== Relevant Prior Memory ==",
        hybrid_hits,
        policy.budget_for(policy.l1_matching_pct, char_budget) / 2,
        &mut seen,
    );
    append_event_section(
        &mut out,
        "== Relevant Session Summaries ==",
        &evidence.l1_matching,
        policy.budget_for(policy.l1_matching_pct, char_budget),
        &mut seen,
    );
    append_event_section(
        &mut out,
        "== Recent Evidence ==",
        &evidence.l0_recent,
        policy.budget_for(policy.l0_recent_pct, char_budget),
        &mut seen,
    );
    append_failure_section(
        &mut out,
        &evidence.failures,
        errors,
        policy.budget_for(policy.failure_pct, char_budget),
        &mut seen,
    );
    append_operational_constraints_section(&mut out, &evidence.operational_constraints);
    let reused_signatures = append_failure_history_section(
        &mut out,
        &evidence.failure_history,
        failure_history_char_budget,
        &mut seen,
    );
    append_open_questions(&mut out, &evidence.l0_recent, &mut seen);

    let mut stats = evidence.stats();
    stats.failure_history_remediation_signatures = reused_signatures;
    stats.operational_constraints_injected = evidence.operational_constraints.len();
    stats.context_chars = out.len();
    stats.context_tokens_estimate = estimate_tokens(&out);
    stats.retrieval_deduped_hits = hybrid_hits.len();

    (out, stats)
}
