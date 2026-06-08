use crate::context_rendering_sections::{
    append_event_section, append_failure_history_section, append_failure_section,
    append_open_questions, append_operational_constraints_section, append_search_section,
};
use crate::db::ContextEvidence;

pub use crate::context_rendering_flat::build_context;

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
