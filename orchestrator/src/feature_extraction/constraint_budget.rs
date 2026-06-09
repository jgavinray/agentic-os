use crate::feature_extraction_types::{OperationalConstraint, SuppressedConstraint};

const CONSTRAINT_PRIORITY: [&str; 9] = [
    "use_known_auth",
    "use_known_endpoint",
    "use_known_migration_fix",
    "avoid_tool_loop",
    "fix_context_retrieval",
    "reduce_context_bloat",
    "separate_summarizer_upstream",
    "handle_user_interruption",
    "handle_summarization_failure",
];

pub(crate) fn constraint_priority(constraint_type: &str) -> usize {
    CONSTRAINT_PRIORITY
        .iter()
        .position(|candidate| *candidate == constraint_type)
        .unwrap_or(usize::MAX)
}

pub fn enforce_constraint_token_budget(
    constraints: &[OperationalConstraint],
    budget_tokens: usize,
) -> (Vec<OperationalConstraint>, Vec<SuppressedConstraint>) {
    if constraints.is_empty() || budget_tokens == 0 {
        return (
            vec![],
            constraints
                .iter()
                .map(|constraint| SuppressedConstraint {
                    constraint_type: constraint.constraint_type.clone(),
                    reason: "token_budget".to_string(),
                })
                .collect(),
        );
    }

    let mut kept = constraints.to_vec();
    kept.sort_by_key(|constraint| constraint_priority(&constraint.constraint_type));
    let mut dropped = Vec::new();
    while estimate_constraint_tokens(&kept) > budget_tokens {
        let Some(constraint) = kept.pop() else {
            break;
        };
        dropped.push(SuppressedConstraint {
            constraint_type: constraint.constraint_type,
            reason: "token_budget".to_string(),
        });
    }
    dropped.sort_by_key(|item| constraint_priority(&item.constraint_type));
    (kept, dropped)
}

fn estimate_constraint_tokens(constraints: &[OperationalConstraint]) -> usize {
    let body_len: usize = constraints
        .iter()
        .map(|constraint| constraint.text.len() + 3)
        .sum();
    ("Operational Constraints:\n".len() + body_len).div_ceil(4)
}
