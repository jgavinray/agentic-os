use crate::feature_constraint_failures::{
    constraint_type_for_failure_key, recovery_after_detection,
};
use crate::feature_extraction_types::{FeatureRecord, OperationalConstraint, SuppressedConstraint};
use chrono::{DateTime, Utc};
use std::collections::{BTreeMap, BTreeSet};

use crate::feature_constraint_budget::constraint_priority;

pub use crate::feature_constraint_budget::enforce_constraint_token_budget;
pub(crate) use crate::feature_constraint_failures::{
    failure_key_for_tag, record_latest_recovery, FailureKey,
};

pub(crate) fn build_constraints(
    record: &mut FeatureRecord,
    latest_detection: &BTreeMap<FailureKey, DateTime<Utc>>,
    latest_recovery: &BTreeMap<FailureKey, DateTime<Utc>>,
    stale_detection_keys: &BTreeSet<FailureKey>,
    max_constraints: usize,
) {
    let mut candidates = Vec::new();
    let mut suppressed = Vec::new();

    let mut active_tool_loop_tools = BTreeSet::new();
    let mut had_fresh_tool_loop = false;
    let mut recovered_tool_loop = false;
    for (key, detected_at) in latest_detection {
        let FailureKey::ToolLoop(tool) = key else {
            continue;
        };
        had_fresh_tool_loop = true;
        if recovery_after_detection(key, *detected_at, latest_recovery) {
            recovered_tool_loop = true;
        } else {
            active_tool_loop_tools.insert(tool.clone());
        }
    }

    if active_detection(
        &FailureKey::MissingAuth,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "use_known_auth",
    ) && record.known_auth_header.is_some()
    {
        candidates.push(OperationalConstraint {
            constraint_type: "use_known_auth".to_string(),
            text: format!(
                "Use `{}` when calling protected orchestrator endpoints.",
                record.known_auth_header.as_deref().unwrap()
            ),
        });
    }
    if active_detection(
        &FailureKey::WrongEndpoint,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "use_known_endpoint",
    ) && record.known_endpoint.is_some()
    {
        candidates.push(OperationalConstraint {
            constraint_type: "use_known_endpoint".to_string(),
            text: format!(
                "Do not use `localhost` for host-side orchestrator testing. The correct endpoint for this environment is `{}`.",
                record.known_endpoint.as_deref().unwrap()
            ),
        });
    }
    if active_detection(
        &FailureKey::MigrationFailure,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "use_known_migration_fix",
    ) && record.known_migration_fix.is_some()
    {
        candidates.push(OperationalConstraint {
            constraint_type: "use_known_migration_fix".to_string(),
            text: format!(
                "When retrying the baseline migration, use `{}` to make extension creation idempotent.",
                record.known_migration_fix.as_deref().unwrap()
            ),
        });
    }

    if !active_tool_loop_tools.is_empty() {
        let tools = active_tool_loop_tools
            .into_iter()
            .collect::<Vec<_>>()
            .join(" and ");
        candidates.push(OperationalConstraint {
            constraint_type: "avoid_tool_loop".to_string(),
            text: format!(
                "Do not repeat identical {tools} tool calls within this trajectory. Summarize the previous result and choose a different action before reusing the same tool with identical parameters."
            ),
        });
    } else if had_fresh_tool_loop && recovered_tool_loop {
        suppressed.push(SuppressedConstraint {
            constraint_type: "avoid_tool_loop".to_string(),
            reason: "recovery_detected".to_string(),
        });
    }

    if active_detection(
        &FailureKey::ContextPackEmpty,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "fix_context_retrieval",
    ) {
        candidates.push(OperationalConstraint {
            constraint_type: "fix_context_retrieval".to_string(),
            text: "If the context pack is empty or near-empty, verify cache warmup and retrieval health before assuming no prior memory exists.".to_string(),
        });
    }

    let context_truncated_active = active_detection(
        &FailureKey::ContextPackTruncated,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "reduce_context_bloat",
    );
    let high_input_active = active_detection(
        &FailureKey::HighInputTokens,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "reduce_context_bloat",
    );
    if context_truncated_active || high_input_active {
        candidates.push(OperationalConstraint {
            constraint_type: "reduce_context_bloat".to_string(),
            text: "Avoid expanding context further; use concise cached memory and inspect why input tokens or context truncation are high before retrying.".to_string(),
        });
    }

    if active_detection(
        &FailureKey::SummarizerSharedUpstream,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "separate_summarizer_upstream",
    ) {
        candidates.push(OperationalConstraint {
            constraint_type: "separate_summarizer_upstream".to_string(),
            text: "Keep background summarization on the dedicated summarizer endpoint instead of sharing foreground LiteLLM capacity.".to_string(),
        });
    }

    if active_detection(
        &FailureKey::UserInterruption,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "handle_user_interruption",
    ) {
        candidates.push(OperationalConstraint {
            constraint_type: "handle_user_interruption".to_string(),
            text: "When the user interrupts due to incorrect operational behavior, explicitly acknowledge the correction and apply it before continuing.".to_string(),
        });
    }

    if active_detection(
        &FailureKey::SummarizationFailure,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "handle_summarization_failure",
    ) {
        candidates.push(OperationalConstraint {
            constraint_type: "handle_summarization_failure".to_string(),
            text: "If summarization returns an empty response, inspect the provider or LiteLLM response body before retrying.".to_string(),
        });
    }

    for key in stale_detection_keys {
        if latest_detection.contains_key(key) {
            continue;
        }
        if let Some(constraint_type) = constraint_type_for_failure_key(key) {
            suppressed.push(SuppressedConstraint {
                constraint_type: constraint_type.to_string(),
                reason: "freshness_window".to_string(),
            });
        }
    }

    candidates.sort_by_key(|constraint| constraint_priority(&constraint.constraint_type));
    if candidates.len() > max_constraints {
        for dropped in candidates.drain(max_constraints..) {
            suppressed.push(SuppressedConstraint {
                constraint_type: dropped.constraint_type,
                reason: "priority_cap".to_string(),
            });
        }
    }

    suppress_duplicates(&mut suppressed);
    record.recommended_constraints = candidates;
    record.suppressed_constraints = suppressed;
}

fn active_detection(
    key: &FailureKey,
    latest_detection: &BTreeMap<FailureKey, DateTime<Utc>>,
    latest_recovery: &BTreeMap<FailureKey, DateTime<Utc>>,
    suppressed: &mut Vec<SuppressedConstraint>,
    constraint_type: &str,
) -> bool {
    let Some(detected_at) = latest_detection.get(key) else {
        return false;
    };
    if recovery_after_detection(key, *detected_at, latest_recovery) {
        suppressed.push(SuppressedConstraint {
            constraint_type: constraint_type.to_string(),
            reason: "recovery_detected".to_string(),
        });
        return false;
    }
    true
}

fn suppress_duplicates(suppressed: &mut Vec<SuppressedConstraint>) {
    let mut seen = BTreeSet::new();
    suppressed.retain(|item| seen.insert((item.constraint_type.clone(), item.reason.clone())));
    suppressed.sort_by_key(|item| {
        (
            constraint_priority(&item.constraint_type),
            item.reason.clone(),
        )
    });
}
