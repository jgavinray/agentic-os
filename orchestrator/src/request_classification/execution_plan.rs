//! Render composite requests as an ordered execution plan.
//!
//! Composition analysis already detects multi-step requests and their
//! sub-intents, but nothing consumed it: a small model received the whole
//! composite prompt at once and routinely interleaved or skipped steps.
//! This module re-derives the step fragments deterministically and renders
//! them as an ordered plan injected into system context — one step at a time,
//! validation before moving on.

use serde_json::{json, Value};

use crate::request_classification::RequestClassification;
use crate::request_classification_features::extract_features;
use crate::request_classification_fragments::{decomposition_fragments, has_subtask_action_signal};
use crate::request_classification_rules::classify_intent;

pub const EXECUTION_PLAN_VERSION: &str = "execution-plan-v1";
const MAX_PLAN_STEPS: usize = 5;
const MAX_STEP_CHARS: usize = 140;

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionPlan {
    pub guidance: String,
    pub metadata: Value,
}

/// Build an ordered execution plan for a composite request, if the
/// classification marked it as a decomposition candidate.
pub fn execution_plan(
    user_content: &str,
    classification: &RequestClassification,
) -> Option<ExecutionPlan> {
    if classification.features["decomposition_candidate"] != Value::Bool(true) {
        return None;
    }

    let lower = user_content.to_ascii_lowercase();
    let (fragments, _reason) = decomposition_fragments(user_content, &lower);
    let steps: Vec<(String, &'static str)> = fragments
        .iter()
        .take(MAX_PLAN_STEPS)
        .filter_map(|fragment| {
            let fragment = fragment.trim();
            if fragment.len() < 3 {
                return None;
            }
            let fragment_lower = fragment.to_ascii_lowercase();
            if !has_subtask_action_signal(&fragment_lower) {
                return None;
            }
            let features = extract_features(fragment, &fragment_lower, "");
            let intent = classify_intent(&features, &fragment_lower, "user_message");
            Some((bounded_step(fragment), intent.as_str()))
        })
        .collect();

    if steps.len() < 2 {
        return None;
    }

    let mut lines = vec![
        "== Execution Plan ==".to_string(),
        "This request has multiple steps. Work strictly in this order. Finish each step \
         (including its validation) before starting the next; do not interleave steps."
            .to_string(),
    ];
    for (index, (step, intent)) in steps.iter().enumerate() {
        lines.push(format!("{}. [{intent}] {step}", index + 1));
    }
    lines.push("After the final step, report the outcome of each step.".to_string());

    let metadata = json!({
        "execution_plan": {
            "version": EXECUTION_PLAN_VERSION,
            "step_count": steps.len(),
            "step_intents": steps.iter().map(|(_, intent)| *intent).collect::<Vec<_>>(),
        }
    });

    Some(ExecutionPlan {
        guidance: lines.join("\n"),
        metadata,
    })
}

fn bounded_step(fragment: &str) -> String {
    let cleaned = fragment.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.chars().count() <= MAX_STEP_CHARS {
        return cleaned;
    }
    let mut bounded: String = cleaned.chars().take(MAX_STEP_CHARS).collect();
    bounded.push('…');
    bounded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_classification::classify_request_text;

    #[test]
    fn composite_request_renders_ordered_plan() {
        let text = "Search the repo for context injection; implement the fix in src/main.rs; \
                    run cargo test; summarize the result";
        let classification =
            classify_request_text("agentic-os", "session-p", text, None, "user_message");

        let plan = execution_plan(text, &classification).expect("plan");

        assert!(plan.guidance.contains("== Execution Plan =="));
        assert!(plan.guidance.contains("1. [search]"));
        assert!(plan.guidance.contains("2. [implement]"));
        assert!(plan.guidance.contains("Work strictly in this order"));
        assert_eq!(plan.metadata["execution_plan"]["step_count"], 4);
    }

    #[test]
    fn single_step_request_has_no_plan() {
        let text = "Implement the retry policy fix in src/handlers.rs";
        let classification =
            classify_request_text("agentic-os", "session-p", text, None, "user_message");

        assert!(execution_plan(text, &classification).is_none());
    }

    #[test]
    fn conjunction_without_actions_has_no_plan() {
        let text = "Explain Docker and Kubernetes networking";
        let classification =
            classify_request_text("agentic-os", "session-p", text, None, "user_message");

        assert!(execution_plan(text, &classification).is_none());
    }
}
