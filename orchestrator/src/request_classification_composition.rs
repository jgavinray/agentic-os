use crate::request_classification_features::extract_features;
use crate::request_classification_fragments::{decomposition_fragments, has_subtask_action_signal};
use crate::request_classification_rules::classify_intent;
use crate::request_classification_types::RequestIntent;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompositeAnalysis {
    pub(crate) is_composite: bool,
    pub(crate) decomposition_candidate: bool,
    pub(crate) reason: &'static str,
    pub(crate) sub_intents: Vec<RequestIntent>,
}

pub(crate) fn analyze_composition(text: &str, lower: &str, event_type: &str) -> CompositeAnalysis {
    if text.trim().is_empty() {
        return CompositeAnalysis {
            is_composite: false,
            decomposition_candidate: false,
            reason: "none",
            sub_intents: Vec::new(),
        };
    }

    let (fragments, reason) = decomposition_fragments(text, lower);
    let mut sub_intents = Vec::new();
    for fragment in fragments.iter().take(5) {
        let fragment = fragment.trim();
        if fragment.len() < 3 {
            continue;
        }
        let fragment_lower = fragment.to_ascii_lowercase();
        if !has_subtask_action_signal(&fragment_lower) {
            continue;
        }
        let features = extract_features(fragment, &fragment_lower, "");
        sub_intents.push(classify_intent(&features, &fragment_lower, event_type));
    }

    let decomposition_candidate = sub_intents.len() >= 2;
    CompositeAnalysis {
        is_composite: decomposition_candidate,
        decomposition_candidate,
        reason: if decomposition_candidate {
            reason
        } else {
            "none"
        },
        sub_intents: if decomposition_candidate {
            sub_intents
        } else {
            Vec::new()
        },
    }
}
