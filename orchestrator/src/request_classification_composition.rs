use crate::request_classification_features::extract_features;
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

pub(crate) fn decomposition_fragments(text: &str, lower: &str) -> (Vec<String>, &'static str) {
    let structured = structured_list_fragments(text);
    if structured.len() >= 2 {
        return (structured, "structured_list");
    }

    for (separator, reason) in [
        ("\n", "line_separated"),
        (";", "sequence_separator"),
        (", then ", "sequence_separator"),
        (" then ", "sequence_separator"),
    ] {
        if lower.contains(separator) {
            let fragments = split_nonempty(text, separator);
            if fragments.len() >= 2 {
                return (fragments, reason);
            }
        }
    }

    if action_signal_count(lower) >= 2 {
        let coordinated = split_nonempty(text, " and ");
        if coordinated.len() >= 2 {
            return (coordinated, "coordinated_actions");
        }

        let comma_actions = split_on_action_commas(text);
        if comma_actions.len() >= 2 {
            return (comma_actions, "coordinated_actions");
        }
    }

    (Vec::new(), "none")
}

fn structured_list_fragments(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            let item = trimmed
                .strip_prefix("- ")
                .or_else(|| trimmed.strip_prefix("* "))
                .or_else(|| numbered_item_text(trimmed))?;
            let item = item.trim();
            (!item.is_empty()).then(|| item.to_string())
        })
        .take(5)
        .collect()
}

fn numbered_item_text(value: &str) -> Option<&str> {
    let split_at = value
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit())
        .last()
        .map(|(idx, ch)| idx + ch.len_utf8())?;
    let rest = value.get(split_at..)?;
    rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") "))
}

fn split_nonempty(text: &str, separator: &str) -> Vec<String> {
    text.split(separator)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .take(5)
        .collect()
}

fn split_on_action_commas(text: &str) -> Vec<String> {
    let mut fragments = Vec::new();
    for part in text.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if fragments.is_empty() || has_subtask_action_signal(&part.to_ascii_lowercase()) {
            fragments.push(part.to_string());
        } else if let Some(last) = fragments.last_mut() {
            last.push_str(", ");
            last.push_str(part);
        }
    }
    fragments.into_iter().take(5).collect()
}

fn action_signal_count(lower: &str) -> usize {
    SUBTASK_ACTION_SIGNALS
        .iter()
        .filter(|signal| lower.contains(**signal))
        .count()
}

pub(crate) fn has_subtask_action_signal(lower: &str) -> bool {
    contains_any(lower, SUBTASK_ACTION_SIGNALS)
}

const SUBTASK_ACTION_SIGNALS: &[&str] = &[
    "investigate",
    "inspect",
    "look at",
    "look up",
    "search",
    "find",
    "read",
    "open",
    "review",
    "explain",
    "summarize",
    "classify",
    "plan",
    "design",
    "debug",
    "fix",
    "patch",
    "edit",
    "modify",
    "change",
    "update",
    "add",
    "remove",
    "create",
    "write",
    "generate",
    "run",
    "execute",
    "validate",
    "test",
    "deploy",
    "restart",
];

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}
