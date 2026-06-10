use crate::request_classification_features::RequestFeatures;
use crate::request_classification_rule_utils::{contains_any, contains_word};
use crate::request_classification_types::RequestIntent;

/// Intent decision with scoring observability.
///
/// `margin` is the weight gap between the winning intent and the runner-up;
/// small margins flag requests worth inspecting when classification looks
/// wrong in captured traffic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ScoredIntent {
    pub(crate) intent: RequestIntent,
    pub(crate) weight: u32,
    pub(crate) runner_up: Option<RequestIntent>,
    pub(crate) margin: u32,
}

pub(crate) fn classify_intent(
    features: &RequestFeatures,
    lower: &str,
    event_type: &str,
) -> RequestIntent {
    classify_intent_scored(features, lower, event_type).intent
}

/// Score every matching intent rule and pick the heaviest.
///
/// Weights encode precedence explicitly (the old if/else cascade encoded it
/// implicitly in branch order). Mutation evidence deliberately outweighs
/// incidental read-intent keywords: "make the failing test pass" is
/// implementation even though "failing test" also matches the debug rule.
pub(crate) fn classify_intent_scored(
    features: &RequestFeatures,
    lower: &str,
    event_type: &str,
) -> ScoredIntent {
    if features.char_count == 0 {
        return ScoredIntent {
            intent: RequestIntent::Unknown,
            weight: 0,
            runner_up: None,
            margin: 0,
        };
    }

    // Negated verbs must not drive intent: "do not implement this yet" is not
    // an implementation request. The prompt-intervention layer separately
    // records the prohibition itself.
    let lower = strip_negated_verbs(lower);
    let lower = lower.as_str();

    let mut candidates: Vec<(RequestIntent, u32)> = Vec::new();

    if contains_any(
        lower,
        &[
            "implement",
            "implementation",
            "build this",
            "build the",
            "add feature",
            "add support",
            "add functionality",
            "refactor",
            "wire up",
            "integrate",
        ],
    ) {
        candidates.push((RequestIntent::Implement, 100));
    }
    // Additive phrasing is implementation unless the request is config-shaped,
    // where ModifyConfig (85) should win ("update config.yaml to add ...").
    if contains_any(lower, &["add a ", "add the ", "create a "]) {
        candidates.push((RequestIntent::Implement, 78));
    }
    if contains_any(
        lower,
        &["generate config", "create yaml", "write yaml", "manifest"],
    ) {
        candidates.push((RequestIntent::GenerateConfig, 90));
    }
    if features.has_config_shape
        && contains_any(lower, &["edit", "modify", "change", "fix", "update"])
    {
        candidates.push((RequestIntent::ModifyConfig, 85));
    }
    if contains_any(lower, &["fix", "repair", "patch", "resolve"])
        && contains_any(
            lower,
            &[
                "bug",
                "issue",
                "regression",
                "failing test",
                "test failure",
                "tests failed",
                "failure",
                "typo",
                "code",
                "repo",
                "software",
            ],
        )
    {
        candidates.push((RequestIntent::Implement, 80));
    }
    if starts_with_mutation_verb(lower) {
        candidates.push((RequestIntent::Implement, 75));
    }
    if contains_word(lower, &["summarize", "summary", "recap"]) {
        candidates.push((RequestIntent::Summarize, 70));
    }
    if contains_word(lower, &["classify", "categorize", "label this"]) {
        candidates.push((RequestIntent::Classify, 65));
    }
    if contains_word(lower, &["diagnose", "troubleshoot", "root cause"]) {
        candidates.push((RequestIntent::Debug, 62));
    }
    if contains_word(lower, &["search", "look up", "find current", "latest"]) {
        candidates.push((RequestIntent::Search, 60));
    }
    if contains_word(lower, &["plan", "proposal", "approach", "design"]) {
        candidates.push((RequestIntent::Plan, 55));
    }
    if contains_word(lower, &["run", "execute", "deploy", "restart"]) || event_type == "tool_call" {
        candidates.push((RequestIntent::OperateTool, 50));
    }
    if features.contains_error_words || features.has_stack_trace || features.has_test_failure {
        candidates.push((RequestIntent::Debug, 48));
    }
    // Questions are explanations unless stronger evidence wins.
    if starts_with_question_word(lower) {
        candidates.push((RequestIntent::Explain, 52));
    }
    if contains_word(
        lower,
        &[
            "explain",
            "explanation",
            "describe",
            "what does",
            "how does",
            "why does",
            "walk me through",
        ],
    ) {
        candidates.push((RequestIntent::Explain, 46));
    }
    // Read verbs keep terse read requests ("Read README.md") out of the
    // code-shaped implement fallback below.
    if contains_word(
        lower,
        &["read", "show", "view", "open", "inspect", "look at", "list"],
    ) {
        candidates.push((RequestIntent::Explain, 40));
    }
    // Code-shaped fallback: terse dev shorthand referencing files or code with
    // no recognized verb gets the implement surface. The operating envelope is
    // the safety layer; a read-only Explain menu just wedges the agent loop.
    if features.has_file_path || features.has_code_block || features.has_diff_or_patch {
        candidates.push((RequestIntent::Implement, 5));
    }
    candidates.push((RequestIntent::Explain, 1));

    pick_winner(&candidates)
}

fn pick_winner(candidates: &[(RequestIntent, u32)]) -> ScoredIntent {
    // Collapse to the heaviest weight per intent, then order by weight with a
    // stable label tie-break so the result is deterministic.
    let mut best: Vec<(RequestIntent, u32)> = Vec::new();
    for (intent, weight) in candidates {
        match best.iter_mut().find(|(existing, _)| existing == intent) {
            Some((_, existing_weight)) => *existing_weight = (*existing_weight).max(*weight),
            None => best.push((*intent, *weight)),
        }
    }
    best.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.as_str().cmp(b.0.as_str())));

    let (intent, weight) = best[0];
    let runner_up = best.get(1).copied();
    ScoredIntent {
        intent,
        weight,
        runner_up: runner_up.map(|(intent, _)| intent),
        margin: weight - runner_up.map(|(_, weight)| weight).unwrap_or(0),
    }
}

/// Remove "do not <verb>" / "don't <verb>" so the negated verb cannot drive
/// intent selection. Only the verb immediately after the negation is dropped.
fn strip_negated_verbs(lower: &str) -> String {
    let mut out = String::with_capacity(lower.len());
    let mut rest = lower;
    loop {
        let negation = ["do not ", "don't "]
            .iter()
            .filter_map(|marker| rest.find(marker).map(|pos| (pos, marker.len())))
            .min_by_key(|(pos, _)| *pos);
        let Some((pos, marker_len)) = negation else {
            out.push_str(rest);
            return out;
        };
        out.push_str(&rest[..pos]);
        let after_marker = &rest[pos + marker_len..];
        let verb_end = after_marker
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-')
            .unwrap_or(after_marker.len());
        rest = &after_marker[verb_end..];
    }
}

fn starts_with_question_word(lower: &str) -> bool {
    const QUESTION_STARTS: &[&str] = &[
        "why ", "what ", "how ", "when ", "where ", "who ", "which ", "can ", "could ", "does ",
        "do ", "is ", "are ", "should ", "would ",
    ];
    let trimmed = lower.trim_start();
    QUESTION_STARTS.iter().any(|word| trimmed.starts_with(word))
}

fn starts_with_mutation_verb(lower: &str) -> bool {
    const MUTATION_STARTS: &[&str] = &[
        "make ", "set ", "ensure ", "convert ", "move ", "rename ", "replace ", "remove ",
        "split ", "extract ", "hook ", "port ", "migrate ", "bump ", "upgrade ", "swap ", "turn ",
        "adjust ",
    ];
    let trimmed = lower.trim_start();
    MUTATION_STARTS.iter().any(|verb| trimmed.starts_with(verb))
}
