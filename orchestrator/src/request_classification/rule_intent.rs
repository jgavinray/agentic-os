use crate::request_classification_features::RequestFeatures;
use crate::request_classification_rule_utils::contains_any;
use crate::request_classification_types::RequestIntent;

pub(crate) fn classify_intent(
    features: &RequestFeatures,
    lower: &str,
    event_type: &str,
) -> RequestIntent {
    if features.char_count == 0 {
        return RequestIntent::Unknown;
    }
    if contains_any(lower, &["summarize", "summary", "recap"]) {
        RequestIntent::Summarize
    } else if contains_any(lower, &["classify", "categorize", "label this"]) {
        RequestIntent::Classify
    } else if contains_any(lower, &["search", "look up", "find current", "latest"]) {
        RequestIntent::Search
    } else if contains_any(lower, &["plan", "proposal", "approach", "design"]) {
        RequestIntent::Plan
    } else if contains_any(
        lower,
        &[
            "implement",
            "implementation",
            "build this",
            "build the",
            "add feature",
            "add support",
            "add functionality",
            "wire up",
            "integrate",
        ],
    ) {
        RequestIntent::Implement
    } else if contains_any(
        lower,
        &["generate config", "create yaml", "write yaml", "manifest"],
    ) {
        RequestIntent::GenerateConfig
    } else if features.has_config_shape
        && contains_any(lower, &["edit", "modify", "change", "fix", "update"])
    {
        RequestIntent::ModifyConfig
    } else if contains_any(lower, &["run ", "execute ", "deploy", "restart"])
        || event_type == "tool_call"
    {
        RequestIntent::OperateTool
    } else if features.contains_error_words || features.has_stack_trace || features.has_test_failure
    {
        RequestIntent::Debug
    } else {
        RequestIntent::Explain
    }
}
