use crate::request_classification_features::RequestFeatures;
use crate::request_classification_types::{RequestComplexity, RequestIntent, RequestRisk};

pub(crate) fn classify_complexity(
    features: &RequestFeatures,
    intent: RequestIntent,
    risks: &[RequestRisk],
    domain_count: usize,
) -> RequestComplexity {
    if features.char_count == 0 {
        RequestComplexity::Unknown
    } else if risks.iter().any(|risk| {
        matches!(
            risk,
            RequestRisk::HighStakes
                | RequestRisk::UnsafeSecurity
                | RequestRisk::DestructiveCommand
                | RequestRisk::SecretPresent
        )
    }) {
        RequestComplexity::L5HighRisk
    } else if matches!(intent, RequestIntent::OperateTool)
        || features.has_shell_command
        || features.asks_for_file_generation
    {
        RequestComplexity::L4ToolRequired
    } else if features.char_count > 2_000
        || features.line_count > 60
        || features.has_stack_trace
        || features.has_diff_or_patch
        || features.has_test_failure
    {
        RequestComplexity::L3Complex
    } else if domain_count > 1
        || matches!(
            intent,
            RequestIntent::Debug
                | RequestIntent::Implement
                | RequestIntent::GenerateConfig
                | RequestIntent::ModifyConfig
        )
        || features.has_config_shape
    {
        RequestComplexity::L2Moderate
    } else if features.char_count <= 40 && !features.contains_error_words {
        RequestComplexity::L0Trivial
    } else {
        RequestComplexity::L1Simple
    }
}
