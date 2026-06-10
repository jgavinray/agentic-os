use crate::request_classification_features::RequestFeatures;
use crate::request_classification_types::{
    RecommendedRoute, RequestArtifactType, RequestComplexity, RequestIntent, RequestRisk,
    ResponseContract,
};

pub(crate) fn recommend_route(
    intent: RequestIntent,
    complexity: RequestComplexity,
    risks: &[RequestRisk],
    features: &RequestFeatures,
) -> RecommendedRoute {
    if risks.iter().any(|risk| {
        matches!(
            risk,
            RequestRisk::UnsafeSecurity
                | RequestRisk::HighStakes
                | RequestRisk::DestructiveCommand
                | RequestRisk::SecretPresent
        )
    }) {
        RecommendedRoute::RefuseOrGuardrail
    } else if risks.contains(&RequestRisk::ExternalCurrentInfoRequired) {
        RecommendedRoute::WebRequired
    } else if matches!(
        intent,
        RequestIntent::OperateTool | RequestIntent::ModifyConfig
    ) && !features.has_file_path
        && !features.has_config_shape
        && !features.has_shell_command
    {
        RecommendedRoute::AskClarification
    } else if matches!(intent, RequestIntent::OperateTool) || features.has_shell_command {
        RecommendedRoute::ToolRequired
    } else if matches!(complexity, RequestComplexity::L0Trivial) {
        RecommendedRoute::DeterministicTemplate
    } else if matches!(
        (intent, complexity),
        (
            RequestIntent::Explain | RequestIntent::Summarize | RequestIntent::Classify,
            RequestComplexity::L1Simple | RequestComplexity::L2Moderate
        )
    ) {
        RecommendedRoute::SmallLocalModel
    } else if matches!(complexity, RequestComplexity::L3Complex) {
        RecommendedRoute::StrongLocalModel
    } else if matches!(complexity, RequestComplexity::Unknown) {
        RecommendedRoute::Unknown
    } else {
        RecommendedRoute::SmallLocalModel
    }
}

/// Map a recommended route to a configured model override.
///
/// Only the small/strong local-model routes participate: guardrail,
/// clarification, web, and tool routes describe *how* to answer, not which
/// model tier to use, and Unknown must never override the client's choice.
pub fn classification_model_override(
    route: RecommendedRoute,
    route_model_small: Option<&str>,
    route_model_strong: Option<&str>,
) -> Option<String> {
    match route {
        RecommendedRoute::SmallLocalModel => route_model_small.map(str::to_string),
        RecommendedRoute::StrongLocalModel => route_model_strong.map(str::to_string),
        _ => None,
    }
}

#[cfg(test)]
mod override_tests {
    use super::*;

    #[test]
    fn small_and_strong_routes_map_to_configured_models() {
        assert_eq!(
            classification_model_override(
                RecommendedRoute::SmallLocalModel,
                Some("small-7b"),
                Some("strong-32b"),
            )
            .as_deref(),
            Some("small-7b")
        );
        assert_eq!(
            classification_model_override(
                RecommendedRoute::StrongLocalModel,
                Some("small-7b"),
                Some("strong-32b"),
            )
            .as_deref(),
            Some("strong-32b")
        );
    }

    #[test]
    fn other_routes_and_missing_config_do_not_override() {
        assert!(classification_model_override(
            RecommendedRoute::RefuseOrGuardrail,
            Some("small-7b"),
            Some("strong-32b"),
        )
        .is_none());
        assert!(classification_model_override(
            RecommendedRoute::Unknown,
            Some("small-7b"),
            Some("strong-32b"),
        )
        .is_none());
        assert!(
            classification_model_override(RecommendedRoute::SmallLocalModel, None, None).is_none()
        );
    }
}

pub(crate) fn response_contract(
    intent: RequestIntent,
    artifact_type: RequestArtifactType,
    route: RecommendedRoute,
) -> ResponseContract {
    match route {
        RecommendedRoute::RefuseOrGuardrail => ResponseContract::Refusal,
        RecommendedRoute::AskClarification => ResponseContract::ClarificationQuestion,
        RecommendedRoute::ToolRequired => ResponseContract::ValidationRequired,
        _ if matches!(intent, RequestIntent::Classify) => ResponseContract::StructuredJson,
        _ if matches!(intent, RequestIntent::Summarize | RequestIntent::Plan) => {
            ResponseContract::MarkdownSummary
        }
        _ if matches!(intent, RequestIntent::Implement) => ResponseContract::ValidationRequired,
        _ if matches!(
            artifact_type,
            RequestArtifactType::Code | RequestArtifactType::Yaml | RequestArtifactType::Json
        ) =>
        {
            ResponseContract::ValidationRequired
        }
        _ if matches!(route, RecommendedRoute::Unknown) => ResponseContract::Unknown,
        _ => ResponseContract::DirectAnswer,
    }
}
