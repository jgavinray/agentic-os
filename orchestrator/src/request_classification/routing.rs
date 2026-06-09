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
