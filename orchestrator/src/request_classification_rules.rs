use crate::request_classification_features::RequestFeatures;
use crate::request_classification_types::{
    RecommendedRoute, RequestArtifactType, RequestComplexity, RequestDomain, RequestIntent,
    RequestRisk, ResponseContract,
};

pub(crate) fn detected_domains(features: &RequestFeatures, lower: &str) -> Vec<RequestDomain> {
    let mut domains = Vec::new();
    push_if(
        &mut domains,
        features.has_kubernetes_terms,
        RequestDomain::Kubernetes,
    );
    push_if(
        &mut domains,
        features.has_docker_terms,
        RequestDomain::Docker,
    );
    push_if(
        &mut domains,
        features.has_llm_terms,
        RequestDomain::LlmInference,
    );
    push_if(
        &mut domains,
        features.has_networking_terms,
        RequestDomain::Networking,
    );
    push_if(
        &mut domains,
        features.has_security_terms,
        RequestDomain::Security,
    );
    push_if(
        &mut domains,
        contains_any(lower, &["medical", "doctor", "diagnosis", "medicine"]),
        RequestDomain::Medical,
    );
    push_if(
        &mut domains,
        contains_any(lower, &["legal", "lawyer", "lawsuit", "contract"]),
        RequestDomain::Legal,
    );
    push_if(
        &mut domains,
        contains_any(lower, &["finance", "tax", "investment", "stock", "loan"]),
        RequestDomain::Finance,
    );
    push_if(
        &mut domains,
        features.has_shell_command,
        RequestDomain::Shell,
    );
    domains
}

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

pub(crate) fn classify_domain(
    features: &RequestFeatures,
    lower: &str,
    detected_domains: &[RequestDomain],
) -> RequestDomain {
    for high_stakes in [
        RequestDomain::Medical,
        RequestDomain::Legal,
        RequestDomain::Finance,
    ] {
        if detected_domains.contains(&high_stakes) {
            return high_stakes;
        }
    }
    if features.char_count == 0 {
        RequestDomain::Unknown
    } else if let Some(domain) = detected_domains.first() {
        *domain
    } else if contains_any(lower, &["shell", "terminal", "bash", "zsh"]) {
        RequestDomain::Shell
    } else {
        RequestDomain::Generic
    }
}

pub(crate) fn classify_artifact(features: &RequestFeatures, lower: &str) -> RequestArtifactType {
    if features.char_count == 0 {
        RequestArtifactType::Unknown
    } else if features.has_logs || features.has_stack_trace || features.has_test_failure {
        RequestArtifactType::Logs
    } else if features.has_json {
        RequestArtifactType::Json
    } else if features.has_yaml {
        RequestArtifactType::Yaml
    } else if contains_any(
        lower,
        &["sql", "select ", "insert into", "update ", "delete from"],
    ) {
        RequestArtifactType::Sql
    } else if contains_any(lower, &["markdown", ".md", "# "]) {
        RequestArtifactType::Markdown
    } else if features.has_code_block || features.has_diff_or_patch {
        RequestArtifactType::Code
    } else if contains_any(lower, &["image", "screenshot", ".png", ".jpg", ".jpeg"]) {
        RequestArtifactType::Image
    } else if features.has_file_path || features.asks_for_file_generation {
        RequestArtifactType::File
    } else {
        RequestArtifactType::PlainText
    }
}

pub(crate) fn classify_risk(
    features: &RequestFeatures,
    lower: &str,
    domain: RequestDomain,
) -> Vec<RequestRisk> {
    if features.char_count == 0 {
        return vec![RequestRisk::Unknown];
    }

    let mut risks = Vec::new();
    push_if(
        &mut risks,
        features.has_secret_candidate,
        RequestRisk::SecretPresent,
    );
    push_if(
        &mut risks,
        features.contains_destructive_verbs,
        RequestRisk::DestructiveCommand,
    );
    push_if(
        &mut risks,
        features.asks_for_latest,
        RequestRisk::ExternalCurrentInfoRequired,
    );
    push_if(
        &mut risks,
        matches!(
            domain,
            RequestDomain::Medical | RequestDomain::Legal | RequestDomain::Finance
        ),
        RequestRisk::HighStakes,
    );
    push_if(
        &mut risks,
        contains_any(
            lower,
            &[
                "ignore previous",
                "ignore all previous",
                "system prompt",
                "developer message",
                "jailbreak",
            ],
        ),
        RequestRisk::PromptInjection,
    );
    push_if(
        &mut risks,
        contains_any(
            lower,
            &[
                "steal credentials",
                "credential dump",
                "phishing",
                "malware",
                "exploit this",
                "bypass auth",
            ],
        ),
        RequestRisk::UnsafeSecurity,
    );

    if risks.is_empty() {
        risks.push(RequestRisk::None);
    }
    risks
}

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
        RequestIntent::OperateTool | RequestIntent::Implement | RequestIntent::ModifyConfig
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

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

fn push_if<T: PartialEq + Copy>(items: &mut Vec<T>, condition: bool, item: T) {
    if condition && !items.contains(&item) {
        items.push(item);
    }
}
