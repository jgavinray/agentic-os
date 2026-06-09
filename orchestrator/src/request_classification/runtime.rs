use crate::request_classification_types::{
    LivePolicyConfig, LivePolicyDecision, RecommendedRoute, RequestArtifactType,
    RequestClassification, RequestComplexity, RequestDomain, RequestIntent, RequestRisk,
    ResponseContract,
};

pub fn evaluate_live_policy(
    classification: &RequestClassification,
    config: &LivePolicyConfig,
) -> Option<LivePolicyDecision> {
    if !config.enabled {
        return None;
    }
    if config.policy_version != "v1" {
        return None;
    }

    if classification
        .risk
        .iter()
        .any(|risk| matches!(risk, RequestRisk::UnsafeSecurity))
    {
        return Some(LivePolicyDecision {
            action: "refuse_or_guardrail",
            reason: "unsafe_security",
            route: RecommendedRoute::RefuseOrGuardrail,
            response_contract: ResponseContract::Refusal,
        });
    }
    if classification.risk.iter().any(|risk| {
        matches!(
            risk,
            RequestRisk::HighStakes | RequestRisk::DestructiveCommand | RequestRisk::SecretPresent
        )
    }) {
        return Some(LivePolicyDecision {
            action: "refuse_or_guardrail",
            reason: "objective_risk",
            route: RecommendedRoute::RefuseOrGuardrail,
            response_contract: ResponseContract::Refusal,
        });
    }
    if classification
        .risk
        .iter()
        .any(|risk| matches!(risk, RequestRisk::ExternalCurrentInfoRequired))
    {
        return Some(LivePolicyDecision {
            action: "web_required",
            reason: "external_current_info_required",
            route: RecommendedRoute::WebRequired,
            response_contract: ResponseContract::DirectAnswer,
        });
    }
    if classification.recommended_route == RecommendedRoute::AskClarification {
        return Some(LivePolicyDecision {
            action: "ask_clarification",
            reason: "missing_target_context",
            route: RecommendedRoute::AskClarification,
            response_contract: ResponseContract::ClarificationQuestion,
        });
    }
    if classification.recommended_route == RecommendedRoute::DeterministicTemplate {
        return Some(LivePolicyDecision {
            action: "deterministic_template",
            reason: "l0_trivial",
            route: RecommendedRoute::DeterministicTemplate,
            response_contract: ResponseContract::DirectAnswer,
        });
    }

    None
}

pub fn record_classification_metrics(classification: &RequestClassification) {
    crate::telemetry::record_request_classification(
        classification.intent.as_str(),
        classification.domain.as_str(),
        classification.recommended_route.as_str(),
    );
    crate::telemetry::record_request_route_recommendation(
        classification.recommended_route.as_str(),
    );
    crate::telemetry::record_request_complexity(classification.complexity.as_str());
    for risk in &classification.risk {
        crate::telemetry::record_request_risk_flag(risk.as_str());
    }
    for (field, is_unknown) in [
        ("intent", classification.intent == RequestIntent::Unknown),
        ("domain", classification.domain == RequestDomain::Unknown),
        (
            "artifact_type",
            classification.artifact_type == RequestArtifactType::Unknown,
        ),
        (
            "complexity",
            classification.complexity == RequestComplexity::Unknown,
        ),
        (
            "recommended_route",
            classification.recommended_route == RecommendedRoute::Unknown,
        ),
        (
            "response_contract",
            classification.response_contract == ResponseContract::Unknown,
        ),
    ] {
        if is_unknown {
            crate::telemetry::record_request_classification_unknown_label(field);
        }
    }
}
