use crate::request_classification_types::{
    RecommendedRoute, RequestComplexity, RequestDomain, RequestIntent, RequestRisk,
};

pub fn bounded_intent(value: &str) -> &'static str {
    RequestIntent::from_label(value).as_str()
}

pub fn bounded_domain(value: &str) -> &'static str {
    RequestDomain::from_label(value).as_str()
}

pub fn bounded_route(value: &str) -> &'static str {
    RecommendedRoute::from_label(value).as_str()
}

pub fn bounded_risk(value: &str) -> &'static str {
    RequestRisk::from_label(value).as_str()
}

pub fn bounded_complexity(value: &str) -> &'static str {
    RequestComplexity::from_label(value).as_str()
}

pub fn bounded_live_policy_action(value: &str) -> &'static str {
    match value {
        "ask_clarification" => "ask_clarification",
        "refuse_or_guardrail" => "refuse_or_guardrail",
        "web_required" => "web_required",
        "deterministic_template" => "deterministic_template",
        _ => "refuse_or_guardrail",
    }
}

pub fn bounded_live_policy_reason(value: &str) -> &'static str {
    match value {
        "unsafe_security" => "unsafe_security",
        "objective_risk" => "objective_risk",
        "external_current_info_required" => "external_current_info_required",
        "missing_target_context" => "missing_target_context",
        "l0_trivial" => "l0_trivial",
        _ => "objective_risk",
    }
}

pub fn bounded_live_policy_bypass(value: &str) -> &'static str {
    match value {
        "disabled" => "disabled",
        "shadow_only" => "shadow_only",
        "unsupported_policy_version" => "unsupported_policy_version",
        _ => "shadow_only",
    }
}
