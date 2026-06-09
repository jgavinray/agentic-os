use crate::request_classification_features::RequestFeatures;
use crate::request_classification_rule_utils::{contains_any, push_if};
use crate::request_classification_types::{RequestDomain, RequestRisk};

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
