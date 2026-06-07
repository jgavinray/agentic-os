use super::*;
use crate::request_classification::{
    RequestArtifactType, RequestClassification, RequestIntent, RequestRisk,
};
use chrono::Utc;

fn classification(
    intent: RequestIntent,
    risk: Vec<RequestRisk>,
    artifact_type: RequestArtifactType,
) -> RequestClassification {
    RequestClassification {
        event_id: "test-event".to_string(),
        repo: "test-repo".to_string(),
        session_id: "test-session".to_string(),
        trajectory_id: None,
        event_created_at: Utc::now(),
        classified_at: Utc::now(),
        classification_schema_version: 1,
        routing_policy_version: "deterministic-v1".to_string(),
        classifier_source: "deterministic_rules".to_string(),
        intent,
        domain: crate::request_classification::RequestDomain::Generic,
        secondary_domains: vec![],
        artifact_type,
        risk,
        complexity: crate::request_classification::RequestComplexity::L1Simple,
        recommended_route: crate::request_classification::RecommendedRoute::SmallLocalModel,
        response_contract: crate::request_classification::ResponseContract::DirectAnswer,
        features: serde_json::json!({}),
    }
}

#[test]
fn test_raw_capture_enabled_true_emits_raw_capture_enabled_no_low_risk() {
    let c = classification(
        RequestIntent::Explain,
        vec![RequestRisk::None],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(&c, "explain this", true);

    assert!(
        policy.risk_policy.contains(&RiskPolicy::RawCaptureEnabled),
        "raw_capture_enabled=true must emit RawCaptureEnabled"
    );
    assert!(
        !policy.risk_policy.contains(&RiskPolicy::LowRisk),
        "LowRisk must never be emitted by derive_orchestration_policy"
    );
}

#[test]
fn test_secret_present_raw_capture_emits_both() {
    let c = classification(
        RequestIntent::Explain,
        vec![RequestRisk::SecretPresent],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(&c, "show me the secrets", true);

    assert!(
        policy.risk_policy.contains(&RiskPolicy::RawCaptureEnabled),
        "SecretPresent+raw_capture must emit RawCaptureEnabled"
    );
    assert!(
        policy
            .risk_policy
            .contains(&RiskPolicy::SecretCaptureAllowed),
        "SecretPresent+raw_capture must emit SecretCaptureAllowed"
    );
    assert_eq!(
        policy
            .risk_policy
            .iter()
            .filter(|&&r| r == RiskPolicy::RawCaptureEnabled)
            .count(),
        1,
        "RawCaptureEnabled must appear exactly once"
    );
}

#[test]
fn test_secret_present_no_raw_capture_emits_neither() {
    let c = classification(
        RequestIntent::Explain,
        vec![RequestRisk::SecretPresent],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(&c, "show me the secrets", false);

    assert!(
        !policy.risk_policy.contains(&RiskPolicy::RawCaptureEnabled),
        "SecretPresent+no raw_capture must not emit RawCaptureEnabled"
    );
    assert!(
        !policy
            .risk_policy
            .contains(&RiskPolicy::SecretCaptureAllowed),
        "SecretPresent+no raw_capture must not emit SecretCaptureAllowed"
    );
}

#[test]
fn test_raw_capture_explain_includes_raw_capture_features_only() {
    let c = classification(
        RequestIntent::Explain,
        vec![RequestRisk::None],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(&c, "explain this", true);

    assert!(
        policy
            .context_sources
            .contains(&ContextSource::RawCaptureFeatures),
        "raw_capture_enabled=true must include RawCaptureFeatures"
    );
    assert!(
        !policy.context_sources.contains(&ContextSource::TotalRecall),
        "raw capture must not widen narrow explain requests into TotalRecall"
    );
    assert!(
        !policy
            .context_sources
            .contains(&ContextSource::CompiledSummaries),
        "raw capture must not inject compiled summaries into narrow explain requests"
    );
}

#[test]
fn test_no_raw_capture_explain_no_raw_capture_features() {
    let c = classification(
        RequestIntent::Explain,
        vec![RequestRisk::None],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(&c, "explain this", false);

    assert!(
        !policy
            .context_sources
            .contains(&ContextSource::RawCaptureFeatures),
        "raw_capture_enabled=false must not include RawCaptureFeatures"
    );
}

#[test]
fn test_raw_capture_features_no_duplicates() {
    let c = classification(
        RequestIntent::Explain,
        vec![RequestRisk::None],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(&c, "explain this", true);

    let count = policy
        .context_sources
        .iter()
        .filter(|&&s| s == ContextSource::RawCaptureFeatures)
        .count();
    assert_eq!(count, 1, "RawCaptureFeatures must appear exactly once");
}
