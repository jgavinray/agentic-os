use crate::request_classification::{
    RequestArtifactType, RequestClassification, RequestIntent, RequestRisk,
};
use chrono::Utc;

pub fn classification(
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
