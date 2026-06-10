use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub use crate::request_classification_feature_keys::FEATURE_KEYS;
pub use crate::request_classification_taxonomy::{
    enum_inventory, RecommendedRoute, RequestArtifactType, RequestComplexity, RequestDomain,
    RequestIntent, RequestRisk, ResponseContract,
};

pub const CLASSIFICATION_SCHEMA_VERSION: i32 = 2;
pub const ROUTING_POLICY_VERSION: &str = "deterministic-v1";
pub const CLASSIFIER_SOURCE_DETERMINISTIC_RULES: &str = "deterministic_rules";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestClassification {
    pub event_id: String,
    pub repo: String,
    pub session_id: String,
    pub trajectory_id: Option<Uuid>,
    pub event_created_at: DateTime<Utc>,
    pub classified_at: DateTime<Utc>,
    pub classification_schema_version: i32,
    pub routing_policy_version: String,
    pub classifier_source: String,
    pub intent: RequestIntent,
    pub domain: RequestDomain,
    pub secondary_domains: Vec<RequestDomain>,
    pub artifact_type: RequestArtifactType,
    pub risk: Vec<RequestRisk>,
    pub complexity: RequestComplexity,
    pub recommended_route: RecommendedRoute,
    pub response_contract: ResponseContract,
    pub features: Value,
}

impl RequestClassification {
    pub fn deterministic(
        event_id: String,
        repo: String,
        session_id: String,
        event_created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            event_id,
            repo,
            session_id,
            trajectory_id: None,
            event_created_at,
            classified_at: event_created_at,
            classification_schema_version: CLASSIFICATION_SCHEMA_VERSION,
            routing_policy_version: ROUTING_POLICY_VERSION.to_string(),
            classifier_source: CLASSIFIER_SOURCE_DETERMINISTIC_RULES.to_string(),
            intent: RequestIntent::Unknown,
            domain: RequestDomain::Unknown,
            secondary_domains: Vec::new(),
            artifact_type: RequestArtifactType::Unknown,
            risk: vec![RequestRisk::Unknown],
            complexity: RequestComplexity::Unknown,
            recommended_route: RecommendedRoute::Unknown,
            response_contract: ResponseContract::Unknown,
            features: Value::Object(Default::default()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BackfillOptions {
    pub repo: Option<String>,
    pub session_id: Option<String>,
    pub since: Option<DateTime<Utc>>,
    pub dry_run: bool,
    pub repair: bool,
    pub repair_stale: bool,
    pub batch_size: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BackfillReport {
    pub events_scanned: usize,
    pub inserted: usize,
    pub updated: usize,
    pub skipped: usize,
    pub dry_run: bool,
    pub batch_size: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistOutcome {
    Inserted,
    Updated,
    Skipped,
}

impl PersistOutcome {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Inserted => "inserted",
            Self::Updated => "updated",
            Self::Skipped => "skipped",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestClassificationReport {
    pub by_route: Vec<LabelCount>,
    pub top_risk_flags: Vec<LabelCount>,
    pub unknown_label_counts: Vec<LabelCount>,
    pub repeated_guardrail_sessions: Vec<SessionRouteCount>,
    /// Contested intent decisions (small weight margin between winner and
    /// runner-up). These are the hand-labeling candidates for the golden
    /// corpus: classifier drift shows up here first.
    pub low_margin_intents: Vec<LowMarginIntent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LowMarginIntent {
    pub event_id: String,
    pub intent: String,
    pub runner_up: Option<String>,
    pub margin: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelCount {
    pub label: String,
    pub count: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionRouteCount {
    pub session_id: String,
    pub count: i64,
}

#[derive(Debug, Clone)]
pub struct ReportOptions {
    pub repo: Option<String>,
    pub since: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LivePolicyConfig {
    pub enabled: bool,
    pub policy_version: String,
}

impl Default for LivePolicyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            policy_version: "v1".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LivePolicyDecision {
    pub action: &'static str,
    pub reason: &'static str,
    pub route: RecommendedRoute,
    pub response_contract: ResponseContract,
}
