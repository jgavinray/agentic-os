//! Deterministic request classification taxonomy.
//!
//! This module defines the bounded labels and table-facing structs for the
//! pre-LLM request classification layer. Feature extraction, backfill, and live
//! routing are later phases.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub const CLASSIFICATION_SCHEMA_VERSION: i32 = 1;
pub const ROUTING_POLICY_VERSION: &str = "deterministic-v1";
pub const CLASSIFIER_SOURCE_DETERMINISTIC_RULES: &str = "deterministic_rules";

macro_rules! request_classification_enums {
    (
        $(
            $(#[$enum_meta:meta])*
            pub enum $name:ident {
                $($variant:ident => $label:literal),* $(,)?
            }
        )*
    ) => {
        $(
            $(#[$enum_meta])*
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
            pub enum $name {
                $($variant,)*
                #[default]
                Unknown,
            }

            impl $name {
                pub const fn as_str(self) -> &'static str {
                    match self {
                        $(Self::$variant => $label,)*
                        Self::Unknown => "unknown",
                    }
                }

                pub fn from_label(value: &str) -> Self {
                    match value {
                        $($label => Self::$variant,)*
                        "unknown" => Self::Unknown,
                        _ => Self::Unknown,
                    }
                }
            }
        )*

        pub fn enum_inventory() -> &'static [(&'static str, &'static [&'static str])] {
            &[
                $(
                    (
                        stringify!($name),
                        &[$($label,)* "unknown"],
                    ),
                )*
            ]
        }
    };
}

request_classification_enums! {
    pub enum RequestIntent {
        Explain => "explain",
        Debug => "debug",
        GenerateConfig => "generate_config",
        ModifyConfig => "modify_config",
        Summarize => "summarize",
        Classify => "classify",
        Search => "search",
        Plan => "plan",
        OperateTool => "operate_tool",
    }

    pub enum RequestDomain {
        Shell => "shell",
        Kubernetes => "kubernetes",
        LlmInference => "llm_inference",
        Docker => "docker",
        Networking => "networking",
        Security => "security",
        Medical => "medical",
        Legal => "legal",
        Finance => "finance",
        Generic => "generic",
    }

    pub enum RequestArtifactType {
        PlainText => "plain_text",
        Code => "code",
        Logs => "logs",
        Yaml => "yaml",
        Json => "json",
        Sql => "sql",
        Markdown => "markdown",
        Image => "image",
        File => "file",
    }

    pub enum RequestComplexity {
        L0Trivial => "l0_trivial",
        L1Simple => "l1_simple",
        L2Moderate => "l2_moderate",
        L3Complex => "l3_complex",
        L4ToolRequired => "l4_tool_required",
        L5HighRisk => "l5_high_risk",
    }

    pub enum RequestRisk {
        None => "none",
        SecretPresent => "secret_present",
        DestructiveCommand => "destructive_command",
        ExternalCurrentInfoRequired => "external_current_info_required",
        HighStakes => "high_stakes",
        PromptInjection => "prompt_injection",
        UnsafeSecurity => "unsafe_security",
    }

    pub enum RecommendedRoute {
        DeterministicTemplate => "deterministic_template",
        SmallLocalModel => "small_local_model",
        StrongLocalModel => "strong_local_model",
        WebRequired => "web_required",
        ToolRequired => "tool_required",
        AskClarification => "ask_clarification",
        RefuseOrGuardrail => "refuse_or_guardrail",
    }

    pub enum ResponseContract {
        DirectAnswer => "direct_answer",
        StructuredJson => "structured_json",
        MarkdownSummary => "markdown_summary",
        PatchRequired => "patch_required",
        ValidationRequired => "validation_required",
        ClarificationQuestion => "clarification_question",
        Refusal => "refusal",
    }
}

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
            classified_at: Utc::now(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const MIGRATION: &str = include_str!("../migrations/V9__request_classifications.sql");

    #[test]
    fn enum_inventory_contains_unknown_for_every_enum() {
        for (name, variants) in enum_inventory() {
            assert!(
                variants.contains(&"unknown"),
                "{name} must include unknown variant"
            );
        }
    }

    #[test]
    fn unknown_labels_map_to_unknown() {
        assert_eq!(
            RequestIntent::from_label("not-a-real-intent"),
            RequestIntent::Unknown
        );
        assert_eq!(
            RequestDomain::from_label("not-a-real-domain"),
            RequestDomain::Unknown
        );
        assert_eq!(
            RequestArtifactType::from_label("not-a-real-artifact"),
            RequestArtifactType::Unknown
        );
        assert_eq!(
            RequestComplexity::from_label("not-a-real-complexity"),
            RequestComplexity::Unknown
        );
        assert_eq!(
            RequestRisk::from_label("not-a-real-risk"),
            RequestRisk::Unknown
        );
        assert_eq!(
            RecommendedRoute::from_label("not-a-real-route"),
            RecommendedRoute::Unknown
        );
        assert_eq!(
            ResponseContract::from_label("not-a-real-contract"),
            ResponseContract::Unknown
        );
    }

    #[test]
    fn migration_has_checks_for_all_closed_labels() {
        for (_name, variants) in enum_inventory() {
            for variant in *variants {
                assert!(
                    MIGRATION.contains(&format!("'{variant}'")),
                    "migration is missing enum label {variant}"
                );
            }
        }
        assert!(MIGRATION.contains("CHECK (intent IN"));
        assert!(MIGRATION.contains("CHECK (domain IN"));
        assert!(MIGRATION.contains("CHECK (artifact_type IN"));
        assert!(MIGRATION.contains("CHECK (complexity IN"));
        assert!(MIGRATION.contains("CHECK (recommended_route IN"));
        assert!(MIGRATION.contains("CHECK (response_contract IN"));
        assert!(MIGRATION.contains("CHECK (risk <@ ARRAY"));
        assert!(MIGRATION.contains("CHECK (secondary_domains <@ ARRAY"));
    }

    #[test]
    fn schema_privacy_allows_only_documented_unbounded_text_columns() {
        let allowed_unbounded_strings: HashSet<&str> = [
            "event_id",
            "repo",
            "session_id",
            "routing_policy_version",
            "classifier_source",
        ]
        .into_iter()
        .collect();
        let text_columns = [
            "event_id",
            "repo",
            "session_id",
            "routing_policy_version",
            "classifier_source",
        ];

        for column in text_columns {
            assert!(
                allowed_unbounded_strings.contains(column),
                "{column} must be explicitly privacy-allowlisted"
            );
        }
        assert!(!MIGRATION.contains("summary TEXT"));
        assert!(!MIGRATION.contains("evidence TEXT"));
        assert!(!MIGRATION.contains("request TEXT"));
        assert!(!MIGRATION.contains("prompt TEXT"));
    }

    #[test]
    fn deterministic_default_row_uses_current_versions() {
        let now = Utc::now();
        let row = RequestClassification::deterministic(
            "event-1".to_string(),
            "repo".to_string(),
            "session".to_string(),
            now,
        );

        assert_eq!(
            row.classification_schema_version,
            CLASSIFICATION_SCHEMA_VERSION
        );
        assert_eq!(row.routing_policy_version, ROUTING_POLICY_VERSION);
        assert_eq!(row.classifier_source, CLASSIFIER_SOURCE_DETERMINISTIC_RULES);
        assert_eq!(row.intent, RequestIntent::Unknown);
        assert_eq!(row.recommended_route, RecommendedRoute::Unknown);
    }
}
