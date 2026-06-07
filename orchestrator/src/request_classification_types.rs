use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub const CLASSIFICATION_SCHEMA_VERSION: i32 = 2;
pub const ROUTING_POLICY_VERSION: &str = "deterministic-v1";
pub const CLASSIFIER_SOURCE_DETERMINISTIC_RULES: &str = "deterministic_rules";

pub const FEATURE_KEYS: &[&str] = &[
    "char_count",
    "line_count",
    "estimated_tokens",
    "has_code_block",
    "has_yaml",
    "has_json",
    "has_stack_trace",
    "has_logs",
    "has_shell_command",
    "has_url",
    "has_file_path",
    "has_secret_candidate",
    "contains_error_words",
    "contains_destructive_verbs",
    "asks_for_latest",
    "asks_for_file_generation",
    "detected_domain_terms",
    "has_kubernetes_terms",
    "has_docker_terms",
    "has_llm_terms",
    "has_networking_terms",
    "has_security_terms",
    "has_config_shape",
    "has_diff_or_patch",
    "has_test_failure",
    "is_composite",
    "decomposition_candidate",
    "decomposition_reason",
    "sub_intent_count",
    "sub_intents",
];

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
        Implement => "implement",
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

pub fn live_policy_config_from_env() -> LivePolicyConfig {
    LivePolicyConfig {
        enabled: std::env::var("REQUEST_CLASSIFICATION_LIVE_POLICY_ENABLED")
            .map(|value| {
                !matches!(
                    value.to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off"
                )
            })
            .unwrap_or(false),
        policy_version: std::env::var("REQUEST_CLASSIFICATION_POLICY_VERSION")
            .unwrap_or_else(|_| "v1".to_string()),
    }
}

pub fn request_classification_startup_backfill_enabled() -> bool {
    std::env::var("REQUEST_CLASSIFICATION_STARTUP_BACKFILL_ENABLED")
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
}

pub fn request_classification_startup_batch_size() -> i64 {
    std::env::var("REQUEST_CLASSIFICATION_STARTUP_BACKFILL_BATCH_SIZE")
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(500)
}
