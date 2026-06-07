use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const TAG_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_FEATURE_WINDOW_SEC: i64 = 3600;
pub const DEFAULT_CONSTRAINT_FRESHNESS_WINDOW_SEC: i64 = 1800;
pub const DEFAULT_MAX_OPERATIONAL_CONSTRAINTS: usize = 5;
pub const DEFAULT_OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET: usize = 300;

pub const DETECTION_TAG_TYPES: [&str; 15] = [
    "tool_loop",
    "user_interruption",
    "missing_auth",
    "wrong_endpoint",
    "summarization_failure",
    "migration_failure",
    "correction_acknowledged",
    "context_pack_empty",
    "context_pack_truncated",
    "high_input_tokens",
    "slow_upstream_model",
    "empty_tool_use_message",
    "abandoned_before_model",
    "single_model_abandoned_no_tools",
    "summarizer_shared_upstream",
];

pub const DETECTION_TAG_SOURCES: [&str; 11] = [
    "hook_parser",
    "tool_loop_detector",
    "validation_parser",
    "failed_attempt_classifier",
    "summarizer",
    "remediation_parser",
    "context_builder",
    "model_response_parser",
    "trajectory_analyzer",
    "config_validator",
    "bootstrap_migration",
];

pub const FEATURE_FAILURE_CLASSES: [&str; 15] = [
    "tool_loop",
    "user_interruption",
    "missing_auth",
    "wrong_endpoint",
    "summarization_failure",
    "migration_failure",
    "context_pack_empty",
    "context_pack_truncated",
    "high_input_tokens",
    "slow_upstream_model",
    "empty_tool_use_message",
    "abandoned_before_model",
    "single_model_abandoned_no_tools",
    "summarizer_shared_upstream",
    "other",
];

pub const OPERATIONAL_CONSTRAINT_TYPES: [&str; 9] = [
    "use_known_auth",
    "use_known_endpoint",
    "use_known_migration_fix",
    "avoid_tool_loop",
    "fix_context_retrieval",
    "reduce_context_bloat",
    "separate_summarizer_upstream",
    "handle_user_interruption",
    "handle_summarization_failure",
];

pub const OPERATIONAL_SUPPRESSION_REASONS: [&str; 4] = [
    "freshness_window",
    "recovery_detected",
    "priority_cap",
    "token_budget",
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectionTag {
    #[serde(rename = "type")]
    pub tag_type: String,
    pub source: String,
    pub tag_schema_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
}

impl DetectionTag {
    pub fn new(tag_type: &str, source: &str) -> Self {
        Self {
            tag_type: tag_type.to_string(),
            source: source.to_string(),
            tag_schema_version: TAG_SCHEMA_VERSION,
            tool: None,
        }
    }

    pub fn tool_loop(source: &str, tool: &str) -> Self {
        Self {
            tag_type: "tool_loop".to_string(),
            source: source.to_string(),
            tag_schema_version: TAG_SCHEMA_VERSION,
            tool: Some(tool.to_string()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationalConstraint {
    pub constraint_type: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuppressedConstraint {
    pub constraint_type: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureRecord {
    pub feature_id: String,
    pub trajectory_id: Option<Uuid>,
    pub session_id: String,
    pub repo: String,
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,

    pub event_count: i64,
    pub user_message_count: i64,
    pub assistant_message_count: i64,
    pub failed_attempt_count: i64,
    pub summary_count: i64,

    pub tool_loop_count: i64,
    pub repeated_read_loop_count: i64,
    pub repeated_bash_loop_count: i64,
    pub user_interruption_count: i64,
    pub missing_auth_count: i64,
    pub wrong_endpoint_count: i64,
    pub summarization_failure_count: i64,
    pub migration_failure_count: i64,
    pub other_failure_count: i64,
    pub context_pack_empty_count: i64,
    pub context_pack_truncated_count: i64,
    pub high_input_token_count: i64,
    pub slow_upstream_model_count: i64,
    pub empty_tool_use_message_count: i64,
    pub abandoned_before_model_count: i64,
    pub single_model_abandoned_no_tools_count: i64,
    pub summarizer_shared_upstream_count: i64,

    pub known_endpoint: Option<String>,
    pub known_auth_header: Option<String>,
    pub known_migration_fix: Option<String>,

    pub loop_detected: bool,
    pub user_interrupted: bool,

    pub failure_classes: Vec<String>,
    pub recommended_constraints: Vec<OperationalConstraint>,
    pub suppressed_constraints: Vec<SuppressedConstraint>,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Clone, Debug)]
pub struct ExtractionConfig {
    pub feature_window_sec: i64,
    pub constraint_freshness_window_sec: i64,
    pub max_operational_constraints: usize,
    pub evaluation_time: DateTime<Utc>,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            feature_window_sec: DEFAULT_FEATURE_WINDOW_SEC,
            constraint_freshness_window_sec: DEFAULT_CONSTRAINT_FRESHNESS_WINDOW_SEC,
            max_operational_constraints: DEFAULT_MAX_OPERATIONAL_CONSTRAINTS,
            evaluation_time: Utc::now(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ExtractionReport {
    pub records: Vec<FeatureRecord>,
    pub unknown_tag_schema_versions: u64,
}

#[derive(Clone, Debug, Default)]
pub struct BackfillReport {
    pub events_scanned: usize,
    pub bootstrap_updated: i64,
    pub feature_records: usize,
    pub persisted_records: usize,
    pub dry_run: bool,
}

#[derive(Clone, Debug)]
pub struct BackfillOptions {
    pub repo: Option<String>,
    pub session_id: Option<String>,
    pub trajectory_id: Option<Uuid>,
    pub since: Option<DateTime<Utc>>,
    pub dry_run: bool,
    pub batch_size: i64,
    pub skip_bootstrap_tagging: bool,
}

#[derive(Clone, Debug)]
pub struct ExtractionScope {
    pub repo: Option<String>,
    pub session_id: Option<String>,
    pub trajectory_id: Option<Uuid>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
}

pub fn feature_extraction_enabled_from_env() -> bool {
    std::env::var("FEATURE_EXTRACTION_ENABLED")
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
}

pub fn feature_window_sec_from_env() -> i64 {
    positive_i64_env("FEATURE_WINDOW_SEC", DEFAULT_FEATURE_WINDOW_SEC)
}

pub fn constraint_freshness_window_sec_from_env() -> i64 {
    positive_i64_env(
        "CONSTRAINT_FRESHNESS_WINDOW_SEC",
        DEFAULT_CONSTRAINT_FRESHNESS_WINDOW_SEC,
    )
}

pub fn max_operational_constraints_from_env() -> usize {
    std::env::var("MAX_OPERATIONAL_CONSTRAINTS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAX_OPERATIONAL_CONSTRAINTS)
}

pub fn operational_constraints_token_budget_from_env() -> usize {
    std::env::var("OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET)
}

fn positive_i64_env(name: &str, default: i64) -> i64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}
