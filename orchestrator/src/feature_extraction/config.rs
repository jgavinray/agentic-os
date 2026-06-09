use chrono::{DateTime, Utc};
use uuid::Uuid;

pub use crate::feature_extraction_env::{
    constraint_freshness_window_sec_from_env, feature_extraction_enabled_from_env,
    feature_window_sec_from_env, max_operational_constraints_from_env,
    operational_constraints_token_budget_from_env,
};

pub const DEFAULT_FEATURE_WINDOW_SEC: i64 = 3600;
pub const DEFAULT_CONSTRAINT_FRESHNESS_WINDOW_SEC: i64 = 1800;
pub const DEFAULT_MAX_OPERATIONAL_CONSTRAINTS: usize = 5;
pub const DEFAULT_OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET: usize = 300;

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
    pub records: Vec<crate::feature_extraction_types::FeatureRecord>,
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
