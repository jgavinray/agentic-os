use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use crate::state_context_policy::{
    ContextPolicy, TaskCategory, TaskContextConfig, CONTEXT_CACHE_TTL_MS,
    DEFAULT_CONTEXT_DECAY_RATE, DEFAULT_FAILURE_HISTORY_TOKEN_BUDGET,
    DEFAULT_OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryLevel {
    L0 = 0,
    L1 = 1,
    L2 = 2,
    L3 = 3,
}

impl MemoryLevel {
    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum MemoryScope {
    Session,
    Repo,
    Project,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum PromotionReason {
    AutomaticSessionSummary,
    RepeatedPattern,
    ReferencedAgain,
    MarkedImportant,
    FailedAttemptRemediated,
    StableOverTime,
    ManualConfirmation,
}

#[derive(Debug, Serialize, Clone)]
pub struct EventMemory {
    pub event_type: String,
    pub summary: String,
    pub evidence: Option<String>,
    pub metadata: Value,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub summary_level: i32,
}
