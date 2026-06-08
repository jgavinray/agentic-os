use serde::{Deserialize, Serialize};

pub use crate::state_task_category::TaskCategory;

/// Default TTL for cached context packs: 5 minutes.
pub const CONTEXT_CACHE_TTL_MS: u64 = 300_000;

/// Default exponential decay rate for search results.
///
/// The plan's acceptance criteria require a 1-hour-old result to score at most
/// 70% of an otherwise identical fresh result. A rate of 0.006 gives
/// e^(-0.006 * 60) ~= 0.698.
pub const DEFAULT_CONTEXT_DECAY_RATE: f64 = 0.006;

/// Default token budget for deterministic Failure History context.
pub const DEFAULT_FAILURE_HISTORY_TOKEN_BUDGET: usize = 1000;

/// Default token budget for compact Operational Constraints context.
pub const DEFAULT_OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET: usize =
    crate::feature_extraction::DEFAULT_OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ContextPolicy {
    pub task_category: TaskCategory,
    pub l0_recent_pct: u8,
    pub l1_matching_pct: u8,
    pub l2_repo_pct: u8,
    pub l3_project_pct: u8,
    pub failure_pct: u8,
    pub l0_recent_limit: i64,
    pub l1_limit: i64,
    pub l2_limit: i64,
    pub l3_limit: i64,
    pub failure_limit: i64,
}

impl ContextPolicy {
    pub fn for_task(task: &str) -> Self {
        Self::for_category(TaskCategory::from_task(task))
    }

    pub fn for_category(task_category: TaskCategory) -> Self {
        match task_category {
            TaskCategory::Narrow => Self {
                task_category,
                l0_recent_pct: 40,
                l1_matching_pct: 30,
                l2_repo_pct: 10,
                l3_project_pct: 0,
                failure_pct: 20,
                l0_recent_limit: 5,
                l1_limit: 4,
                l2_limit: 2,
                l3_limit: 0,
                failure_limit: 4,
            },
            TaskCategory::Moderate => Self {
                task_category,
                l0_recent_pct: 25,
                l1_matching_pct: 35,
                l2_repo_pct: 20,
                l3_project_pct: 5,
                failure_pct: 15,
                l0_recent_limit: 5,
                l1_limit: 6,
                l2_limit: 4,
                l3_limit: 2,
                failure_limit: 3,
            },
            TaskCategory::Broad => Self {
                task_category,
                l0_recent_pct: 10,
                l1_matching_pct: 35,
                l2_repo_pct: 35,
                l3_project_pct: 10,
                failure_pct: 10,
                l0_recent_limit: 3,
                l1_limit: 8,
                l2_limit: 8,
                l3_limit: 3,
                failure_limit: 3,
            },
            TaskCategory::Architecture => Self {
                task_category,
                l0_recent_pct: 5,
                l1_matching_pct: 20,
                l2_repo_pct: 35,
                l3_project_pct: 30,
                failure_pct: 10,
                l0_recent_limit: 2,
                l1_limit: 5,
                l2_limit: 8,
                l3_limit: 6,
                failure_limit: 3,
            },
        }
    }

    pub fn budget_for(&self, pct: u8, char_budget: usize) -> usize {
        (char_budget * pct as usize) / 100
    }
}
