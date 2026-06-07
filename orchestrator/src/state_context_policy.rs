use serde::{Deserialize, Serialize};
use std::str::FromStr;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskCategory {
    Narrow,
    Moderate,
    Broad,
    Architecture,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TaskContextConfig {
    pub max_events: i64,
    pub semantic_limit: usize,
    pub char_budget: usize,
}

impl FromStr for TaskCategory {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "narrow" => Ok(TaskCategory::Narrow),
            "moderate" => Ok(TaskCategory::Moderate),
            "broad" => Ok(TaskCategory::Broad),
            "architecture" => Ok(TaskCategory::Architecture),
            _ => Err(format!("unknown task category: {s}")),
        }
    }
}

impl TaskContextConfig {
    pub fn for_category(cat: TaskCategory) -> Self {
        use TaskCategory::*;
        let (max_events, semantic_limit, char_budget) = match cat {
            Narrow => (3i64, 3usize, 3000usize),
            Moderate => (8i64, 5usize, 5000usize),
            Broad => (15i64, 10usize, 7000usize),
            Architecture => (20i64, 12usize, 10000usize),
        };
        Self {
            max_events,
            semantic_limit,
            char_budget,
        }
    }
}

impl TaskCategory {
    pub const ARCHITECTURE_KEYWORDS: &'static [&'static str] = &[
        "architecture",
        "architectural",
        "design",
        "invariant",
        "durable",
        "project truth",
        "system design",
    ];
    pub const NARROW_KEYWORDS: &'static [&'static str] = &[
        "fix", "bug", "debug", "error", "issue", "warn", "patch", "hotfix",
    ];
    pub const BROAD_KEYWORDS: &'static [&'static str] = &[
        "migrate",
        "rewrite",
        "refactor",
        "redesign",
        "deploy",
        "infrastructure",
        "setup",
        "configure",
    ];

    pub fn from_task(task: &str) -> Self {
        let task_lower = task.to_lowercase();
        for kw in Self::ARCHITECTURE_KEYWORDS {
            if task_lower.contains(kw) {
                return Self::Architecture;
            }
        }
        for kw in Self::BROAD_KEYWORDS {
            if task_lower.contains(kw) {
                return Self::Broad;
            }
        }
        for kw in Self::NARROW_KEYWORDS {
            if task_lower.contains(kw) {
                return Self::Narrow;
            }
        }
        Self::Moderate
    }
}

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
