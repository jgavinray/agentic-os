use serde::{Deserialize, Serialize};
use std::str::FromStr;

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
