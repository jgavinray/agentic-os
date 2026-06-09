use crate::feature_extraction_types::{
    DEFAULT_CONSTRAINT_FRESHNESS_WINDOW_SEC, DEFAULT_FEATURE_WINDOW_SEC,
    DEFAULT_MAX_OPERATIONAL_CONSTRAINTS, DEFAULT_OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET,
};

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
