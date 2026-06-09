use crate::request_classification_types::LivePolicyConfig;

pub fn live_policy_config_from_env() -> LivePolicyConfig {
    LivePolicyConfig {
        enabled: env_flag_enabled("REQUEST_CLASSIFICATION_LIVE_POLICY_ENABLED", false),
        policy_version: std::env::var("REQUEST_CLASSIFICATION_POLICY_VERSION")
            .unwrap_or_else(|_| "v1".to_string()),
    }
}

pub fn request_classification_startup_backfill_enabled() -> bool {
    env_flag_enabled("REQUEST_CLASSIFICATION_STARTUP_BACKFILL_ENABLED", true)
}

pub fn request_classification_startup_batch_size() -> i64 {
    std::env::var("REQUEST_CLASSIFICATION_STARTUP_BACKFILL_BATCH_SIZE")
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(500)
}

fn env_flag_enabled(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(default)
}
