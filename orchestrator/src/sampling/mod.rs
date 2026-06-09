use serde_json::Value;
use std::env;

pub mod audit;

pub use audit::{capture_with_override, capture_without_override, SamplingAudit, SamplingParams};

pub const REQUEST_EVENT_SCHEMA_VERSION: u64 = 1;

pub const PARAM_TEMPERATURE: &str = "temperature";
pub const PARAM_TOP_P: &str = "top_p";
pub const PARAM_MAX_TOKENS: &str = "max_tokens";
pub const PARAM_SEED: &str = "seed";
pub const PARAM_NONE: &str = "none";

pub const REASON_OVERRIDDEN_BY_ORCHESTRATOR: &str = "overridden_by_orchestrator";
pub const REASON_NOOP: &str = "noop";

const PARAMETER_NAMES: [&str; 4] = [PARAM_TEMPERATURE, PARAM_TOP_P, PARAM_MAX_TOKENS, PARAM_SEED];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingConfig {
    pub capture_enabled: bool,
    pub override_enabled: bool,
}

impl SamplingConfig {
    pub fn from_env() -> Result<Self, anyhow::Error> {
        Self::new(
            env_bool("SAMPLING_CAPTURE_ENABLED", true),
            env_bool("SAMPLING_OVERRIDE_ENABLED", false),
        )
    }

    pub fn new(capture_enabled: bool, override_enabled: bool) -> Result<Self, anyhow::Error> {
        if override_enabled && !capture_enabled {
            anyhow::bail!(
                "invalid sampling configuration: SAMPLING_OVERRIDE_ENABLED=true requires \
                 SAMPLING_CAPTURE_ENABLED=true; override without audit capture is unsafe"
            );
        }
        Ok(Self {
            capture_enabled,
            override_enabled,
        })
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(default)
}

pub trait SamplingPolicy: Send + Sync {
    fn apply(&self, request: &mut Value);
}

#[derive(Debug, Default)]
pub struct NoOpSamplingPolicy;

impl SamplingPolicy for NoOpSamplingPolicy {
    fn apply(&self, _request: &mut Value) {}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplingOverrideChange {
    pub parameter: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplingOverrideOutcome {
    Noop,
    Overridden(Vec<SamplingOverrideChange>),
}

pub fn apply_sampling_policy(
    request: &mut Value,
    policy: &dyn SamplingPolicy,
) -> SamplingOverrideOutcome {
    let before = SamplingParams::from_request(request);
    policy.apply(request);
    let after = SamplingParams::from_request(request);

    let changes: Vec<SamplingOverrideChange> = PARAMETER_NAMES
        .iter()
        .copied()
        .filter(|parameter| before.get(parameter) != after.get(parameter))
        .map(|parameter| SamplingOverrideChange { parameter })
        .collect();

    if changes.is_empty() {
        crate::telemetry::record_sampling_param_override(PARAM_NONE, REASON_NOOP);
        SamplingOverrideOutcome::Noop
    } else {
        for change in &changes {
            crate::telemetry::record_sampling_param_override(
                change.parameter,
                REASON_OVERRIDDEN_BY_ORCHESTRATOR,
            );
        }
        SamplingOverrideOutcome::Overridden(changes)
    }
}

pub fn capture_and_maybe_override(
    original_request: &Value,
    forwarded_request: &mut Value,
    config: SamplingConfig,
    policy: &dyn SamplingPolicy,
) -> Option<SamplingAudit> {
    if !config.capture_enabled {
        return None;
    }

    let requested = SamplingParams::from_request(original_request);
    if !config.override_enabled {
        return Some(capture_without_override(requested));
    }

    apply_sampling_policy(forwarded_request, policy);
    Some(capture_with_override(original_request, forwarded_request))
}

#[cfg(test)]
mod tests;
