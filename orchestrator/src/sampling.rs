use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: Value,
    pub top_p: Value,
    pub max_tokens: Value,
    pub seed: Value,
}

impl SamplingParams {
    pub fn from_request(request: &Value) -> Self {
        Self {
            temperature: request
                .get(PARAM_TEMPERATURE)
                .cloned()
                .unwrap_or(Value::Null),
            top_p: request.get(PARAM_TOP_P).cloned().unwrap_or(Value::Null),
            max_tokens: request
                .get(PARAM_MAX_TOKENS)
                .cloned()
                .unwrap_or(Value::Null),
            seed: request.get(PARAM_SEED).cloned().unwrap_or(Value::Null),
        }
    }

    fn get(&self, parameter: &str) -> &Value {
        match parameter {
            PARAM_TEMPERATURE => &self.temperature,
            PARAM_TOP_P => &self.top_p,
            PARAM_MAX_TOKENS => &self.max_tokens,
            PARAM_SEED => &self.seed,
            _ => &Value::Null,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingAudit {
    pub sampling_params: SamplingParams,
    pub forwarded_sampling_params: SamplingParams,
}

impl SamplingAudit {
    pub fn metadata(&self, endpoint: &str, requested_model: &str, routed_model: &str) -> Value {
        json!({
            "event_schema_version": REQUEST_EVENT_SCHEMA_VERSION,
            "endpoint": endpoint,
            "requested_model": requested_model,
            "routed_model": routed_model,
            "sampling_params": self.sampling_params,
            "forwarded_sampling_params": self.forwarded_sampling_params,
        })
    }
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

pub fn capture_without_override(requested: SamplingParams) -> SamplingAudit {
    SamplingAudit {
        sampling_params: requested.clone(),
        forwarded_sampling_params: requested,
    }
}

pub fn capture_with_override(original_request: &Value, forwarded_request: &Value) -> SamplingAudit {
    SamplingAudit {
        sampling_params: SamplingParams::from_request(original_request),
        forwarded_sampling_params: SamplingParams::from_request(forwarded_request),
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
#[derive(Debug, Default)]
pub struct ForceTemperaturePolicy;

#[cfg(test)]
impl SamplingPolicy for ForceTemperaturePolicy {
    fn apply(&self, request: &mut Value) {
        request[PARAM_TEMPERATURE] = json!(0.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct CountingPolicy {
        count: AtomicUsize,
    }

    impl SamplingPolicy for CountingPolicy {
        fn apply(&self, _request: &mut Value) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn sampling_params_include_all_fields_with_nulls_for_missing_values() {
        let params = SamplingParams::from_request(&json!({
            "messages": [],
            "temperature": 0.3
        }));

        assert_eq!(params.temperature, json!(0.3));
        assert!(params.top_p.is_null());
        assert!(params.max_tokens.is_null());
        assert!(params.seed.is_null());
    }

    #[test]
    fn capture_without_override_copies_forwarded_params() {
        let requested = SamplingParams::from_request(&json!({
            "temperature": 0.3,
            "top_p": 0.9
        }));
        let audit = capture_without_override(requested);

        assert_eq!(audit.sampling_params, audit.forwarded_sampling_params);
    }

    #[test]
    fn audit_metadata_has_stable_request_event_shape() {
        let requested = SamplingParams::from_request(&json!({"temperature": 0.3}));
        let audit = capture_without_override(requested);

        let metadata = audit.metadata("/v1/chat/completions", "client-model", "routed-model");

        assert_eq!(
            metadata["event_schema_version"],
            REQUEST_EVENT_SCHEMA_VERSION
        );
        assert_eq!(metadata["requested_model"], "client-model");
        assert_eq!(metadata["routed_model"], "routed-model");
        assert_eq!(metadata["sampling_params"]["temperature"], json!(0.3));
        assert_eq!(
            metadata["forwarded_sampling_params"],
            metadata["sampling_params"]
        );
        for key in [PARAM_TEMPERATURE, PARAM_TOP_P, PARAM_MAX_TOKENS, PARAM_SEED] {
            assert!(metadata["sampling_params"].get(key).is_some());
            assert!(metadata["forwarded_sampling_params"].get(key).is_some());
        }
    }

    #[test]
    fn force_temperature_policy_changes_forwarded_request() {
        let mut request = json!({
            "messages": [],
            "temperature": 0.7
        });

        let outcome = apply_sampling_policy(&mut request, &ForceTemperaturePolicy);

        assert_eq!(request["temperature"], json!(0.1));
        assert_eq!(
            outcome,
            SamplingOverrideOutcome::Overridden(vec![SamplingOverrideChange {
                parameter: PARAM_TEMPERATURE
            }])
        );
    }

    #[test]
    fn forced_temperature_policy_is_reflected_in_forwarded_audit() {
        let original = json!({"temperature": 0.7});
        let mut forwarded = original.clone();

        let audit = capture_and_maybe_override(
            &original,
            &mut forwarded,
            SamplingConfig::new(true, true).unwrap(),
            &ForceTemperaturePolicy,
        )
        .unwrap();

        assert_eq!(forwarded["temperature"], json!(0.1));
        assert_eq!(audit.sampling_params.temperature, json!(0.7));
        assert_eq!(audit.forwarded_sampling_params.temperature, json!(0.1));
    }

    #[test]
    fn noop_policy_leaves_params_unchanged() {
        let mut request = json!({
            "messages": [],
            "temperature": 0.3
        });

        let outcome = apply_sampling_policy(&mut request, &NoOpSamplingPolicy);

        assert_eq!(request["temperature"], json!(0.3));
        assert_eq!(outcome, SamplingOverrideOutcome::Noop);
    }

    #[test]
    fn capture_disabled_bypasses_policy() {
        let policy = CountingPolicy::default();
        let mut request = json!({"temperature": 0.3});
        let original = request.clone();

        let audit = capture_and_maybe_override(
            &original,
            &mut request,
            SamplingConfig::new(false, false).unwrap(),
            &policy,
        );

        assert!(audit.is_none());
        assert_eq!(policy.count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn override_enabled_invokes_policy_once() {
        let policy = CountingPolicy::default();
        let mut request = json!({"temperature": 0.3});
        let original = request.clone();

        let audit = capture_and_maybe_override(
            &original,
            &mut request,
            SamplingConfig::new(true, true).unwrap(),
            &policy,
        );

        assert!(audit.is_some());
        assert_eq!(policy.count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn override_requires_capture() {
        let err = SamplingConfig::new(false, true).unwrap_err().to_string();

        assert!(err.contains("SAMPLING_OVERRIDE_ENABLED=true"));
        assert!(err.contains("SAMPLING_CAPTURE_ENABLED=true"));
        assert!(err.contains("override without audit capture is unsafe"));
    }

    #[test]
    fn flag_defaults_enable_capture_and_disable_override() {
        let config = SamplingConfig::new(true, false).unwrap();

        assert!(config.capture_enabled);
        assert!(!config.override_enabled);
    }

    #[test]
    fn sampling_capture_p99_under_one_ms() {
        let original = json!({
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 4096,
            "seed": 123
        });
        let mut durations = Vec::new();

        for _ in 0..1000 {
            let started = std::time::Instant::now();
            let mut forwarded = original.clone();
            let audit = capture_and_maybe_override(
                &original,
                &mut forwarded,
                SamplingConfig::new(true, false).unwrap(),
                &NoOpSamplingPolicy,
            );
            assert!(audit.is_some());
            durations.push(started.elapsed());
        }

        durations.sort();
        let p99 = durations[(durations.len() * 99) / 100 - 1];
        assert!(p99 < std::time::Duration::from_millis(1), "p99 was {p99:?}");
    }

    #[test]
    fn noop_override_p99_under_one_ms() {
        let original = json!({
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 4096,
            "seed": 123
        });
        let mut durations = Vec::new();

        for _ in 0..1000 {
            let started = std::time::Instant::now();
            let mut forwarded = original.clone();
            let audit = capture_and_maybe_override(
                &original,
                &mut forwarded,
                SamplingConfig::new(true, true).unwrap(),
                &NoOpSamplingPolicy,
            );
            assert!(audit.is_some());
            durations.push(started.elapsed());
        }

        durations.sort();
        let p99 = durations[(durations.len() * 99) / 100 - 1];
        assert!(p99 < std::time::Duration::from_millis(1), "p99 was {p99:?}");
    }
}
