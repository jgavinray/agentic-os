use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Default)]
struct ForceTemperaturePolicy;

impl SamplingPolicy for ForceTemperaturePolicy {
    fn apply(&self, request: &mut Value) {
        request[PARAM_TEMPERATURE] = json!(0.1);
    }
}

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

    assert_eq!(forwarded[PARAM_TEMPERATURE], json!(0.1));
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

    assert_eq!(request[PARAM_TEMPERATURE], json!(0.3));
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
