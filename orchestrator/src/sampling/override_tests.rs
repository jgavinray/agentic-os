use super::*;

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
