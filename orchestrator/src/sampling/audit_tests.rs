use super::*;

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
