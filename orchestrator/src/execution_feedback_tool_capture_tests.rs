use super::*;

#[test]
fn tool_result_parser_finds_synthetic_block() {
    let value = json!({
        "content": [{
            "type": "tool_result",
            "tool_name": "pytest",
            "exit_code": 1,
            "duration_ms": 12,
            "content": "1 failed, 2 passed in 0.3s"
        }]
    });
    let results = tool_results_from_value(&value);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].tool_name, "pytest");
    assert_eq!(results[0].exit_code, 1);
}

#[test]
fn sse_tool_result_parser_finds_synthetic_block() {
    let raw = concat!(
        "data: {\"content\":[{\"type\":\"tool_result\",\"tool_name\":\"cargo test\",\"content\":\"test result: FAILED. 2 passed; 1 failed; 0 ignored\",\"exit_code\":101}]}\n",
        "data: [DONE]\n"
    );
    let results = tool_results_from_sse(raw);
    assert_eq!(results.len(), 1);
    let events = events_for_tool_result(&ctx(), &results[0]);
    assert!(events
        .iter()
        .any(|event| event.event_type == EVENT_TYPE_TEST_RESULT));
    assert!(!events
        .iter()
        .any(|event| event.event_type == "failure_signature"));
    let test_event = events
        .iter()
        .find(|event| event.event_type == EVENT_TYPE_TEST_RESULT)
        .unwrap();
    assert_eq!(test_event.metadata["payload"]["signature"], "unknown");
    assert_eq!(
        test_event.metadata["payload"]["signature_category"],
        "unknown"
    );
    assert_eq!(
        test_event.metadata["payload"]["fingerprint_version"],
        FINGERPRINT_VERSION
    );
}

#[test]
fn validation_parser_recognizes_initial_validator_list() {
    for tool in [
        "cargo",
        "pytest",
        "npm test",
        "eslint",
        "tsc",
        "mypy",
        "ruff",
        "terraform",
        "kubectl",
    ] {
        assert!(classify_validator(tool, "").is_some(), "{tool}");
    }
    assert!(classify_validator("printf", "").is_none());
}

#[test]
fn unrecognized_tool_only_emits_generic_tool_result() {
    let result = CapturedToolResult {
        tool_name: "printf".to_string(),
        content: "hello".to_string(),
        exit_code: 0,
        duration_ms: 1,
        stdout_summary: "hello".to_string(),
        stderr_summary: String::new(),
    };
    let events = events_for_tool_result(&ctx(), &result);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event_type, EVENT_TYPE_TOOL_RESULT);
}
