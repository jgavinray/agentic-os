use super::*;

fn ctx() -> ExecutionEventContext {
    ExecutionEventContext {
        session_id: "session".to_string(),
        repo: "repo".to_string(),
        task: "task".to_string(),
        actor: "agent".to_string(),
        correlation_id: Uuid::new_v4(),
        parent_event_id: None,
        trajectory: None,
    }
}

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

#[test]
fn explicit_validation_report_matches_tool_result_event_shape() {
    let report = ValidationReportRequest {
        session_id: Some("session".to_string()),
        repo: "repo".to_string(),
        task: "task".to_string(),
        actor: None,
        event_type: None,
        validator_name: "pytest".to_string(),
        validator_type: Some("test".to_string()),
        success: Some(false),
        exit_code: Some(1),
        content: Some("1 failed, 2 passed".to_string()),
        payload: None,
        correlation_id: Some(ctx().correlation_id),
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
    };
    let events = events_for_validation_report(&ctx(), &report).unwrap();
    assert!(events
        .iter()
        .any(|event| event.event_type == EVENT_TYPE_TOOL_RESULT));
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
    assert_eq!(test_event.metadata["payload"]["validator_type"], "test");
    assert!(test_event.metadata["payload"]
        .get("failure_excerpt")
        .is_some());
}

#[test]
fn invalid_validator_type_is_rejected() {
    let report = ValidationReportRequest {
        session_id: Some("session".to_string()),
        repo: "repo".to_string(),
        task: "task".to_string(),
        actor: None,
        event_type: None,
        validator_name: "custom".to_string(),
        validator_type: Some("free_form".to_string()),
        success: Some(true),
        exit_code: Some(0),
        content: Some("ok".to_string()),
        payload: None,
        correlation_id: Some(ctx().correlation_id),
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
    };
    assert!(events_for_validation_report(&ctx(), &report).is_err());
}

#[test]
fn execution_capture_p99_under_one_ms() {
    let result = CapturedToolResult {
        tool_name: "cargo".to_string(),
        content: "error[E0308]: mismatched types\nexit code 101".to_string(),
        exit_code: 101,
        duration_ms: 10,
        stdout_summary: String::new(),
        stderr_summary: "error[E0308]: mismatched types".to_string(),
    };
    let ctx = ctx();
    let _ = events_for_tool_result(&ctx, &result);
    let mut durations = Vec::new();
    for _ in 0..1000 {
        let started = std::time::Instant::now();
        let events = events_for_tool_result(&ctx, &result);
        assert!(!events.is_empty());
        durations.push(started.elapsed());
    }
    durations.sort();
    let p99 = durations[(durations.len() * 99) / 100 - 1];
    let threshold = if cfg!(debug_assertions) {
        std::time::Duration::from_millis(10)
    } else {
        std::time::Duration::from_millis(1)
    };
    assert!(p99 < threshold, "p99 was {p99:?}");
}

#[test]
fn mixed_event_load_test_builds_1000_under_five_seconds() {
    let started = std::time::Instant::now();
    let ctx = ctx();
    let _ = fingerprint("error[E0308]: mismatched types");
    let mut count = 0usize;
    for idx in 0..250 {
        let result = CapturedToolResult {
            tool_name: if idx % 2 == 0 {
                "pytest".to_string()
            } else {
                "tsc".to_string()
            },
            content: if idx % 2 == 0 {
                "2 passed, 0 failed".to_string()
            } else {
                "src/a.ts(1,1): error TS2322: type mismatch".to_string()
            },
            exit_code: if idx % 2 == 0 { 0 } else { 2 },
            duration_ms: idx,
            stdout_summary: String::new(),
            stderr_summary: String::new(),
        };
        count += events_for_tool_result(&ctx, &result).len();
    }
    while count < 1000 {
        let _ = build_execution_event(
            &ctx,
            ExecutionEventKind::PatchResult,
            true,
            patch_result_payload(vec!["src/lib.rs".to_string()], "applied", vec![]),
        );
        count += 1;
    }
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
    assert!(count >= 1000);
}
