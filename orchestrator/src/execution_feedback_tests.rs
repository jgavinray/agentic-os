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
fn event_types_are_defined_in_one_place() {
    assert_eq!(EXECUTION_EVENT_TYPES.len(), 7);
    assert!(EXECUTION_EVENT_TYPES.contains(&EVENT_TYPE_TOOL_RESULT));
    assert!(!EXECUTION_EVENT_TYPES.contains(&"failure_signature"));
}

#[test]
fn fingerprint_rules_cover_initial_classes() {
    let cases = [
        ("error[E0382]: use of moved value", "rust:borrow-checker"),
        ("error[E0308]: mismatched types", "rust:type-mismatch"),
        (
            "ModuleNotFoundError: No module named 'x'",
            "python:import-error",
        ),
        ("src/a.ts:1:2 - error TS2322", "typescript:TS2322"),
        ("JSONDecodeError: Expecting value", "json:parse-error"),
        ("process exited with code 2", "process:non-zero-exit"),
    ];
    for (input, expected) in cases {
        assert_eq!(fingerprint(input).signature, expected);
    }
}

#[test]
fn fingerprint_is_deterministic() {
    let input = "error[E0382]: borrow of moved value";
    assert_eq!(fingerprint(input), fingerprint(input));
}

#[test]
fn unknown_fingerprint_preserves_excerpt() {
    let fp = fingerprint("very strange failure");
    assert_eq!(fp.signature, "unknown");
    assert!(fp.raw_excerpt.contains("very strange failure"));
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
fn event_builder_writes_required_envelope_fields() {
    for event_type in EXECUTION_EVENT_TYPES {
        let kind = ExecutionEventKind::from_str(event_type).unwrap();
        let event = build_execution_event(&ctx(), kind, true, json!({"example": true}));
        assert_eq!(event.event_type, event_type);
        assert_eq!(event.metadata["event_type"], event_type);
        assert_eq!(event.metadata["success"], true);
        assert!(event.metadata["correlation_id"].is_string());
        assert_eq!(event.metadata["repo"], "repo");
        assert_eq!(event.metadata["task"], "task");
        assert!(event.metadata["payload"].is_object());
        if is_failure_outcome_event_type(event_type) {
            assert!(event.metadata["payload"]["signature"].is_null());
            assert!(event.metadata["payload"]["signature_category"].is_null());
            assert!(event.metadata["payload"]["fingerprint_version"].is_null());
        }
    }
}

#[test]
fn failed_outcome_events_carry_inline_signatures() {
    let payload = compile_result_payload("rust", "cargo", 101, "error[E0308]: mismatched types");
    let event = build_execution_event(&ctx(), ExecutionEventKind::CompileResult, false, payload);

    assert_eq!(event.metadata["payload"]["signature"], "rust:type-mismatch");
    assert_eq!(
        event.metadata["payload"]["signature_category"],
        "type_error"
    );
    assert_eq!(
        event.metadata["payload"]["fingerprint_version"],
        FINGERPRINT_VERSION
    );
}

#[test]
fn event_type_examples_write_and_retrieve_intact() {
    let ctx = ctx();
    let examples = [
        (
            ExecutionEventKind::ToolResult,
            true,
            tool_result_payload(&CapturedToolResult {
                tool_name: "cargo".to_string(),
                content: String::new(),
                exit_code: 0,
                duration_ms: 1,
                stdout_summary: "ok".to_string(),
                stderr_summary: String::new(),
            }),
        ),
        (
            ExecutionEventKind::CompileResult,
            true,
            compile_result_payload("rust", "cargo", 0, ""),
        ),
        (
            ExecutionEventKind::TestResult,
            true,
            test_result_payload("pytest", "2 passed"),
        ),
        (
            ExecutionEventKind::LintResult,
            true,
            lint_result_payload("ruff", "0 errors, 0 warnings"),
        ),
        (
            ExecutionEventKind::ValidationResult,
            true,
            validation_result_payload("schema", true, ""),
        ),
        (
            ExecutionEventKind::PatchResult,
            true,
            patch_result_payload(vec!["src/lib.rs".to_string()], "applied", vec![]),
        ),
        (
            ExecutionEventKind::Remediation,
            true,
            remediation_payload(Uuid::new_v4(), Uuid::new_v4(), "rust:type-mismatch"),
        ),
    ];

    let mut store = std::collections::BTreeMap::new();
    for (kind, success, payload) in examples {
        let event = build_execution_event(&ctx, kind, success, payload.clone());
        store.insert(event.id.clone(), event.clone());
        let retrieved = store.get(&event.id).unwrap();
        assert_eq!(retrieved.event_type, kind.as_str());
        assert_eq!(retrieved.metadata["payload"], payload);
        assert_eq!(retrieved.metadata["success"], success);
    }
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
fn patch_payload_includes_lineage_metadata_fields() {
    let payload = patch_result_payload(vec!["src/lib.rs".to_string()], "applied", vec![]);
    assert_eq!(payload["files_touched"][0], "src/lib.rs");
    assert_eq!(payload["lines_added"], 0);
    assert_eq!(payload["lines_removed"], 0);
    assert_eq!(payload["patch_applied"], true);
    assert_eq!(payload["patch_reverted"], false);
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

#[test]
fn patch_validation_references_resolve_inside_chain() {
    let validation = build_execution_event(
        &ctx(),
        ExecutionEventKind::ValidationResult,
        true,
        validation_result_payload("schema", true, ""),
    );
    let patch = build_execution_event(
        &ctx(),
        ExecutionEventKind::PatchResult,
        true,
        patch_result_payload(
            vec!["src/lib.rs".to_string()],
            "applied",
            vec![Uuid::parse_str(&validation.id).unwrap()],
        ),
    );
    assert!(patch_validation_ids_resolve(&[validation], &patch));
}
