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
fn patch_payload_includes_lineage_metadata_fields() {
    let payload = patch_result_payload(vec!["src/lib.rs".to_string()], "applied", vec![]);
    assert_eq!(payload["files_touched"][0], "src/lib.rs");
    assert_eq!(payload["lines_added"], 0);
    assert_eq!(payload["lines_removed"], 0);
    assert_eq!(payload["patch_applied"], true);
    assert_eq!(payload["patch_reverted"], false);
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
