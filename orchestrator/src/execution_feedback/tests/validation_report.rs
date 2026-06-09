use super::*;

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
