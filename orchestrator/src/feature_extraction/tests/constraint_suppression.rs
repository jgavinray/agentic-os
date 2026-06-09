use super::*;

#[test]
fn stale_detection_suppresses_by_freshness_window() {
    let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
    let old = now - Duration::seconds(4000);
    let events = vec![event(
        "e1",
        "failed_attempt",
        json!({
            "detection_tags": [{"type": "summarization_failure", "source": "summarizer", "tag_schema_version": 1}]
        }),
        old,
    )];

    let report = extract_records(&events, &config(now));
    let record = &report.records[0];
    assert_eq!(record.summarization_failure_count, 1);
    assert!(record.recommended_constraints.is_empty());
    assert!(record.suppressed_constraints.iter().any(|item| {
        item.constraint_type == "handle_summarization_failure" && item.reason == "freshness_window"
    }));
}

#[test]
fn fresh_recovery_suppresses_then_new_detection_resets() {
    let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
    let detection = event(
        "e1",
        "failed_attempt",
        json!({
            "detection_tags": [{"type": "tool_loop", "tool": "Bash", "source": "tool_loop_detector", "tag_schema_version": 1}]
        }),
        now,
    );
    let recovery = event(
        "e2",
        crate::execution_feedback::EVENT_TYPE_TOOL_RESULT,
        json!({"success": true, "payload": {"tool_name": "Bash"}}),
        now + Duration::seconds(10),
    );

    let recovered = extract_records(
        &[detection.clone(), recovery.clone()],
        &config(now + Duration::seconds(20)),
    );
    assert!(recovered.records[0].recommended_constraints.is_empty());
    assert!(recovered.records[0]
        .suppressed_constraints
        .iter()
        .any(
            |item| item.constraint_type == "avoid_tool_loop" && item.reason == "recovery_detected"
        ));

    let new_detection = event(
        "e3",
        "failed_attempt",
        json!({
            "detection_tags": [{"type": "tool_loop", "tool": "Bash", "source": "tool_loop_detector", "tag_schema_version": 1}]
        }),
        now + Duration::seconds(30),
    );
    let reset = extract_records(
        &[detection, recovery, new_detection],
        &config(now + Duration::seconds(40)),
    );
    assert!(reset.records[0]
        .recommended_constraints
        .iter()
        .any(|item| item.constraint_type == "avoid_tool_loop"));
}

#[test]
fn stale_recovery_does_not_suppress_fresh_failure() {
    let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
    let stale_recovery = event(
        "e1",
        crate::execution_feedback::EVENT_TYPE_TOOL_RESULT,
        json!({"success": true, "payload": {"tool_name": "Read"}}),
        now - Duration::seconds(4000),
    );
    let detection = event(
        "e2",
        "failed_attempt",
        json!({
            "detection_tags": [{"type": "tool_loop", "tool": "Read", "source": "tool_loop_detector", "tag_schema_version": 1}]
        }),
        now,
    );

    let report = extract_records(&[stale_recovery, detection], &config(now));
    let record = report
        .records
        .iter()
        .find(|record| record.tool_loop_count == 1)
        .unwrap();
    assert!(record
        .recommended_constraints
        .iter()
        .any(|item| item.constraint_type == "avoid_tool_loop"));
}
