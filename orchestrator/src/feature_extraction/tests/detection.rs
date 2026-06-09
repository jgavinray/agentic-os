use super::test_support::{config, event};
use super::*;
use chrono::{TimeZone, Utc};
use serde_json::json;

#[test]
fn unknown_schema_tags_are_not_consumed() {
    let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
    let events = vec![event(
        "e1",
        "failed_attempt",
        json!({
            "detection_tags": [{
                "type": "wrong_endpoint",
                "source": "hook_parser",
                "tag_schema_version": 99
            }]
        }),
        now,
    )];

    let report = extract_records(&events, &config(now));
    let record = &report.records[0];
    assert_eq!(report.unknown_tag_schema_versions, 1);
    assert_eq!(record.wrong_endpoint_count, 0);
    assert_eq!(record.other_failure_count, 1);
}

#[test]
fn same_condition_from_two_producers_is_retained_but_not_double_counted() {
    let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
    let events = vec![event(
        "e1",
        "failed_attempt",
        json!({
            "detection_tags": [
                {"type": "tool_loop", "tool": "Read", "source": "hook_parser", "tag_schema_version": 1},
                {"type": "tool_loop", "tool": "Read", "source": "summarizer", "tag_schema_version": 1}
            ]
        }),
        now,
    )];

    let report = extract_records(&events, &config(now));
    let record = &report.records[0];
    assert_eq!(record.tool_loop_count, 1);
    assert_eq!(record.repeated_read_loop_count, 1);
    assert_eq!(record.other_failure_count, 0);
    assert_eq!(record.recommended_constraints.len(), 1);
    assert_eq!(
        record.recommended_constraints[0].constraint_type,
        "avoid_tool_loop"
    );
}
