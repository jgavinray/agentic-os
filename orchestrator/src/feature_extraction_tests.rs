use super::*;
use chrono::{DateTime, Duration, TimeZone, Utc};
use serde_json::json;
use serde_json::Value;

fn event(id: &str, event_type: &str, metadata: Value, created_at: DateTime<Utc>) -> AgentEvent {
    AgentEvent {
        id: id.to_string(),
        session_id: "s1".to_string(),
        repo: "repo".to_string(),
        actor: "agent".to_string(),
        event_type: event_type.to_string(),
        summary: event_type.to_string(),
        evidence: None,
        metadata,
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: None,
        created_at,
        summary_level: 0,
    }
}

fn config(now: DateTime<Utc>) -> ExtractionConfig {
    ExtractionConfig {
        evaluation_time: now,
        feature_window_sec: 3600,
        constraint_freshness_window_sec: 1800,
        max_operational_constraints: 5,
    }
}

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

#[test]
fn known_facts_feed_constraint_templates() {
    let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
    let events = vec![
        event(
            "e1",
            "failed_attempt",
            json!({
                "known_facts": {"auth_header": "Bearer test"},
                "detection_tags": [{"type": "missing_auth", "source": "validation_parser", "tag_schema_version": 1}]
            }),
            now,
        ),
        event(
            "e2",
            "failed_attempt",
            json!({
                "known_facts": {"endpoint": "http://host.docker.internal:8088"},
                "detection_tags": [{"type": "wrong_endpoint", "source": "validation_parser", "tag_schema_version": 1}]
            }),
            now + Duration::seconds(1),
        ),
    ];

    let report = extract_records(&events, &config(now + Duration::seconds(1)));
    let record = &report.records[0];
    assert_eq!(record.missing_auth_count, 1);
    assert_eq!(record.wrong_endpoint_count, 1);
    assert_eq!(record.known_auth_header.as_deref(), Some("Bearer test"));
    assert_eq!(
        record.known_endpoint.as_deref(),
        Some("http://host.docker.internal:8088")
    );
    assert!(record
        .recommended_constraints
        .iter()
        .any(|constraint| constraint.constraint_type == "use_known_auth"));
    assert!(record
        .recommended_constraints
        .iter()
        .any(|constraint| constraint.constraint_type == "use_known_endpoint"));
}

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

#[test]
fn priority_cap_keeps_highest_priority_constraints() {
    let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
    let mut cfg = config(now);
    cfg.max_operational_constraints = 4;
    let events = vec![
        event(
            "e1",
            "failed_attempt",
            json!({"known_facts": {"auth_header": "Bearer t"}, "detection_tags": [{"type": "missing_auth", "source": "validation_parser", "tag_schema_version": 1}]}),
            now,
        ),
        event(
            "e2",
            "failed_attempt",
            json!({"known_facts": {"endpoint": "http://host:8088"}, "detection_tags": [{"type": "wrong_endpoint", "source": "validation_parser", "tag_schema_version": 1}]}),
            now,
        ),
        event(
            "e3",
            "failed_attempt",
            json!({"known_facts": {"migration_fix": "CREATE EXTENSION IF NOT EXISTS vector"}, "detection_tags": [{"type": "migration_failure", "source": "validation_parser", "tag_schema_version": 1}]}),
            now,
        ),
        event(
            "e4",
            "failed_attempt",
            json!({"detection_tags": [{"type": "tool_loop", "tool": "Read", "source": "tool_loop_detector", "tag_schema_version": 1}]}),
            now,
        ),
        event(
            "e5",
            "failed_attempt",
            json!({"detection_tags": [{"type": "user_interruption", "source": "failed_attempt_classifier", "tag_schema_version": 1}]}),
            now,
        ),
        event(
            "e6",
            "failed_attempt",
            json!({"detection_tags": [{"type": "summarization_failure", "source": "summarizer", "tag_schema_version": 1}]}),
            now,
        ),
    ];

    let record = &extract_records(&events, &cfg).records[0];
    let types: Vec<_> = record
        .recommended_constraints
        .iter()
        .map(|constraint| constraint.constraint_type.as_str())
        .collect();
    assert_eq!(
        types,
        vec![
            "use_known_auth",
            "use_known_endpoint",
            "use_known_migration_fix",
            "avoid_tool_loop"
        ]
    );
    assert!(record.suppressed_constraints.iter().any(|item| {
        item.constraint_type == "handle_user_interruption" && item.reason == "priority_cap"
    }));
    assert!(record.suppressed_constraints.iter().any(|item| {
        item.constraint_type == "handle_summarization_failure" && item.reason == "priority_cap"
    }));
}

#[test]
fn read_and_bash_loops_combine_before_priority_cap() {
    let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
    let events = vec![
        event(
            "e1",
            "failed_attempt",
            json!({"detection_tags": [{"type": "tool_loop", "tool": "Read", "source": "tool_loop_detector", "tag_schema_version": 1}]}),
            now,
        ),
        event(
            "e2",
            "failed_attempt",
            json!({"detection_tags": [{"type": "tool_loop", "tool": "Bash", "source": "tool_loop_detector", "tag_schema_version": 1}]}),
            now,
        ),
    ];
    let record = &extract_records(&events, &config(now)).records[0];
    assert_eq!(record.recommended_constraints.len(), 1);
    let text = &record.recommended_constraints[0].text;
    assert!(text.contains("Read"));
    assert!(text.contains("Bash"));
}

#[test]
fn token_budget_drops_lowest_priority_first() {
    let constraints = vec![
        OperationalConstraint { constraint_type: "use_known_auth".to_string(), text: "Use `Bearer t` when calling protected orchestrator endpoints.".to_string() },
        OperationalConstraint { constraint_type: "handle_summarization_failure".to_string(), text: "If summarization returns an empty response, inspect the provider or LiteLLM response body before retrying.".to_string() },
    ];
    let (kept, suppressed) = enforce_constraint_token_budget(&constraints, 30);
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].constraint_type, "use_known_auth");
    assert_eq!(
        suppressed[0].constraint_type,
        "handle_summarization_failure"
    );
    assert_eq!(suppressed[0].reason, "token_budget");
}
