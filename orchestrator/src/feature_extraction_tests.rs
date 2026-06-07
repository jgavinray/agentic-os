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
fn live_operational_health_tags_populate_counters() {
    let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
    let events = vec![
        event(
            "e1",
            "context_pack",
            annotate_event_metadata(
                "context_pack",
                "",
                None,
                json!({
                    "total_context_tokens": 26,
                    "retrieved_event_ids": [],
                    "memory_levels_used": [],
                    "truncated": true
                }),
            ),
            now,
        ),
        event(
            "e2",
            "assistant_message",
            annotate_event_metadata(
                "assistant_message",
                "",
                None,
                json!({
                    "input_tokens": 120000,
                    "latency_ms": 65000,
                    "finish_reason": "tool_use"
                }),
            ),
            now + Duration::seconds(1),
        ),
        event(
            "e3",
            "trajectory_result",
            annotate_event_metadata(
                "trajectory_result",
                "abandoned before model",
                None,
                json!({
                    "final_status": "abandoned",
                    "total_model_calls": 0,
                    "total_input_tokens": 125000,
                    "total_latency_ms": 70000
                }),
            ),
            now + Duration::seconds(2),
        ),
        event(
            "e4",
            "trajectory_result",
            annotate_event_metadata(
                "trajectory_result",
                "single call abandoned",
                None,
                json!({
                    "final_status": "abandoned",
                    "total_model_calls": 1,
                    "total_tool_calls": 0,
                    "total_validations": 0
                }),
            ),
            now + Duration::seconds(3),
        ),
        event(
            "e5",
            "config_check",
            annotate_event_metadata(
                "config_check",
                "shared summarizer upstream",
                None,
                json!({
                    "summarizer_base_url": "http://litellm:4000",
                    "litellm_url": "http://litellm:4000/"
                }),
            ),
            now + Duration::seconds(4),
        ),
    ];

    let record = &extract_records(&events, &config(now + Duration::seconds(4))).records[0];
    assert_eq!(record.context_pack_empty_count, 1);
    assert_eq!(record.context_pack_truncated_count, 1);
    assert_eq!(record.high_input_token_count, 2);
    assert_eq!(record.slow_upstream_model_count, 2);
    assert_eq!(record.empty_tool_use_message_count, 1);
    assert_eq!(record.abandoned_before_model_count, 1);
    assert_eq!(record.single_model_abandoned_no_tools_count, 1);
    assert_eq!(record.summarizer_shared_upstream_count, 1);
    assert!(record
        .recommended_constraints
        .iter()
        .any(|item| item.constraint_type == "fix_context_retrieval"));
    assert!(record
        .recommended_constraints
        .iter()
        .any(|item| item.constraint_type == "reduce_context_bloat"));
    assert!(record
        .recommended_constraints
        .iter()
        .any(|item| item.constraint_type == "separate_summarizer_upstream"));
}

#[test]
fn bootstrap_catches_loop_warning_and_bracketed_interruption() {
    let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
    let read_loop = AgentEvent {
        summary: "LOOP WARNING: Tool 'Read' called 3 times with identical params".to_string(),
        ..event("e1", "failed_attempt", json!({}), now)
    };
    let bash_loop = AgentEvent {
        summary: "LOOP WARNING: Tool 'Bash' called 3 times with identical params".to_string(),
        ..event("e2", "failed_attempt", json!({}), now)
    };
    let interrupted = AgentEvent {
        summary: "[Request interrupted by user for tool use]".to_string(),
        ..event("e3", "failed_attempt", json!({}), now)
    };

    assert!(bootstrap_detection_tags_for_event(&read_loop)
        .iter()
        .any(|tag| tag.tag_type == "tool_loop" && tag.tool.as_deref() == Some("Read")));
    assert!(bootstrap_detection_tags_for_event(&bash_loop)
        .iter()
        .any(|tag| tag.tag_type == "tool_loop" && tag.tool.as_deref() == Some("Bash")));
    assert!(bootstrap_detection_tags_for_event(&interrupted)
        .iter()
        .any(|tag| tag.tag_type == "user_interruption"));
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

#[test]
fn bootstrap_structured_detection_matches_live_except_source() {
    let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
    let metadata = json!({
        "failure_class": "wrong_endpoint",
        "known_facts": {"endpoint": "http://host.docker.internal:8088"}
    });
    let live = live_detection_tags_for_event("failed_attempt", "bad endpoint", None, &metadata)
        .into_iter()
        .map(|mut tag| {
            tag.source = "bootstrap_migration".to_string();
            tag
        })
        .collect::<Vec<_>>();
    let historical = event("e1", "failed_attempt", metadata, now);
    let bootstrap = bootstrap_detection_tags_for_event(&historical);
    assert_eq!(live, bootstrap);
}

#[test]
fn backward_compatible_extra_fields_do_not_break_extraction() {
    let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
    let events = vec![event(
        "e1",
        "failed_attempt",
        json!({
            "known_facts": {"endpoint": "http://host.docker.internal:8088"},
            "detection_tags": [{
                "type": "wrong_endpoint",
                "source": "validation_parser",
                "tag_schema_version": 1,
                "new_optional_field": "ok"
            }]
        }),
        now,
    )];

    let record = &extract_records(&events, &config(now)).records[0];
    assert_eq!(record.wrong_endpoint_count, 1);
    assert!(record
        .recommended_constraints
        .iter()
        .any(|item| item.constraint_type == "use_known_endpoint"));
}
