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
