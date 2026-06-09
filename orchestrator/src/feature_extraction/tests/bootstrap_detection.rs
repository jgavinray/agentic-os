use super::*;

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
