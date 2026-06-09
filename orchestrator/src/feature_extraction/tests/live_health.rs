use super::*;

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
