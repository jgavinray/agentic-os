use super::*;

#[test]
fn event_chain_orders_from_root_to_leaf() {
    let correlation_id = uuid::Uuid::new_v4();
    let root_id = uuid::Uuid::new_v4();
    let response_id = uuid::Uuid::new_v4();
    let tool_id = uuid::Uuid::new_v4();
    let validation_id = uuid::Uuid::new_v4();

    let mut root = event_at_level("user_message", "request", 0);
    root.id = root_id.to_string();
    root.correlation_id = Some(correlation_id);
    root.parent_event_id = None;

    let mut response = event_at_level("assistant_message", "patch", 0);
    response.id = response_id.to_string();
    response.correlation_id = Some(correlation_id);
    response.parent_event_id = Some(root_id);

    let mut tool = event_at_level(
        crate::execution_feedback::EVENT_TYPE_TOOL_RESULT,
        "apply",
        0,
    );
    tool.id = tool_id.to_string();
    tool.correlation_id = Some(correlation_id);
    tool.parent_event_id = Some(response_id);

    let mut validation = event_at_level(
        crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT,
        "schema ok",
        0,
    );
    validation.id = validation_id.to_string();
    validation.correlation_id = Some(correlation_id);
    validation.parent_event_id = Some(tool_id);

    let ordered = order_event_chain(
        vec![
            validation.clone(),
            tool.clone(),
            root.clone(),
            response.clone(),
        ],
        &validation.id,
    );
    let ids: Vec<String> = ordered.into_iter().map(|event| event.id).collect();
    assert_eq!(
        ids,
        vec![
            root_id.to_string(),
            response_id.to_string(),
            tool_id.to_string(),
            validation_id.to_string()
        ]
    );
}

#[test]
fn failure_history_context_build_p99_under_five_ms() {
    let mut durations = Vec::new();
    let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
    let mut evidence = ContextEvidence::default();
    evidence.failure_history = (0..50)
        .map(|idx| FailureHistoryItem {
            signature: format!("typescript:TS{idx:04}"),
            category: "typescript".to_string(),
            failure: event_at_level(
                crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT,
                &format!("failure {idx}"),
                0,
            ),
            remediation: Some(event_at_level(
                crate::execution_feedback::EVENT_TYPE_REMEDIATION,
                &format!("remediation {idx}"),
                0,
            )),
        })
        .collect();

    for _ in 0..100 {
        let started = std::time::Instant::now();
        let _ = build_layered_context(
            "r",
            "typescript:TS2322",
            &evidence,
            &[],
            &[],
            &policy,
            8000,
            4000,
        );
        durations.push(started.elapsed());
    }
    durations.sort();
    let p99 = durations[durations.len() - 1];
    let threshold = if cfg!(debug_assertions) {
        std::time::Duration::from_millis(20)
    } else {
        std::time::Duration::from_millis(5)
    };
    assert!(p99 < threshold, "p99 was {p99:?}");
}
