use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColumnKind {
    UnboundedString,
    BoundedString,
    ClosedEnum,
    Integer,
    BigInt,
    Boolean,
}

#[test]
fn schema_privacy_test() {
    let allowed_unbounded_strings: HashSet<&str> = [
        "decision_id",
        "candidate_id",
        "source_record_id",
        "repo",
        "session_id",
        "trajectory_id",
        "source_decision_id",
        "context_policy_version",
        "drop_reason_other_detail",
    ]
    .iter()
    .copied()
    .collect();

    let envelope_cols: Vec<(&str, ColumnKind)> = vec![
        ("decision_id", ColumnKind::UnboundedString),
        ("source_decision_id", ColumnKind::UnboundedString),
        ("repo", ColumnKind::UnboundedString),
        ("session_id", ColumnKind::UnboundedString),
        ("trajectory_id", ColumnKind::UnboundedString),
        ("request_route", ColumnKind::ClosedEnum),
        ("task_type", ColumnKind::ClosedEnum),
        ("created_at_micros", ColumnKind::BigInt),
        ("context_policy_version", ColumnKind::UnboundedString),
        ("feature_schema_version", ColumnKind::Integer),
        ("total_token_budget", ColumnKind::Integer),
        ("candidate_tokens_total", ColumnKind::Integer),
        ("fixed_overhead_tokens", ColumnKind::Integer),
        ("final_estimated_context_tokens", ColumnKind::Integer),
        ("cache_status", ColumnKind::ClosedEnum),
        ("candidate_count", ColumnKind::Integer),
        ("injected_candidate_count", ColumnKind::Integer),
        ("assembly_duration_micros", ColumnKind::BigInt),
        ("assembly_status", ColumnKind::ClosedEnum),
    ];
    let candidate_cols: Vec<(&str, ColumnKind)> = vec![
        ("candidate_id", ColumnKind::UnboundedString),
        ("decision_id", ColumnKind::UnboundedString),
        ("source_type", ColumnKind::ClosedEnum),
        ("source_record_id", ColumnKind::UnboundedString),
        ("event_type", ColumnKind::ClosedEnum),
        ("summary_level", ColumnKind::ClosedEnum),
        ("age_seconds", ColumnKind::Integer),
        ("same_repo", ColumnKind::Boolean),
        ("same_session", ColumnKind::Boolean),
        ("same_trajectory", ColumnKind::Boolean),
        ("estimated_token_cost", ColumnKind::Integer),
        ("token_cost_bucket", ColumnKind::ClosedEnum),
        ("retrieval_score_bucket", ColumnKind::ClosedEnum),
        ("failure_class", ColumnKind::ClosedEnum),
        ("operational_constraint_type", ColumnKind::ClosedEnum),
        ("context_section", ColumnKind::ClosedEnum),
        ("duplicate_coverage", ColumnKind::ClosedEnum),
        ("deterministic_score_bucket", ColumnKind::ClosedEnum),
        ("deterministic_rank", ColumnKind::Integer),
        ("budget_before_candidate", ColumnKind::Integer),
        ("budget_after_candidate", ColumnKind::Integer),
        ("injected", ColumnKind::Boolean),
        ("drop_reason", ColumnKind::ClosedEnum),
        ("drop_reason_other_detail", ColumnKind::BoundedString),
        ("representation_selected", ColumnKind::ClosedEnum),
    ];
    let outcome_cols: Vec<(&str, ColumnKind)> = vec![
        ("decision_id", ColumnKind::UnboundedString),
        ("trajectory_outcome", ColumnKind::ClosedEnum),
        ("user_correction_after_decision", ColumnKind::Boolean),
        ("retry_after_decision", ColumnKind::Boolean),
        ("validation_result", ColumnKind::ClosedEnum),
        ("tool_loop_detected_after_decision", ColumnKind::Boolean),
        ("request_latency_bucket", ColumnKind::ClosedEnum),
        ("input_token_bucket", ColumnKind::ClosedEnum),
        ("output_token_bucket", ColumnKind::ClosedEnum),
        ("outcome_joined_at_micros", ColumnKind::BigInt),
        ("outcome_window_closed", ColumnKind::Boolean),
        ("feature_schema_version", ColumnKind::Integer),
    ];

    check_columns("envelopes", &envelope_cols, &allowed_unbounded_strings);
    check_columns("candidates", &candidate_cols, &allowed_unbounded_strings);
    check_columns("outcomes", &outcome_cols, &allowed_unbounded_strings);
}

fn check_columns(name: &str, cols: &[(&str, ColumnKind)], allowed: &HashSet<&str>) {
    for (col_name, col_kind) in cols {
        let string_kind = matches!(
            col_kind,
            ColumnKind::UnboundedString | ColumnKind::BoundedString
        );
        if string_kind && !allowed.contains(*col_name) {
            panic!(
                "{}: column '{}' is a string field but is not in the privacy allowlist",
                name, col_name
            );
        }
    }
}
