use metrics::describe_counter;

pub(crate) fn describe_request_metrics() {
    describe_counter!(
        "rate_limited_total",
        "Requests rejected by the per-key rate limiter."
    );
    describe_counter!(
        "sampling_param_overrides_total",
        "Sampling parameter override hook executions by bounded parameter and reason."
    );
    describe_counter!(
        "request_classification_backfill_runs_total",
        "Request classification startup and operator backfill runs by bounded result."
    );
    describe_counter!(
        "request_classifications_written_total",
        "Request classification persistence attempts by bounded result."
    );
    describe_counter!(
        "request_classification_unknown_labels_total",
        "Request classification rows with bounded unknown labels by field."
    );
    describe_counter!(
        "request_classifications_total",
        "Request classifications by bounded intent, domain, and route."
    );
    describe_counter!(
        "request_route_recommendations_total",
        "Shadow route recommendations by bounded route."
    );
    describe_counter!(
        "request_risk_flags_total",
        "Request classification risk flags by bounded risk."
    );
    describe_counter!(
        "request_complexity_total",
        "Request classification complexity by bounded complexity."
    );
    describe_counter!(
        "request_live_policy_actions_total",
        "Feature-flagged live request policy actions by bounded action and reason."
    );
    describe_counter!(
        "request_live_policy_bypassed_total",
        "Requests bypassing live request policy by bounded reason."
    );
    describe_counter!(
        "tool_mediation_decisions_total",
        "Deterministic tool menu and tool-call decisions by bounded decision and reason."
    );
    describe_counter!(
        "tool_mediation_tools_total",
        "Tools observed by deterministic mediation by bounded action and capability."
    );
}
