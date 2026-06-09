use metrics::{describe_counter, describe_gauge, describe_histogram};

pub(crate) fn describe_execution_metrics() {
    describe_counter!(
        "execution_artifacts_total",
        "Structured execution artifact events written."
    );
    describe_counter!(
        "failure_signatures_total",
        "Canonical failure fingerprints recorded."
    );
    describe_counter!(
        "patch_lifecycle_total",
        "Patch lifecycle outcomes recorded."
    );
    describe_counter!(
        "validation_results_total",
        "Deterministic validation outcomes recorded."
    );
    describe_counter!(
        "remediation_reuse_total",
        "Prior remediations surfaced in context packs."
    );
    describe_counter!(
        "task_retries_total",
        "Historical retry behavior by task type, outcome, and trigger category."
    );
    describe_gauge!(
        "task_retries",
        "Currently active retry chains across tasks."
    );
    describe_counter!(
        "trajectory_results_total",
        "Completed trajectory results by bounded operational status."
    );
    describe_counter!(
        "trajectory_attempts_total",
        "Attempts observed across completed trajectories."
    );
    describe_counter!(
        "trajectory_validation_failures_total",
        "Failed trajectory validations by bounded validator type."
    );
    describe_counter!(
        "trajectory_tokens_total",
        "Trajectory model tokens by direction."
    );
    describe_counter!(
        "trajectory_features_total",
        "Operational feature records written or updated."
    );
    describe_counter!(
        "feature_failure_classes_total",
        "Operational feature failure classes observed."
    );
    describe_counter!(
        "operational_constraints_injected_total",
        "Operational constraints injected into context packs."
    );
    describe_counter!(
        "operational_constraints_suppressed_total",
        "Operational constraints suppressed by bounded deterministic reasons."
    );
    describe_histogram!(
        "feature_extraction_duration_seconds",
        "Inline feature extraction latency in seconds."
    );
    describe_counter!(
        "feature_extraction_failures_total",
        "Feature extraction failures by stage."
    );
    describe_counter!(
        "feature_tag_schema_version_unknown_total",
        "Detection tags skipped due to unknown schema versions."
    );
    describe_counter!(
        "harness_feedback_signals_total",
        "Deterministic harness feedback signals recorded on persisted events."
    );
    describe_counter!(
        "harness_feedback_quarantined_total",
        "Persisted events marked as excluded from future context memory."
    );
    describe_counter!(
        "harness_feedback_learning_records_total",
        "Harness feedback records available for learning-oriented analysis."
    );
    describe_counter!(
        "harness_feedback_repair_runs_total",
        "Harness feedback startup and background repair runs by bounded result."
    );
    describe_counter!(
        "harness_guardrail_decisions_total",
        "Deterministic runtime guardrail decisions by bounded action and reason."
    );
    describe_counter!(
        "prompt_intervention_records_total",
        "Prompt intervention records emitted by bounded intervention type and labeler type."
    );
    describe_counter!(
        "prompt_intervention_signal_family_total",
        "Prompt intervention records emitted by bounded signal family."
    );
    describe_counter!(
        "prompt_intervention_burden_type_total",
        "Prompt intervention records emitted by bounded burden type."
    );
    describe_counter!(
        "prompt_intervention_failure_relation_total",
        "Prompt intervention records emitted by bounded failure relation."
    );
    describe_counter!(
        "prompt_intervention_backfill_runs_total",
        "Prompt intervention backfill runs by bounded result."
    );
    describe_counter!(
        "prompt_intervention_runtime_write_attempts_total",
        "Runtime prompt intervention write attempts by bounded result."
    );
}
