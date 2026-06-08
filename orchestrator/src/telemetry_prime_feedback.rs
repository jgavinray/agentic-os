use metrics::{counter, histogram};

pub(crate) fn prime_execution_feedback_metrics() {
    for event_type in crate::execution_feedback::EXECUTION_EVENT_TYPES {
        for success in ["true", "false"] {
            counter!(
                "execution_artifacts_total",
                "event_type" => event_type,
                "success" => success
            )
            .increment(0);
        }
    }
    for outcome in ["applied", "rejected", "reverted"] {
        counter!("patch_lifecycle_total", "outcome" => outcome).increment(0);
    }
    for validator in [
        "cargo",
        "pytest",
        "npm test",
        "eslint",
        "tsc",
        "mypy",
        "ruff",
        "terraform",
        "kubectl",
        "other",
    ] {
        for result in ["pass", "fail"] {
            counter!(
                "validation_results_total",
                "validator" => validator,
                "result" => result
            )
            .increment(0);
        }
    }
}

pub(crate) fn prime_trajectory_metrics() {
    for status in crate::trajectory::FINAL_STATUSES {
        counter!("trajectory_results_total", "status" => status).increment(0);
    }
    counter!("trajectory_attempts_total").increment(0);
    for validator_type in crate::execution_feedback::VALIDATOR_TYPES {
        counter!("trajectory_validation_failures_total", "validator_type" => validator_type)
            .increment(0);
    }
    for direction in ["input", "output"] {
        counter!("trajectory_tokens_total", "direction" => direction).increment(0);
    }
    counter!("trajectory_features_total").increment(0);
}

pub(crate) fn prime_feature_feedback_metrics() {
    for failure_class in crate::feature_extraction::FEATURE_FAILURE_CLASSES {
        counter!("feature_failure_classes_total", "failure_class" => failure_class).increment(0);
    }
    for constraint_type in crate::feature_extraction::OPERATIONAL_CONSTRAINT_TYPES {
        counter!(
            "operational_constraints_injected_total",
            "constraint_type" => constraint_type
        )
        .increment(0);
        for reason in crate::feature_extraction::OPERATIONAL_SUPPRESSION_REASONS {
            counter!(
                "operational_constraints_suppressed_total",
                "constraint_type" => constraint_type,
                "reason" => reason
            )
            .increment(0);
        }
    }
    histogram!("feature_extraction_duration_seconds").record(0.0);
    for stage in ["extraction", "constraint_build", "persistence"] {
        counter!("feature_extraction_failures_total", "stage" => stage).increment(0);
    }
    counter!("feature_tag_schema_version_unknown_total").increment(0);
    for signal_type in crate::harness_feedback::HARNESS_SIGNAL_TYPES {
        counter!("harness_feedback_signals_total", "signal_type" => signal_type).increment(0);
    }
    for reason in crate::harness_feedback::HARNESS_QUARANTINE_REASONS {
        counter!("harness_feedback_quarantined_total", "reason" => reason).increment(0);
    }
    for status in crate::harness_feedback::HARNESS_LEARNING_STATUSES {
        counter!("harness_feedback_learning_records_total", "status" => status).increment(0);
    }
    for result in ["success", "failure"] {
        counter!("harness_feedback_repair_runs_total", "result" => result).increment(0);
    }
    for action in crate::harness_feedback::HARNESS_GUARDRAIL_ACTIONS {
        for reason in crate::harness_feedback::HARNESS_GUARDRAIL_REASONS {
            counter!(
                "harness_guardrail_decisions_total",
                "action" => action,
                "reason" => reason
            )
            .increment(0);
        }
    }
}
