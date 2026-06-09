use metrics::{counter, gauge};

pub fn record_execution_artifact(event: &crate::db::AgentEvent) {
    if !crate::execution_feedback::EXECUTION_EVENT_TYPES.contains(&event.event_type.as_str()) {
        return;
    }

    // Only persisted execution events reach here. Keep labels bounded to the
    // canonical event taxonomy and validator/signature normalizers.
    let success = event
        .metadata
        .get("success")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let success_label = if success { "true" } else { "false" };
    counter!(
        "execution_artifacts_total",
        "event_type" => event.event_type.clone(),
        "success" => success_label
    )
    .increment(1);

    let payload = &event.metadata["payload"];
    if !success && crate::execution_feedback::is_failure_outcome_event_type(&event.event_type) {
        let signature = payload["signature"].as_str().unwrap_or("unknown");
        let category = crate::execution_feedback::retry_trigger_category_from_payload(payload);
        counter!(
            "failure_signatures_total",
            "signature" => crate::execution_feedback::bounded_failure_signature_label(signature),
            "category" => category
        )
        .increment(1);

        if event.event_type != crate::execution_feedback::EVENT_TYPE_TOOL_RESULT {
            let task = event
                .metadata
                .get("task")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let task_type = crate::execution_feedback::task_retry_type(task);
            // This is a coarse process-local gauge for outstanding failure
            // chains. It is intentionally bounded by normalized outcome writes
            // and remediations, not by task names, to avoid user-controlled labels.
            gauge!("task_retries").increment(1.0);
            counter!(
                "task_retries_total",
                "task_type" => task_type,
                "outcome" => "still_active",
                "trigger_category" => category
            )
            .increment(1);
        }
    }

    match event.event_type.as_str() {
        crate::execution_feedback::EVENT_TYPE_PATCH_RESULT => {
            let outcome = match event.metadata["payload"]["outcome"].as_str() {
                Some("applied") => "applied",
                Some("reverted") => "reverted",
                _ => "rejected",
            };
            counter!("patch_lifecycle_total", "outcome" => outcome).increment(1);
        }
        crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT
        | crate::execution_feedback::EVENT_TYPE_TEST_RESULT
        | crate::execution_feedback::EVENT_TYPE_LINT_RESULT
        | crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT => {
            // Different artifact types name the validator differently. Collapse
            // them into the bounded validator label set before recording.
            let validator = payload["validator_name"]
                .as_str()
                .or_else(|| payload["framework"].as_str())
                .or_else(|| payload["tool_name"].as_str())
                .or_else(|| payload["target"].as_str())
                .unwrap_or("other");
            let result = if success { "pass" } else { "fail" };
            counter!(
                "validation_results_total",
                "validator" => crate::execution_feedback::bounded_validator_label(validator),
                "result" => result
            )
            .increment(1);
            if !success
                && event.event_role.as_deref()
                    == Some(crate::trajectory::EventRole::Validation.as_str())
            {
                let validator_type = bounded_validator_type_label(
                    payload["validator_type"].as_str().unwrap_or("other"),
                );
                counter!(
                    "trajectory_validation_failures_total",
                    "validator_type" => validator_type
                )
                .increment(1);
            }
        }
        crate::execution_feedback::EVENT_TYPE_REMEDIATION => {
            let task = event
                .metadata
                .get("task")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let signature = payload["signature"].as_str();
            gauge!("task_retries").decrement(1.0);
            counter!(
                "task_retries_total",
                "task_type" => crate::execution_feedback::task_retry_type(task),
                "outcome" => "succeeded",
                "trigger_category" => crate::execution_feedback::retry_trigger_category_from_signature(signature)
            )
            .increment(1);
        }
        _ => {}
    }
}

fn bounded_validator_type_label(value: &str) -> &'static str {
    match value {
        "compile" => "compile",
        "test" => "test",
        "lint" => "lint",
        "type_check" => "type_check",
        "schema" => "schema",
        "static_analysis" => "static_analysis",
        "other" => "other",
        _ => "other",
    }
}
