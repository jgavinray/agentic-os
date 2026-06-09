use metrics::counter;

use crate::prompt_intervention_records::PromptInterventionRecord;

pub const PROMPT_INTERVENTION_RUNTIME_WRITE_RESULTS: &[&str] =
    &["success", "error", "no_pool", "assembly_error"];
pub const PROMPT_INTERVENTION_BACKFILL_RESULTS: &[&str] = &["success", "error", "skipped"];

pub fn record_prompt_intervention_emitted(record: &PromptInterventionRecord) {
    counter!(
        "prompt_intervention_records_total",
        "intervention_type" => record.intervention_type.as_str(),
        "labeler_type" => record.labeler_type.as_str()
    )
    .increment(1);
    counter!(
        "prompt_intervention_signal_family_total",
        "signal_family" => record.signal_family.as_str()
    )
    .increment(1);
    counter!(
        "prompt_intervention_burden_type_total",
        "burden_type" => record.burden_type.as_str()
    )
    .increment(1);
    counter!(
        "prompt_intervention_failure_relation_total",
        "failure_relation" => record.failure_relation.as_str()
    )
    .increment(1);
}

pub fn record_prompt_intervention_runtime_write_attempt(result: &'static str) {
    counter!("prompt_intervention_runtime_write_attempts_total", "result" => result).increment(1);
}

pub fn record_prompt_intervention_backfill_run(result: &'static str) {
    counter!("prompt_intervention_backfill_runs_total", "result" => result).increment(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_write_results_are_bounded() {
        assert_eq!(
            PROMPT_INTERVENTION_RUNTIME_WRITE_RESULTS,
            &["success", "error", "no_pool", "assembly_error"]
        );
    }

    #[test]
    fn backfill_results_are_bounded() {
        assert_eq!(
            PROMPT_INTERVENTION_BACKFILL_RESULTS,
            &["success", "error", "skipped"]
        );
    }

    #[test]
    fn metric_result_labels_are_safe() {
        for value in PROMPT_INTERVENTION_RUNTIME_WRITE_RESULTS
            .iter()
            .chain(PROMPT_INTERVENTION_BACKFILL_RESULTS)
        {
            assert!(value.len() <= 64);
            assert!(value
                .chars()
                .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_'));
        }
    }
}
