use crate::request_classification_types::{PersistOutcome, RequestClassification};

pub fn record_write_result(
    result: &Result<PersistOutcome, anyhow::Error>,
    metric_outcome: PersistOutcome,
    classification: &RequestClassification,
) {
    match result {
        Ok(outcome) => {
            crate::telemetry::record_request_classification_write(outcome.as_str());
            if matches!(outcome, observed if *observed == metric_outcome) {
                crate::request_classification::record_classification_metrics(classification);
            }
        }
        Err(_) => crate::telemetry::record_request_classification_write("error"),
    }
}
