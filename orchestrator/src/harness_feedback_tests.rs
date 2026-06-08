use super::*;
use serde_json::{json, Value};

fn signal_names(metadata: &Value) -> Vec<&str> {
    metadata["harness_feedback"]["signals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect()
}

#[path = "harness_feedback_annotation_tests.rs"]
mod annotation_tests;

#[path = "harness_feedback_guardrail_tests.rs"]
mod guardrail_tests;
