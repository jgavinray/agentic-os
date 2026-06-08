use super::*;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Default)]
struct ForceTemperaturePolicy;

impl SamplingPolicy for ForceTemperaturePolicy {
    fn apply(&self, request: &mut Value) {
        request[PARAM_TEMPERATURE] = json!(0.1);
    }
}

#[derive(Default)]
struct CountingPolicy {
    count: AtomicUsize,
}

impl SamplingPolicy for CountingPolicy {
    fn apply(&self, _request: &mut Value) {
        self.count.fetch_add(1, Ordering::SeqCst);
    }
}

#[path = "sampling_audit_tests.rs"]
mod audit_tests;

#[path = "sampling_override_tests.rs"]
mod override_tests;

#[path = "sampling_performance_tests.rs"]
mod performance_tests;
