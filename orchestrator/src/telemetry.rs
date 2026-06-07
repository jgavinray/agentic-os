use metrics::counter;
use serde::Serialize;
use std::sync::{Arc, RwLock};

pub use crate::telemetry_context::{
    record_cache_invalidation, record_context_cache_replacement, record_context_pack,
    record_promotion,
};
pub use crate::telemetry_execution::record_execution_artifact;
pub use crate::telemetry_feature_feedback::{
    record_feature_extraction_duration, record_feature_extraction_failure,
    record_feature_failure_class, record_harness_feedback_learning,
    record_harness_feedback_quarantine, record_harness_feedback_repair_run,
    record_harness_feedback_signal, record_harness_guardrail_decision,
    record_operational_constraint_injected, record_operational_constraint_suppressed,
    record_unknown_tag_schema_version,
};
pub use crate::telemetry_http::{http_metrics_middleware, prometheus_content_type};
pub use crate::telemetry_infrastructure::{
    record_db_query, record_qdrant_request, record_rate_limited, record_sampling_param_override,
};
pub use crate::telemetry_model_services::{
    record_embedder_inference, record_embedder_input_tokens, record_sentiment,
    record_summarizer_candidate, record_summarizer_duration, record_summarizer_tick,
    record_summarizer_written,
};
pub use crate::telemetry_request_classification::{
    record_request_classification, record_request_classification_backfill_run,
    record_request_classification_unknown_label, record_request_classification_write,
    record_request_complexity, record_request_live_policy_action,
    record_request_live_policy_bypass, record_request_risk_flag,
    record_request_route_recommendation,
};
pub use crate::telemetry_runtime_metrics::{record_pool_gauges, record_process_metrics};
pub use crate::telemetry_setup::{install_recorder, prime_metrics};
pub use crate::telemetry_streaming::StreamTracker;
pub use crate::telemetry_tool_mediation::{record_tool_authorization, record_tool_menu_outcome};
pub use crate::telemetry_upstream::{
    record_upstream_litellm, record_upstream_litellm_error, record_upstream_summarizer,
    record_upstream_summarizer_error, reqwest_error_kind, upstream_error_kind,
};
pub use crate::telemetry_usage::{
    record_tokens, record_trajectory_feature_written, record_trajectory_result,
    record_vllm_cache_delta, record_vllm_cache_token_flow,
};

#[derive(Debug, Clone, Serialize, Default)]
pub struct MetricsSnapshot {
    pub context_pack_requests: u64,
    pub context_cache_hits: u64,
    pub context_cache_misses: u64,
    pub context_cache_replacements: u64,
    pub context_pack_build_ms_total: u64,
    pub context_pack_chars_total: u64,
    pub context_pack_tokens_estimate_total: u64,
    pub l0_items_injected: u64,
    pub l1_items_injected: u64,
    pub l2_items_injected: u64,
    pub l3_items_injected: u64,
    pub failed_attempts_injected: u64,
    pub remediations_injected: u64,
    pub failure_history_items_injected: u64,
    pub operational_constraints_injected: u64,
    pub retrieval_semantic_hits: u64,
    pub retrieval_fts_hits: u64,
    pub retrieval_deduped_hits: u64,
    pub processed_tokens: u64,
    pub cached_tokens: u64,
    pub generated_tokens: u64,
    pub promotion_attempts: u64,
    pub promotion_accepted: u64,
    pub promotion_rejected: u64,
    pub memory_source_items: u64,
    pub memory_source_items_with_sources: u64,
    pub memory_source_coverage: f64,
    pub stale_cache_invalidations: u64,
    pub execution_artifacts: u64,
    pub failure_signatures: u64,
    pub patch_lifecycle_events: u64,
    pub validation_results: u64,
    pub remediation_reuse: u64,
    pub trajectory_results: u64,
    pub trajectory_attempts: u64,
    pub trajectory_validation_failures: u64,
    pub trajectory_input_tokens: u64,
    pub trajectory_output_tokens: u64,
    pub trajectory_features: u64,
    pub tool_mediation_decisions: u64,
    pub tool_mediation_tools: u64,
    pub feature_extraction_failures: u64,
    pub feature_tag_schema_version_unknown: u64,
}

#[derive(Clone)]
pub struct MetricsRegistry {
    inner: Arc<RwLock<MetricsSnapshot>>,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(MetricsSnapshot::default())),
        }
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        self.inner.read().unwrap().clone()
    }

    pub(crate) fn increment_tool_mediation_decisions(&self) {
        self.inner.write().unwrap().tool_mediation_decisions += 1;
    }

    pub(crate) fn increment_tool_mediation_tools(&self) {
        self.inner.write().unwrap().tool_mediation_tools += 1;
    }

    pub(crate) fn with_snapshot_mut(&self, update: impl FnOnce(&mut MetricsSnapshot)) {
        update(&mut self.inner.write().unwrap());
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn record_auth_attempt(accepted: bool) {
    let result = if accepted { "accepted" } else { "rejected" };
    counter!("auth_attempts_total", "result" => result).increment(1);
}
