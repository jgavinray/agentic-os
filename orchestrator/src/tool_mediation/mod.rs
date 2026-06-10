//! Deterministic tool menu shaping and tool-call authorization.
//!
//! This module is the client-tool boundary. It sees the concrete tools a client
//! offered and the concrete tool calls a client is about to execute. It maps
//! those tool names and shell command shapes into a small local capability
//! vocabulary, then optionally applies the broader request-level
//! `OrchestrationPolicy`.
//!
//! The split is deliberate: orchestration policy describes what the request may
//! do, while tool mediation translates that request-level policy into the
//! client-specific tool menus and authorization decisions available in proxy
//! mode.

pub mod authorization;
pub mod authorization_types;
pub mod broadening;
pub mod canonical;
pub mod classification;
pub mod decision_store;
pub mod edit_targets;
pub mod payload;
pub mod policy;
pub mod shaping;
pub mod shaping_policy;
pub mod shell;
pub mod telemetry;
pub mod types;
pub mod validation_gate;

pub use authorization::{
    authorize_tool_call, authorize_tool_call_with_policy, edit_target_for_request,
    single_file_target_denial,
};
pub use broadening::{
    broaden_policy_for_observed_edits, trajectory_tool_evidence, TrajectoryToolEvidence,
};
pub use classification::{
    bounded_capability, bounded_decision, bounded_reason, bounded_tool_action, detect_tool_intent,
};
pub(crate) use policy::{policy_allows_tool_capability, policy_blocks_tool_capability};
pub use shaping::{
    shape_anthropic_request, shape_anthropic_request_with_policy, shape_openai_request,
    shape_openai_request_with_policy,
};
pub use types::{
    missing_implementation_tool_capabilities, policy_requires_implementation_tool_surface,
    ToolAuthorizeRequest, ToolAuthorizeResponse, ToolCapability, ToolIntent, ToolMenuOutcome,
    ToolPayloadFormat, ToolSummary, TOOL_MEDIATION_POLICY_VERSION,
};
pub use validation_gate::{
    evaluate_discovery_gate, evaluate_validation_gate, ValidationGateOutcome,
};

#[cfg(test)]
mod tests {
    pub(crate) use super::*;
    pub(crate) mod support;
    pub(crate) use support as test_support;

    mod allow;
    mod canonical;
    mod compat;
    mod deny;
    mod filter;
    mod mapping;
}
