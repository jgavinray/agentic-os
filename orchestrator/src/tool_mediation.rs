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

pub use crate::tool_mediation_authorization::{
    authorize_tool_call, authorize_tool_call_with_policy,
};
pub use crate::tool_mediation_classification::{
    bounded_capability, bounded_decision, bounded_reason, bounded_tool_action, detect_tool_intent,
};
pub(crate) use crate::tool_mediation_policy::{
    policy_allows_tool_capability, policy_blocks_tool_capability,
};
pub use crate::tool_mediation_shaping::{
    shape_anthropic_request, shape_anthropic_request_with_policy, shape_openai_request,
    shape_openai_request_with_policy,
};
pub use crate::tool_mediation_types::{
    ToolAuthorizeRequest, ToolAuthorizeResponse, ToolCapability, ToolIntent, ToolMenuOutcome,
    ToolPayloadFormat, ToolSummary, TOOL_MEDIATION_POLICY_VERSION,
};

#[cfg(test)]
#[path = "tool_mediation_policy_allow_tests.rs"]
mod policy_allow_tests;

#[cfg(test)]
#[path = "tool_mediation_policy_deny_tests.rs"]
mod policy_deny_tests;

#[cfg(test)]
#[path = "tool_mediation_policy_test_support.rs"]
mod test_support;

#[cfg(test)]
#[path = "tool_mediation_canonical_tests.rs"]
mod canonical_tests;

#[cfg(test)]
#[path = "tool_mediation_compat_tests.rs"]
mod compat_tests;

#[cfg(test)]
#[path = "tool_mediation_policy_filter_tests.rs"]
mod policy_filter_tests;

#[cfg(test)]
#[path = "tool_mediation_policy_mapping_tests.rs"]
mod policy_mapping_tests;
