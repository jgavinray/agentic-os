use serde::{Deserialize, Serialize};

pub use crate::orchestration_policy_context_types::ContextSource;
pub use crate::orchestration_policy_posture_types::{
    EditPolicy, GitPolicy, PromptRefinementPolicy, RiskPolicy, RuntimePolicy, ScopePolicy,
    ValidationPolicy,
};
pub use crate::orchestration_policy_tool_types::ToolCapability;

/// Current orchestration policy schema version.
pub const POLICY_SCHEMA_VERSION: i32 = 1;

/// Source label for deterministic-rule derived policies.
pub const POLICY_SOURCE_DETERMINISTIC_RULES: &str = "deterministic_rules";

/// Bounded orchestration policy derived from a request classification.
///
/// `blocked_tools` is authoritative. A capability that appears in
/// `blocked_tools` is removed from both `allowed_tools` and `required_tools`
/// before the policy is returned. That invariant lets downstream callers check
/// either list without needing to resolve conflicts themselves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrchestrationPolicy {
    /// Context systems eligible to contribute to the request context pack.
    pub context_sources: Vec<ContextSource>,
    /// Capabilities the model/client may use if offered by the client.
    pub allowed_tools: Vec<ToolCapability>,
    /// Capabilities required for a correct answer or operation.
    pub required_tools: Vec<ToolCapability>,
    /// Capabilities that must be hidden or denied.
    pub blocked_tools: Vec<ToolCapability>,
    /// File edit posture.
    pub edit_policy: EditPolicy,
    /// Validation expectation.
    pub validation_policy: ValidationPolicy,
    /// Git mutation expectation.
    pub git_policy: GitPolicy,
    /// Runtime/service operation expectation.
    pub runtime_policy: RuntimePolicy,
    /// Scope invariants and flags.
    pub scope_policy: Vec<ScopePolicy>,
    /// Prompt/spec review workflow mode.
    pub prompt_refinement_policy: PromptRefinementPolicy,
    /// Applied risk overlays.
    pub risk_policy: Vec<RiskPolicy>,
}
