use super::*;
use crate::orchestration_policy::{
    EditPolicy, GitPolicy, OrchestrationPolicy, RuntimePolicy, ToolCapability as PolicyCap,
};
use serde_json::Value;

pub fn policy_with_tools(
    allowed_tools: Vec<PolicyCap>,
    blocked_tools: Vec<PolicyCap>,
) -> OrchestrationPolicy {
    OrchestrationPolicy {
        context_sources: vec![],
        allowed_tools,
        required_tools: vec![],
        blocked_tools,
        edit_policy: EditPolicy::ReadOnly,
        validation_policy: crate::orchestration_policy::ValidationPolicy::None,
        git_policy: GitPolicy::NoGitChanges,
        runtime_policy: RuntimePolicy::NoRestart,
        scope_policy: vec![],
        prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
        risk_policy: vec![],
    }
}

pub fn tool_request(
    user_intent: &str,
    tool_name: &str,
    arguments: Value,
    available_tools: Vec<&str>,
) -> ToolAuthorizeRequest {
    ToolAuthorizeRequest {
        session_id: None,
        repo: None,
        task: None,
        trajectory_id: None,
        attempt_index: None,
        parent_event_id: None,
        user_intent: Some(user_intent.to_string()),
        tool_name: tool_name.to_string(),
        arguments,
        available_tools: available_tools.into_iter().map(str::to_string).collect(),
    }
}
