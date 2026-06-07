use crate::tool_mediation_classification::{
    capability_for_tool_name, command_capability, replacement_for,
};
use crate::tool_mediation_types::{
    ToolAuthorizeRequest, ToolAuthorizeResponse, ToolCapability, TOOL_MEDIATION_POLICY_VERSION,
};

pub fn authorize_tool_call(req: &ToolAuthorizeRequest, enabled: bool) -> ToolAuthorizeResponse {
    authorize_tool_call_with_policy(req, enabled, None)
}

pub fn authorize_tool_call_with_policy(
    req: &ToolAuthorizeRequest,
    enabled: bool,
    policy: Option<&crate::orchestration_policy::OrchestrationPolicy>,
) -> ToolAuthorizeResponse {
    if !enabled {
        return ToolAuthorizeResponse {
            policy_version: TOOL_MEDIATION_POLICY_VERSION,
            decision: "allow",
            reason: "disabled",
            capability: ToolCapability::Unknown.as_str(),
            attempted_tool: req.tool_name.clone(),
            preferred_tool: None,
            replacement: None,
            message: "Tool mediation is disabled; tool call allowed.".to_string(),
        };
    }

    let attempted_capability = capability_for_tool_name(&req.tool_name);
    let command_capability = if attempted_capability == ToolCapability::Shell {
        command_capability(&req.arguments)
    } else {
        attempted_capability
    };

    // Enforce policy before canonical-tool preference. A blocked tool must not
    // get a canonical replacement suggestion first, because blocked means the
    // capability itself is outside the current request envelope.
    if let Some(p) = policy {
        if policy_blocks_tool_capability(p, command_capability) {
            return ToolAuthorizeResponse {
                policy_version: TOOL_MEDIATION_POLICY_VERSION,
                decision: "deny",
                reason: "policy_blocked_tool",
                capability: command_capability.as_str(),
                attempted_tool: req.tool_name.clone(),
                preferred_tool: None,
                replacement: None,
                message: "Tool call denied by orchestration policy.".to_string(),
            };
        }
        if !policy_allows_tool_capability(p, command_capability) {
            return ToolAuthorizeResponse {
                policy_version: TOOL_MEDIATION_POLICY_VERSION,
                decision: "deny",
                reason: "policy_tool_not_allowed",
                capability: command_capability.as_str(),
                attempted_tool: req.tool_name.clone(),
                preferred_tool: None,
                replacement: None,
                message: "Tool call is not allowed by orchestration policy.".to_string(),
            };
        }
    }

    if attempted_capability == ToolCapability::Shell {
        if let Some(preferred) =
            preferred_tool_for_capability(command_capability, &req.available_tools)
        {
            let replacement = replacement_for(command_capability, &preferred, &req.arguments);
            return ToolAuthorizeResponse {
                policy_version: TOOL_MEDIATION_POLICY_VERSION,
                decision: "deny",
                reason: "prefer_canonical_tool",
                capability: command_capability.as_str(),
                attempted_tool: req.tool_name.clone(),
                preferred_tool: Some(preferred.clone()),
                replacement,
                message: format!(
                    "Use the {preferred} tool for {} instead of shell fallback commands.",
                    command_capability.as_str()
                ),
            };
        }
    }

    ToolAuthorizeResponse {
        policy_version: TOOL_MEDIATION_POLICY_VERSION,
        decision: "allow",
        reason: "not_applicable",
        capability: command_capability.as_str(),
        attempted_tool: req.tool_name.clone(),
        preferred_tool: None,
        replacement: None,
        message: "Tool call allowed.".to_string(),
    }
}

/// Map a tool-mediation capability to the corresponding orchestration-policy
/// capability for allowed/blocked checks.
///
/// Mapping semantics:
/// - FileRead -> FileRead (allowed by FileRead or RepoRead; blocked by FileRead)
/// - TextSearch -> RepoRead (allowed by RepoRead; blocked by RepoRead)
/// - FileList -> RepoRead (allowed by RepoRead; blocked by RepoRead)
/// - FileEdit -> FileEdit (allowed/blocked by FileEdit)
/// - Validation -> ShellRead (allowed/blocked by ShellRead)
/// - Publishing -> GitWrite (allowed/blocked by GitWrite)
/// - Shell -> ShellRead (allowed/blocked by ShellRead)
/// - ShellMutation -> ShellMutation (allowed/blocked by ShellMutation)
/// - Unknown -> always false
fn map_capability_to_policy(
    capability: ToolCapability,
) -> crate::orchestration_policy::ToolCapability {
    match capability {
        ToolCapability::FileRead => crate::orchestration_policy::ToolCapability::FileRead,
        ToolCapability::TextSearch => crate::orchestration_policy::ToolCapability::RepoRead,
        ToolCapability::FileList => crate::orchestration_policy::ToolCapability::RepoRead,
        ToolCapability::FileEdit => crate::orchestration_policy::ToolCapability::FileEdit,
        ToolCapability::Validation => crate::orchestration_policy::ToolCapability::ShellRead,
        ToolCapability::Publishing => crate::orchestration_policy::ToolCapability::GitWrite,
        ToolCapability::Shell => crate::orchestration_policy::ToolCapability::ShellRead,
        ToolCapability::ShellMutation => crate::orchestration_policy::ToolCapability::ShellMutation,
        ToolCapability::Unknown => crate::orchestration_policy::ToolCapability::Unknown,
    }
}

/// Check whether the policy allows the given tool capability.
///
/// - FileRead: allowed if allowed_tools contains FileRead OR RepoRead.
/// - TextSearch/FileList: allowed if allowed_tools contains RepoRead.
/// - FileEdit: allowed if allowed_tools contains FileEdit.
/// - Validation/Shell: allowed if allowed_tools contains ShellRead.
/// - Publishing: allowed if allowed_tools contains GitWrite.
/// - Unknown: always false (even if allowed_tools contains Unknown).
pub(crate) fn policy_allows_tool_capability(
    policy: &crate::orchestration_policy::OrchestrationPolicy,
    capability: ToolCapability,
) -> bool {
    match capability {
        ToolCapability::Unknown => return false,
        // FileRead is allowed if either FileRead or RepoRead is in allowed_tools.
        ToolCapability::FileRead => {
            policy
                .allowed_tools
                .contains(&crate::orchestration_policy::ToolCapability::FileRead)
                || policy
                    .allowed_tools
                    .contains(&crate::orchestration_policy::ToolCapability::RepoRead)
        }
        _ => {
            let policy_cap = map_capability_to_policy(capability);
            policy.allowed_tools.contains(&policy_cap)
        }
    }
}

/// Check whether the policy blocks the given tool capability.
///
/// - FileRead: blocked if blocked_tools contains FileRead.
/// - TextSearch/FileList: blocked if blocked_tools contains RepoRead.
/// - FileEdit: blocked if blocked_tools contains FileEdit.
/// - Validation/Shell: blocked if blocked_tools contains ShellRead.
/// - Publishing: blocked if blocked_tools contains GitWrite.
/// - Unknown: always false (even if blocked_tools contains Unknown).
pub(crate) fn policy_blocks_tool_capability(
    policy: &crate::orchestration_policy::OrchestrationPolicy,
    capability: ToolCapability,
) -> bool {
    if matches!(capability, ToolCapability::Unknown) {
        return false;
    }
    let policy_cap = map_capability_to_policy(capability);
    policy.blocked_tools.contains(&policy_cap)
}

fn preferred_tool_for_capability(
    capability: ToolCapability,
    available_tools: &[String],
) -> Option<String> {
    if matches!(capability, ToolCapability::Shell | ToolCapability::Unknown) {
        return None;
    }
    available_tools
        .iter()
        .find(|name| capability_for_tool_name(name) == capability)
        .cloned()
}
