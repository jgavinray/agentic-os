use crate::tool_mediation_types::ToolCapability;

/// Map a tool-mediation capability to the corresponding orchestration-policy
/// capability for allowed/blocked checks.
///
/// Mapping semantics:
/// - FileRead -> FileRead (allowed by FileRead or RepoRead; blocked by FileRead)
/// - TextSearch -> RepoRead (allowed by RepoRead; blocked by RepoRead)
/// - FileList -> RepoRead (allowed by RepoRead; blocked by RepoRead)
/// - FileEdit -> FileEdit (allowed/blocked by FileEdit)
/// - Validation -> Validation (allowed by named validation tools, not shell)
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
        ToolCapability::Validation => crate::orchestration_policy::ToolCapability::Validation,
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
/// - Validation: allowed if allowed_tools contains Validation.
/// - Shell: allowed if allowed_tools contains ShellRead.
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
/// - Validation: blocked if blocked_tools contains Validation.
/// - Shell: blocked if blocked_tools contains ShellRead.
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
