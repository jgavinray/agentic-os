use crate::tool_mediation_classification::{
    capability_for_tool_name, command_capability, replacement_for,
};
use crate::tool_mediation_policy::{policy_allows_tool_capability, policy_blocks_tool_capability};
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
        if attempted_capability == ToolCapability::Shell {
            if policy_blocks_tool_capability(p, ToolCapability::Shell) {
                return ToolAuthorizeResponse {
                    policy_version: TOOL_MEDIATION_POLICY_VERSION,
                    decision: "deny",
                    reason: "policy_blocked_tool",
                    capability: ToolCapability::Shell.as_str(),
                    attempted_tool: req.tool_name.clone(),
                    preferred_tool: None,
                    replacement: None,
                    message: "Tool call denied by orchestration policy.".to_string(),
                };
            }
            if !policy_allows_tool_capability(p, ToolCapability::Shell) {
                return ToolAuthorizeResponse {
                    policy_version: TOOL_MEDIATION_POLICY_VERSION,
                    decision: "deny",
                    reason: "policy_tool_not_allowed",
                    capability: ToolCapability::Shell.as_str(),
                    attempted_tool: req.tool_name.clone(),
                    preferred_tool: None,
                    replacement: None,
                    message: "Tool call is not allowed by orchestration policy.".to_string(),
                };
            }
        }
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

    if command_capability == ToolCapability::FileEdit {
        if let Some(p) = policy {
            if let Some(denial) = edit_scope_denial(p, req) {
                return denial;
            }
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

/// Enforce file-scoped edit policies on a concrete edit tool call.
///
/// When the policy limits edits to explicitly named files and the request
/// text names at least one file, an edit targeting a file the request never
/// mentions is denied. When the request names no files there is no scope
/// evidence, so the call passes (the operating envelope still applies).
fn edit_scope_denial(
    policy: &crate::orchestration_policy::OrchestrationPolicy,
    req: &ToolAuthorizeRequest,
) -> Option<ToolAuthorizeResponse> {
    use crate::orchestration_policy::EditPolicy;

    if !matches!(
        policy.edit_policy,
        EditPolicy::ExplicitFileOnly | EditPolicy::SingleFileEdit
    ) {
        return None;
    }
    let intent = req.user_intent.as_deref().unwrap_or("");
    let intent_lower = intent.to_ascii_lowercase();
    if !names_files(&intent_lower) {
        return None;
    }
    let target = edit_target_path(&req.arguments)?;
    let basename = target.rsplit('/').next().unwrap_or(&target);
    if intent_lower.contains(&basename.to_ascii_lowercase()) {
        return None;
    }

    Some(ToolAuthorizeResponse {
        policy_version: TOOL_MEDIATION_POLICY_VERSION,
        decision: "deny",
        reason: "edit_scope_violation",
        capability: ToolCapability::FileEdit.as_str(),
        attempted_tool: req.tool_name.clone(),
        preferred_tool: None,
        replacement: None,
        message: format!(
            "Edit denied: {target} is not named in the request. The current edit policy \
             ({}) limits edits to files the request names.",
            policy.edit_policy.as_str()
        ),
    })
}

fn edit_target_path(arguments: &serde_json::Value) -> Option<String> {
    for key in ["file_path", "path", "filename", "notebook_path"] {
        if let Some(path) = arguments.get(key).and_then(serde_json::Value::as_str) {
            if !path.trim().is_empty() {
                return Some(path.trim().to_string());
            }
        }
    }
    None
}

fn names_files(lower: &str) -> bool {
    const FILE_MARKERS: &[&str] = &[
        ".rs", ".ts", ".js", ".py", ".go", ".md", ".json", ".yaml", ".yml", ".toml", "src/",
        "docs/", "tests/",
    ];
    FILE_MARKERS.iter().any(|marker| lower.contains(marker))
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
