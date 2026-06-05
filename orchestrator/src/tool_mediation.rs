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

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use uuid::Uuid;

pub const TOOL_MEDIATION_POLICY_VERSION: &str = "deterministic-v1";

/// Local capability vocabulary for client tools.
///
/// This enum is intentionally smaller and more concrete than
/// `orchestration_policy::ToolCapability`. For example, `TextSearch` and
/// `FileList` are local tool shapes that both map to policy-level `RepoRead`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolPayloadFormat {
    OpenAi,
    Anthropic,
}

impl ToolPayloadFormat {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCapability {
    FileRead,
    TextSearch,
    FileList,
    FileEdit,
    Validation,
    Publishing,
    Shell,
    ShellMutation,
    Unknown,
}

impl ToolCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FileRead => "file_read",
            Self::TextSearch => "text_search",
            Self::FileList => "file_list",
            Self::FileEdit => "file_edit",
            Self::Validation => "validation",
            Self::Publishing => "publishing",
            Self::Shell => "shell",
            Self::ShellMutation => "shell_mutation",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolIntent {
    FileRead,
    TextSearch,
    FileList,
    FileEdit,
    Validation,
    Publishing,
    General,
    Unknown,
}

impl ToolIntent {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FileRead => "file_read",
            Self::TextSearch => "text_search",
            Self::FileList => "file_list",
            Self::FileEdit => "file_edit",
            Self::Validation => "validation",
            Self::Publishing => "publishing",
            Self::General => "general",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ToolSummary {
    /// Client-visible tool name, such as `Read`, `Bash`, or `str_replace_editor`.
    pub name: String,
    /// Bounded local capability label derived from the tool name.
    pub capability: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ToolMenuOutcome {
    /// Tool mediation policy version used for this shaping decision.
    pub policy_version: &'static str,
    /// Whether the request used OpenAI or Anthropic tool schema.
    pub endpoint_format: &'static str,
    /// Deterministic intent inferred from the user text for canonical shaping.
    pub intent: &'static str,
    /// `pass` when nothing changed, `shape` when tools were hidden.
    pub decision: &'static str,
    /// Bounded reason label for metrics and event metadata.
    pub reason: &'static str,
    /// Tools originally offered by the client.
    pub offered_tools: Vec<ToolSummary>,
    /// Tools left visible after canonical and policy shaping.
    pub allowed_tools: Vec<ToolSummary>,
    /// Tools hidden from the forwarded model request.
    pub hidden_tools: Vec<ToolSummary>,
    /// True when a forced tool choice was reset because that tool was hidden.
    pub tool_choice_changed: bool,
}

impl ToolMenuOutcome {
    pub fn metadata(&self) -> Value {
        json!({
            "tool_mediation": self,
        })
    }

    pub fn hidden_capabilities(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.hidden_tools.iter().map(|tool| tool.capability)
    }

    pub fn allowed_capabilities(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.allowed_tools.iter().map(|tool| tool.capability)
    }

    pub fn offered_capabilities(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.offered_tools.iter().map(|tool| tool.capability)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolAuthorizeRequest {
    /// Optional session for durable event/policy persistence.
    pub session_id: Option<String>,
    /// Repository or namespace for classification and event storage.
    pub repo: Option<String>,
    /// Human task label used for event grouping.
    pub task: Option<String>,
    /// Current trajectory id when the client has one.
    pub trajectory_id: Option<Uuid>,
    /// Attempt index inside the current trajectory.
    pub attempt_index: Option<i32>,
    /// Parent event id for lineage.
    pub parent_event_id: Option<Uuid>,
    /// Optional user-facing intent text supplied by the client/hook adapter.
    pub user_intent: Option<String>,
    /// Tool name the client is about to execute.
    pub tool_name: String,
    /// Raw client tool arguments. These are persisted for audit when a session
    /// exists, but only bounded keys are copied into classifier input.
    #[serde(default)]
    pub arguments: Value,
    /// Complete set of tools the client can execute. Used for canonical
    /// replacement suggestions, such as preferring `Read` over `Bash cat`.
    #[serde(default)]
    pub available_tools: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ToolAuthorizeResponse {
    /// Tool mediation policy version used for this authorization decision.
    pub policy_version: &'static str,
    /// `allow` or `deny`.
    pub decision: &'static str,
    /// Bounded reason label for clients, metrics, and event metadata.
    pub reason: &'static str,
    /// Capability inferred for the attempted call.
    pub capability: &'static str,
    /// Original attempted tool name.
    pub attempted_tool: String,
    /// Preferred canonical tool when denial is only a canonical-tool redirect.
    pub preferred_tool: Option<String>,
    /// Optional replacement payload the client can feed back to the model loop.
    pub replacement: Option<Value>,
    /// Human-readable explanation for the model/client loop.
    pub message: String,
}

impl ToolAuthorizeResponse {
    pub fn metadata(&self, request: &ToolAuthorizeRequest) -> Value {
        json!({
            "tool_authorization": {
                "policy_version": self.policy_version,
                "decision": self.decision,
                "reason": self.reason,
                "capability": self.capability,
                "attempted_tool": self.attempted_tool,
                "preferred_tool": self.preferred_tool,
                "replacement": self.replacement,
                "available_tool_count": request.available_tools.len(),
            }
        })
    }
}

pub fn shape_openai_request(req: &mut Value, user_content: &str) -> ToolMenuOutcome {
    shape_openai_request_with_policy(req, user_content, None)
}

pub fn shape_openai_request_with_policy(
    req: &mut Value,
    user_content: &str,
    policy: Option<&crate::orchestration_policy::OrchestrationPolicy>,
) -> ToolMenuOutcome {
    // OpenAI and Anthropic share the same policy logic. Only the tool-name
    // extraction and forced-tool-choice normalization differ by payload format.
    shape_request(req, user_content, ToolPayloadFormat::OpenAi, policy)
}

pub fn shape_anthropic_request(req: &mut Value, user_content: &str) -> ToolMenuOutcome {
    shape_anthropic_request_with_policy(req, user_content, None)
}

pub fn shape_anthropic_request_with_policy(
    req: &mut Value,
    user_content: &str,
    policy: Option<&crate::orchestration_policy::OrchestrationPolicy>,
) -> ToolMenuOutcome {
    shape_request(req, user_content, ToolPayloadFormat::Anthropic, policy)
}

fn shape_request(
    req: &mut Value,
    user_content: &str,
    format: ToolPayloadFormat,
    policy: Option<&crate::orchestration_policy::OrchestrationPolicy>,
) -> ToolMenuOutcome {
    let intent = detect_tool_intent(user_content);
    let Some(tools) = req.get_mut("tools").and_then(Value::as_array_mut) else {
        return outcome(format, intent, "pass", "no_tools", vec![], vec![], false);
    };

    let offered = tool_summaries(tools, format);
    if offered.is_empty() {
        return outcome(format, intent, "pass", "no_tools", vec![], vec![], false);
    }

    // Canonical shaping prefers narrower client tools over shell fallbacks even
    // before orchestration policy is considered. Example: when both `Read` and
    // `Bash` are available for a file-read intent, `Bash` is hidden.
    let hidden_names = hidden_tool_names(intent, &offered);

    // Policy shaping is subtractive. The orchestrator cannot add tools in proxy
    // mode because the client still owns execution; it can only hide offered
    // tools that violate the derived operating envelope.
    let policy_hidden_names: BTreeSet<String> = if let Some(p) = policy {
        offered
            .iter()
            .filter(|tool| {
                let cap = capability_for_tool_name(&tool.name);
                policy_blocks_tool_capability(p, cap) || !policy_allows_tool_capability(p, cap)
            })
            .map(|tool| tool.name.clone())
            .collect()
    } else {
        BTreeSet::new()
    };

    // Merge canonical and policy hidden names so the model sees the intersection
    // of "best available client tool" and "allowed by orchestration policy".
    let mut all_hidden: BTreeSet<String> = hidden_names.clone();
    all_hidden.extend(policy_hidden_names);

    // Determine if we have any policy-driven hidden tools (beyond canonical).
    let policy_hidden_count = if let Some(_p) = policy {
        offered
            .iter()
            .filter(|tool| {
                let cap = capability_for_tool_name(&tool.name);
                // Count tools hidden by policy but not by canonical rules
                !hidden_names.contains(&tool.name)
                    && (policy_blocks_tool_capability(_p, cap)
                        || !policy_allows_tool_capability(_p, cap))
            })
            .count()
    } else {
        0
    };

    // If canonical hiding already hides everything, use canonical path.
    if all_hidden.len() == offered.len() && hidden_names.len() == offered.len() {
        tools.retain(|tool| {
            tool_name(tool, format)
                .map(|name| !all_hidden.contains(&name))
                .unwrap_or(true)
        });
        let tool_choice_changed = normalize_tool_choice(req, format, &all_hidden);
        let allowed = offered
            .iter()
            .filter(|tool| !all_hidden.contains(&tool.name))
            .cloned()
            .collect::<Vec<_>>();
        let hidden = offered
            .iter()
            .filter(|tool| all_hidden.contains(&tool.name))
            .cloned()
            .collect::<Vec<_>>();

        return outcome(
            format,
            intent,
            "shape",
            "prefer_canonical_tool",
            offered,
            allowed,
            tool_choice_changed,
        )
        .with_hidden(hidden);
    }

    // If policy would hide all offered tools, shape to an empty tool menu. That
    // is preferable to leaving an unsafe or out-of-policy fallback visible.
    if let Some(_p) = policy {
        let all_blocked_by_policy = offered.iter().all(|tool| {
            let cap = capability_for_tool_name(&tool.name);
            policy_blocks_tool_capability(_p, cap) || !policy_allows_tool_capability(_p, cap)
        });
        if all_blocked_by_policy {
            // Clear the tools array so the request reflects zero tools.
            tools.clear();
            // Normalize tool_choice against all_hidden before returning.
            let tool_choice_changed = normalize_tool_choice(req, format, &all_hidden);
            let allowed: Vec<ToolSummary> = vec![];
            let hidden = offered.clone();
            return outcome(
                format,
                intent,
                "shape",
                "policy_filtered_all_tools",
                offered,
                allowed,
                tool_choice_changed,
            )
            .with_hidden(hidden);
        }
    }

    // If canonical hiding found nothing and policy found nothing, pass through.
    if hidden_names.is_empty() && policy_hidden_count == 0 {
        return outcome(
            format,
            intent,
            "pass",
            "not_applicable",
            offered.clone(),
            offered,
            false,
        );
    }

    // Apply filtering: hide tools that are in all_hidden.
    tools.retain(|tool| {
        tool_name(tool, format)
            .map(|name| !all_hidden.contains(&name))
            .unwrap_or(true)
    });
    let tool_choice_changed = normalize_tool_choice(req, format, &all_hidden);

    // Determine the reason.
    let reason = if policy_hidden_count > 0 {
        "policy_filtered"
    } else {
        "prefer_canonical_tool"
    };

    let allowed = offered
        .iter()
        .filter(|tool| !all_hidden.contains(&tool.name))
        .cloned()
        .collect::<Vec<_>>();
    let hidden = offered
        .iter()
        .filter(|tool| all_hidden.contains(&tool.name))
        .cloned()
        .collect::<Vec<_>>();

    outcome(
        format,
        intent,
        "shape",
        reason,
        offered,
        allowed,
        tool_choice_changed,
    )
    .with_hidden(hidden)
}

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

fn outcome(
    format: ToolPayloadFormat,
    intent: ToolIntent,
    decision: &'static str,
    reason: &'static str,
    offered_tools: Vec<ToolSummary>,
    allowed_tools: Vec<ToolSummary>,
    tool_choice_changed: bool,
) -> ToolMenuOutcome {
    ToolMenuOutcome {
        policy_version: TOOL_MEDIATION_POLICY_VERSION,
        endpoint_format: format.as_str(),
        intent: intent.as_str(),
        decision,
        reason,
        offered_tools,
        allowed_tools,
        hidden_tools: vec![],
        tool_choice_changed,
    }
}

trait WithHidden {
    fn with_hidden(self, hidden_tools: Vec<ToolSummary>) -> Self;
}

impl WithHidden for ToolMenuOutcome {
    fn with_hidden(mut self, hidden_tools: Vec<ToolSummary>) -> Self {
        self.hidden_tools = hidden_tools;
        self
    }
}

fn hidden_tool_names(intent: ToolIntent, offered: &[ToolSummary]) -> BTreeSet<String> {
    // Canonical hiding only applies when the client already offered the
    // narrower tool. If `Bash` is the only available way to read a file, proxy
    // mode leaves it visible because the orchestrator cannot execute a missing
    // `Read` tool for the client.
    let Some(canonical) = canonical_capability_for_intent(intent) else {
        return BTreeSet::new();
    };
    if !offered
        .iter()
        .any(|tool| tool.capability == canonical.as_str())
    {
        return BTreeSet::new();
    }

    match canonical {
        ToolCapability::FileRead
        | ToolCapability::TextSearch
        | ToolCapability::FileList
        | ToolCapability::ShellMutation => offered
            .iter()
            .filter(|tool| tool.capability == ToolCapability::Shell.as_str())
            .map(|tool| tool.name.clone())
            .collect(),
        _ => BTreeSet::new(),
    }
}

fn canonical_capability_for_intent(intent: ToolIntent) -> Option<ToolCapability> {
    match intent {
        ToolIntent::FileRead => Some(ToolCapability::FileRead),
        ToolIntent::TextSearch => Some(ToolCapability::TextSearch),
        ToolIntent::FileList => Some(ToolCapability::FileList),
        ToolIntent::FileEdit => Some(ToolCapability::FileEdit),
        ToolIntent::Validation => Some(ToolCapability::Validation),
        ToolIntent::Publishing => Some(ToolCapability::Publishing),
        ToolIntent::General | ToolIntent::Unknown => None,
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
fn policy_allows_tool_capability(
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
fn policy_blocks_tool_capability(
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

fn tool_summaries(tools: &[Value], format: ToolPayloadFormat) -> Vec<ToolSummary> {
    tools
        .iter()
        .filter_map(|tool| {
            tool_name(tool, format).map(|name| ToolSummary {
                capability: capability_for_tool_name(&name).as_str(),
                name,
            })
        })
        .collect()
}

fn tool_name(tool: &Value, format: ToolPayloadFormat) -> Option<String> {
    match format {
        ToolPayloadFormat::OpenAi => tool
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .or_else(|| tool.get("name").and_then(Value::as_str))
            .map(str::to_string),
        ToolPayloadFormat::Anthropic => {
            tool.get("name").and_then(Value::as_str).map(str::to_string)
        }
    }
}

fn normalize_tool_choice(
    req: &mut Value,
    format: ToolPayloadFormat,
    hidden_names: &BTreeSet<String>,
) -> bool {
    let Some(choice) = req.get("tool_choice") else {
        return false;
    };
    let hidden = match format {
        ToolPayloadFormat::OpenAi => {
            if let Some(name) = choice
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
            {
                hidden_names.contains(name)
            } else {
                choice
                    .as_str()
                    .map(|name| hidden_names.contains(name))
                    .unwrap_or(false)
            }
        }
        ToolPayloadFormat::Anthropic => choice
            .get("name")
            .and_then(Value::as_str)
            .map(|name| hidden_names.contains(name))
            .unwrap_or(false),
    };
    if !hidden {
        return false;
    }

    req["tool_choice"] = match format {
        ToolPayloadFormat::OpenAi => Value::String("auto".to_string()),
        ToolPayloadFormat::Anthropic => json!({"type": "auto"}),
    };
    true
}

pub fn detect_tool_intent(text: &str) -> ToolIntent {
    let lower = text.to_ascii_lowercase();
    if lower.trim().is_empty() {
        return ToolIntent::Unknown;
    }
    if contains_any(
        &lower,
        &["commit", "push", "pull request", "open a pr", "create pr"],
    ) {
        ToolIntent::Publishing
    } else if contains_any(
        &lower,
        &[
            "run test",
            "run the test",
            "cargo test",
            "npm test",
            "lint",
            "typecheck",
            "type check",
        ],
    ) {
        ToolIntent::Validation
    } else if contains_any(
        &lower,
        &[
            "edit ", "modify ", "change ", "update ", "fix ", "patch ", "write to",
        ],
    ) {
        ToolIntent::FileEdit
    } else if contains_any(
        &lower,
        &[
            "grep",
            "search",
            "find usages",
            "find references",
            "look for",
        ],
    ) {
        ToolIntent::TextSearch
    } else if contains_any(
        &lower,
        &["list files", "show files", "tree", "directory listing"],
    ) {
        ToolIntent::FileList
    } else if (contains_any(
        &lower,
        &[
            "read ", "show me", "show the", "open ", "view ", "inspect ", "look at", "cat ",
        ],
    ) && (looks_like_file_reference(&lower) || lower.contains("file")))
        || looks_like_file_reference(&lower)
            && contains_any(&lower, &["what is in", "what's in", "contents of"])
    {
        ToolIntent::FileRead
    } else {
        ToolIntent::General
    }
}

fn capability_for_tool_name(name: &str) -> ToolCapability {
    let lower = name.to_ascii_lowercase();
    let normalized = lower.replace('-', "_");
    if matches!(
        normalized.as_str(),
        "read" | "read_file" | "file_read" | "view" | "open_file"
    ) || normalized.ends_with("__read_file")
    {
        ToolCapability::FileRead
    } else if matches!(
        normalized.as_str(),
        "grep" | "rg" | "ripgrep" | "search" | "text_search" | "find_text"
    ) || normalized.ends_with("__search")
    {
        ToolCapability::TextSearch
    } else if matches!(
        normalized.as_str(),
        "glob" | "ls" | "list" | "list_files" | "file_list" | "find_files"
    ) || normalized.ends_with("__list_files")
    {
        ToolCapability::FileList
    } else if matches!(
        normalized.as_str(),
        "edit" | "write" | "write_file" | "apply_patch" | "patch" | "str_replace_editor"
    ) || normalized.ends_with("__write_file")
    {
        ToolCapability::FileEdit
    } else if matches!(
        normalized.as_str(),
        "test" | "run_tests" | "lint" | "typecheck" | "type_check" | "compile"
    ) {
        ToolCapability::Validation
    } else if matches!(
        normalized.as_str(),
        "git" | "commit" | "push" | "pull_request" | "pr" | "create_pr"
    ) {
        ToolCapability::Publishing
    } else if matches!(
        normalized.as_str(),
        "bash" | "shell" | "terminal" | "run_command" | "execute_command"
    ) {
        ToolCapability::Shell
    } else {
        ToolCapability::Unknown
    }
}

fn command_capability(arguments: &Value) -> ToolCapability {
    let Some(command) = command_from_arguments(arguments) else {
        return ToolCapability::Shell;
    };
    let lower = command.trim().to_ascii_lowercase();
    if is_shell_file_read(&lower) {
        ToolCapability::FileRead
    } else if is_shell_text_search(&lower) {
        ToolCapability::TextSearch
    } else if is_shell_file_list(&lower) {
        ToolCapability::FileList
    } else {
        // Unknown shell means mutation, not "safe shell". This is intentionally
        // conservative: read-only policy must not authorize arbitrary shell just
        // because the command is unfamiliar.
        ToolCapability::ShellMutation
    }
}

fn command_from_arguments(arguments: &Value) -> Option<String> {
    // Different clients serialize shell commands differently. Keep this list
    // deliberately small and explicit so authorization remains predictable.
    arguments
        .as_str()
        .map(str::to_string)
        .or_else(|| {
            arguments
                .get("command")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .or_else(|| {
            arguments
                .get("cmd")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .or_else(|| {
            arguments
                .get("input")
                .and_then(|input| input.get("command"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

fn is_shell_file_read(command: &str) -> bool {
    starts_with_command(command, "cat")
        || starts_with_command(command, "head")
        || starts_with_command(command, "tail")
        || command.starts_with("sed -n ")
        || command.starts_with("awk ")
}

fn is_shell_text_search(command: &str) -> bool {
    starts_with_command(command, "grep")
        || starts_with_command(command, "rg")
        || starts_with_command(command, "ripgrep")
}

fn is_shell_file_list(command: &str) -> bool {
    starts_with_command(command, "ls")
        || starts_with_command(command, "find")
        || starts_with_command(command, "tree")
}

fn starts_with_command(command: &str, binary: &str) -> bool {
    command == binary || command.starts_with(&format!("{binary} "))
}

fn replacement_for(
    capability: ToolCapability,
    preferred_tool: &str,
    arguments: &Value,
) -> Option<Value> {
    match capability {
        ToolCapability::FileRead => command_from_arguments(arguments)
            .and_then(|command| path_from_read_command(&command))
            .map(|path| json!({"tool_name": preferred_tool, "arguments": {"path": path}})),
        _ => None,
    }
}

fn path_from_read_command(command: &str) -> Option<String> {
    let parts = command
        .split_whitespace()
        .map(|part| part.trim_matches(|c| c == '\'' || c == '"'))
        .filter(|part| {
            !part.is_empty()
                && !part.starts_with('-')
                && *part != "cat"
                && *part != "head"
                && *part != "tail"
                && *part != "sed"
                && *part != "awk"
                && !is_sed_range(part)
        })
        .collect::<Vec<_>>();
    parts.last().map(|part| (*part).to_string())
}

fn is_sed_range(part: &str) -> bool {
    part.ends_with('p')
        && part
            .trim_end_matches('p')
            .chars()
            .all(|c| c.is_ascii_digit() || c == ',' || c == '$')
}

fn looks_like_file_reference(value: &str) -> bool {
    value.contains('/')
        || value.contains(".rs")
        || value.contains(".ts")
        || value.contains(".js")
        || value.contains(".py")
        || value.contains(".md")
        || value.contains(".json")
        || value.contains(".yaml")
        || value.contains(".yml")
        || value.contains(".toml")
}

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

pub fn bounded_decision(value: &str) -> &'static str {
    match value {
        "allow" => "allow",
        "deny" => "deny",
        "shape" => "shape",
        "pass" => "pass",
        _ => "unknown",
    }
}

pub fn bounded_reason(value: &str) -> &'static str {
    match value {
        "prefer_canonical_tool" => "prefer_canonical_tool",
        "no_tools" => "no_tools",
        "not_applicable" => "not_applicable",
        "disabled" => "disabled",
        "policy_filtered" => "policy_filtered",
        "policy_would_hide_all_tools" => "policy_would_hide_all_tools",
        "policy_filtered_all_tools" => "policy_filtered_all_tools",
        "policy_blocked_tool" => "policy_blocked_tool",
        "policy_tool_not_allowed" => "policy_tool_not_allowed",
        _ => "unknown",
    }
}

pub fn bounded_capability(value: &str) -> &'static str {
    match value {
        "file_read" => "file_read",
        "text_search" => "text_search",
        "file_list" => "file_list",
        "file_edit" => "file_edit",
        "validation" => "validation",
        "publishing" => "publishing",
        "shell" => "shell",
        "shell_mutation" => "shell_mutation",
        "unknown" => "unknown",
        _ => "unknown",
    }
}

pub fn bounded_tool_action(value: &str) -> &'static str {
    match value {
        "offered" => "offered",
        "allowed" => "allowed",
        "hidden" => "hidden",
        "denied" => "denied",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn openai_file_read_hides_shell_when_read_exists() {
        let mut req = json!({
            "messages": [{"role": "user", "content": "Read README.md"}],
            "tools": [
                {"type": "function", "function": {"name": "Read"}},
                {"type": "function", "function": {"name": "Bash"}}
            ],
            "tool_choice": {"type": "function", "function": {"name": "Bash"}}
        });

        let outcome = shape_openai_request(&mut req, "Read README.md");

        assert_eq!(outcome.decision, "shape");
        assert_eq!(outcome.reason, "prefer_canonical_tool");
        assert_eq!(outcome.hidden_tools[0].name, "Bash");
        assert_eq!(req["tools"].as_array().unwrap().len(), 1);
        assert_eq!(req["tools"][0]["function"]["name"], "Read");
        assert_eq!(req["tool_choice"], "auto");
    }

    #[test]
    fn anthropic_file_read_hides_shell_when_read_exists() {
        let mut req = json!({
            "messages": [],
            "tools": [
                {"name": "read", "input_schema": {}},
                {"name": "bash", "input_schema": {}}
            ],
            "tool_choice": {"type": "tool", "name": "bash"}
        });

        let outcome = shape_anthropic_request(&mut req, "show the file docs/TRAJECTORIES.md");

        assert_eq!(outcome.decision, "shape");
        assert_eq!(outcome.hidden_tools[0].name, "bash");
        assert_eq!(req["tools"].as_array().unwrap().len(), 1);
        assert_eq!(req["tools"][0]["name"], "read");
        assert_eq!(req["tool_choice"], json!({"type": "auto"}));
    }

    #[test]
    fn shell_stays_available_when_no_canonical_tool_exists() {
        let mut req = json!({
            "messages": [],
            "tools": [
                {"type": "function", "function": {"name": "Bash"}}
            ]
        });

        let outcome = shape_openai_request(&mut req, "Read README.md");

        assert_eq!(outcome.decision, "pass");
        assert_eq!(outcome.reason, "not_applicable");
        assert_eq!(req["tools"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn authorization_denies_shell_file_read_with_read_available() {
        let req = ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("read README.md".to_string()),
            tool_name: "Bash".to_string(),
            arguments: json!({"command": "cat README.md"}),
            available_tools: vec!["Read".to_string(), "Bash".to_string()],
        };

        let response = authorize_tool_call(&req, true);

        assert_eq!(response.decision, "deny");
        assert_eq!(response.reason, "prefer_canonical_tool");
        assert_eq!(response.preferred_tool.as_deref(), Some("Read"));
        assert_eq!(
            response.replacement.unwrap(),
            json!({"tool_name": "Read", "arguments": {"path": "README.md"}})
        );
    }

    #[test]
    fn authorization_allows_shell_file_read_without_read_available() {
        let req = ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("read README.md".to_string()),
            tool_name: "Bash".to_string(),
            arguments: json!({"command": "cat README.md"}),
            available_tools: vec!["Bash".to_string()],
        };

        let response = authorize_tool_call(&req, true);

        assert_eq!(response.decision, "allow");
        assert_eq!(response.reason, "not_applicable");
    }

    #[test]
    fn shape_openai_request_with_policy_matches_shape_openai_request() {
        let mut req1 = json!({
            "messages": [{"role": "user", "content": "Read README.md"}],
            "tools": [
                {"type": "function", "function": {"name": "Read"}},
                {"type": "function", "function": {"name": "Bash"}}
            ],
            "tool_choice": {"type": "function", "function": {"name": "Bash"}}
        });
        let mut req2 = serde_json::to_value(&req1).unwrap();

        let outcome1 = shape_openai_request(&mut req1, "Read README.md");
        let outcome2 = shape_openai_request_with_policy(&mut req2, "Read README.md", None);

        assert_eq!(outcome1.decision, outcome2.decision);
        assert_eq!(outcome1.reason, outcome2.reason);
        assert_eq!(outcome1.hidden_tools.len(), outcome2.hidden_tools.len());
        assert_eq!(outcome1.hidden_tools[0].name, outcome2.hidden_tools[0].name);
    }

    #[test]
    fn implementation_policy_hides_broad_and_unknown_tools() {
        let text = "Implement the request classifier change in src/request_classification.rs";
        let classification = crate::request_classification::classify_request_text(
            "agentic-os",
            "session-a",
            text,
            None,
            "user_message",
        );
        let policy =
            crate::orchestration_policy::derive_orchestration_policy(&classification, text, false);
        let mut req = json!({
            "messages": [],
            "tools": [
                {"type": "function", "function": {"name": "Read"}},
                {"type": "function", "function": {"name": "Grep"}},
                {"type": "function", "function": {"name": "Edit"}},
                {"type": "function", "function": {"name": "Write"}},
                {"type": "function", "function": {"name": "Bash"}},
                {"type": "function", "function": {"name": "MultiEdit"}},
                {"type": "function", "function": {"name": "Delete"}},
                {"type": "function", "function": {"name": "CreatePR"}}
            ]
        });

        let outcome = shape_openai_request_with_policy(&mut req, text, Some(&policy));
        let allowed = req["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["function"]["name"].as_str().unwrap())
            .collect::<Vec<_>>();
        let hidden = outcome
            .hidden_tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(classification.intent.as_str(), "implement");
        assert_eq!(allowed, vec!["Read", "Grep", "Edit", "Write"]);
        assert!(hidden.contains(&"Bash"));
        assert!(hidden.contains(&"MultiEdit"));
        assert!(hidden.contains(&"Delete"));
        assert!(hidden.contains(&"CreatePR"));
    }

    #[test]
    fn authorize_tool_call_with_policy_matches_authorize_tool_call() {
        let req = ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("read README.md".to_string()),
            tool_name: "Bash".to_string(),
            arguments: json!({"command": "cat README.md"}),
            available_tools: vec!["Read".to_string(), "Bash".to_string()],
        };

        let response1 = authorize_tool_call(&req, true);
        let response2 = authorize_tool_call_with_policy(&req, true, None);

        assert_eq!(response1.decision, response2.decision);
        assert_eq!(response1.reason, response2.reason);
        assert_eq!(response1.preferred_tool, response2.preferred_tool);
        assert_eq!(response1.replacement, response2.replacement);
    }

    // -----------------------------------------------------------------------
    // Policy mapping helpers
    // -----------------------------------------------------------------------

    #[test]
    fn policy_allows_file_read_when_repo_read_allowed() {
        use crate::orchestration_policy::{
            EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
        };

        let policy = crate::orchestration_policy::OrchestrationPolicy {
            context_sources: vec![],
            allowed_tools: vec![PolicyCap::RepoRead],
            required_tools: vec![],
            blocked_tools: vec![],
            edit_policy: EditPolicy::ReadOnly,
            validation_policy: crate::orchestration_policy::ValidationPolicy::None,
            git_policy: GitPolicy::NoGitChanges,
            runtime_policy: RuntimePolicy::NoRestart,
            scope_policy: vec![],
            prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
            risk_policy: vec![],
        };

        assert!(policy_allows_tool_capability(
            &policy,
            ToolCapability::FileRead
        ));
    }

    #[test]
    fn policy_blocks_text_search_when_repo_read_blocked() {
        use crate::orchestration_policy::{
            EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
        };

        let policy = crate::orchestration_policy::OrchestrationPolicy {
            context_sources: vec![],
            allowed_tools: vec![],
            required_tools: vec![],
            blocked_tools: vec![PolicyCap::RepoRead],
            edit_policy: EditPolicy::ReadOnly,
            validation_policy: crate::orchestration_policy::ValidationPolicy::None,
            git_policy: GitPolicy::NoGitChanges,
            runtime_policy: RuntimePolicy::NoRestart,
            scope_policy: vec![],
            prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
            risk_policy: vec![],
        };

        assert!(policy_blocks_tool_capability(
            &policy,
            ToolCapability::TextSearch
        ));
    }

    #[test]
    fn policy_maps_validation_to_shell_read() {
        use crate::orchestration_policy::{
            EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
        };

        let policy = crate::orchestration_policy::OrchestrationPolicy {
            context_sources: vec![],
            allowed_tools: vec![PolicyCap::ShellRead],
            required_tools: vec![],
            blocked_tools: vec![],
            edit_policy: EditPolicy::ReadOnly,
            validation_policy: crate::orchestration_policy::ValidationPolicy::None,
            git_policy: GitPolicy::NoGitChanges,
            runtime_policy: RuntimePolicy::NoRestart,
            scope_policy: vec![],
            prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
            risk_policy: vec![],
        };

        assert!(policy_allows_tool_capability(
            &policy,
            ToolCapability::Validation
        ));
    }

    #[test]
    fn policy_maps_publishing_to_git_write() {
        use crate::orchestration_policy::{
            EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
        };

        let policy = crate::orchestration_policy::OrchestrationPolicy {
            context_sources: vec![],
            allowed_tools: vec![PolicyCap::GitWrite],
            required_tools: vec![],
            blocked_tools: vec![],
            edit_policy: EditPolicy::ReadOnly,
            validation_policy: crate::orchestration_policy::ValidationPolicy::None,
            git_policy: GitPolicy::NoGitChanges,
            runtime_policy: RuntimePolicy::NoRestart,
            scope_policy: vec![],
            prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
            risk_policy: vec![],
        };

        assert!(policy_allows_tool_capability(
            &policy,
            ToolCapability::Publishing
        ));
    }

    #[test]
    fn policy_unknown_is_not_allowed_and_not_blocked() {
        use crate::orchestration_policy::{
            EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
        };

        let policy = crate::orchestration_policy::OrchestrationPolicy {
            context_sources: vec![],
            allowed_tools: vec![PolicyCap::FileRead],
            required_tools: vec![],
            blocked_tools: vec![PolicyCap::FileRead],
            edit_policy: EditPolicy::ReadOnly,
            validation_policy: crate::orchestration_policy::ValidationPolicy::None,
            git_policy: GitPolicy::NoGitChanges,
            runtime_policy: RuntimePolicy::NoRestart,
            scope_policy: vec![],
            prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
            risk_policy: vec![],
        };

        assert!(!policy_allows_tool_capability(
            &policy,
            ToolCapability::Unknown
        ));
        assert!(!policy_blocks_tool_capability(
            &policy,
            ToolCapability::Unknown
        ));
    }

    // -----------------------------------------------------------------------
    // Policy enforcement in authorize_tool_call_with_policy
    // -----------------------------------------------------------------------

    #[test]
    fn policy_blocks_file_read_denies_read() {
        use crate::orchestration_policy::{
            EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
        };

        let policy = crate::orchestration_policy::OrchestrationPolicy {
            context_sources: vec![],
            allowed_tools: vec![],
            required_tools: vec![],
            blocked_tools: vec![PolicyCap::FileRead],
            edit_policy: EditPolicy::ReadOnly,
            validation_policy: crate::orchestration_policy::ValidationPolicy::None,
            git_policy: GitPolicy::NoGitChanges,
            runtime_policy: RuntimePolicy::NoRestart,
            scope_policy: vec![],
            prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
            risk_policy: vec![],
        };

        let req = ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("read README.md".to_string()),
            tool_name: "Read".to_string(),
            arguments: json!({}),
            available_tools: vec!["Read".to_string()],
        };

        let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

        assert_eq!(response.decision, "deny");
        assert_eq!(response.reason, "policy_blocked_tool");
        assert_eq!(response.capability, "file_read");
        assert_eq!(response.attempted_tool, "Read");
        assert_eq!(response.preferred_tool, None);
        assert_eq!(
            response.message,
            "Tool call denied by orchestration policy."
        );
    }

    #[test]
    fn policy_not_allowing_shell_read_denies_bash() {
        use crate::orchestration_policy::{EditPolicy, GitPolicy, RuntimePolicy};

        // Empty allowed_tools and empty blocked_tools means nothing is allowed.
        let policy = crate::orchestration_policy::OrchestrationPolicy {
            context_sources: vec![],
            allowed_tools: vec![],
            required_tools: vec![],
            blocked_tools: vec![],
            edit_policy: EditPolicy::ReadOnly,
            validation_policy: crate::orchestration_policy::ValidationPolicy::None,
            git_policy: GitPolicy::NoGitChanges,
            runtime_policy: RuntimePolicy::NoRestart,
            scope_policy: vec![],
            prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
            risk_policy: vec![],
        };

        let req = ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("run a command".to_string()),
            tool_name: "Bash".to_string(),
            arguments: json!({"command": "echo hello"}),
            available_tools: vec!["Bash".to_string()],
        };

        let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

        assert_eq!(response.decision, "deny");
        assert_eq!(response.reason, "policy_tool_not_allowed");
        assert_eq!(response.capability, "shell_mutation");
        assert_eq!(response.attempted_tool, "Bash");
        assert_eq!(response.preferred_tool, None);
        assert_eq!(
            response.message,
            "Tool call is not allowed by orchestration policy."
        );
    }

    #[test]
    fn policy_shell_read_does_not_allow_generic_bash() {
        use crate::orchestration_policy::{
            EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
        };

        let policy = crate::orchestration_policy::OrchestrationPolicy {
            context_sources: vec![],
            allowed_tools: vec![PolicyCap::ShellRead],
            required_tools: vec![],
            blocked_tools: vec![],
            edit_policy: EditPolicy::ReadOnly,
            validation_policy: crate::orchestration_policy::ValidationPolicy::None,
            git_policy: GitPolicy::NoGitChanges,
            runtime_policy: RuntimePolicy::NoRestart,
            scope_policy: vec![],
            prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
            risk_policy: vec![],
        };

        let req = ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("run a command".to_string()),
            tool_name: "Bash".to_string(),
            arguments: json!({"command": "echo hello"}),
            available_tools: vec!["Bash".to_string()],
        };

        let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

        assert_eq!(response.decision, "deny");
        assert_eq!(response.reason, "policy_tool_not_allowed");
        assert_eq!(response.capability, "shell_mutation");
        assert_eq!(response.attempted_tool, "Bash");
        assert_eq!(response.preferred_tool, None);
        assert_eq!(
            response.message,
            "Tool call is not allowed by orchestration policy."
        );
    }

    #[test]
    fn policy_allows_file_read_but_canonical_read_exists() {
        use crate::orchestration_policy::{
            EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
        };

        let policy = crate::orchestration_policy::OrchestrationPolicy {
            context_sources: vec![],
            allowed_tools: vec![PolicyCap::FileRead],
            required_tools: vec![],
            blocked_tools: vec![],
            edit_policy: EditPolicy::ReadOnly,
            validation_policy: crate::orchestration_policy::ValidationPolicy::None,
            git_policy: GitPolicy::NoGitChanges,
            runtime_policy: RuntimePolicy::NoRestart,
            scope_policy: vec![],
            prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
            risk_policy: vec![],
        };

        let req = ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("read README.md".to_string()),
            tool_name: "Bash".to_string(),
            arguments: json!({"command": "cat README.md"}),
            available_tools: vec!["Read".to_string(), "Bash".to_string()],
        };

        let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

        // Policy allows FileRead, but canonical-tool preference still applies.
        assert_eq!(response.decision, "deny");
        assert_eq!(response.reason, "prefer_canonical_tool");
        assert_eq!(response.capability, "file_read");
        assert_eq!(response.preferred_tool.as_deref(), Some("Read"));
    }

    // -----------------------------------------------------------------------
    // policy_filtered_all_tools — OpenAI
    // -----------------------------------------------------------------------

    #[test]
    fn openai_bash_only_with_empty_policy_hides_all() {
        use crate::orchestration_policy::{EditPolicy, GitPolicy, RuntimePolicy};

        let policy = crate::orchestration_policy::OrchestrationPolicy {
            context_sources: vec![],
            allowed_tools: vec![],
            required_tools: vec![],
            blocked_tools: vec![],
            edit_policy: EditPolicy::ReadOnly,
            validation_policy: crate::orchestration_policy::ValidationPolicy::None,
            git_policy: GitPolicy::NoGitChanges,
            runtime_policy: RuntimePolicy::NoRestart,
            scope_policy: vec![],
            prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
            risk_policy: vec![],
        };

        let mut req = json!({
            "messages": [{"role": "user", "content": "run a command"}],
            "tools": [
                {"type": "function", "function": {"name": "Bash"}}
            ],
            "tool_choice": {"type": "function", "function": {"name": "Bash"}}
        });

        let outcome = shape_openai_request_with_policy(&mut req, "run a command", Some(&policy));

        assert_eq!(outcome.decision, "shape");
        assert_eq!(outcome.reason, "policy_filtered_all_tools");
        assert_eq!(req["tools"].as_array().unwrap().len(), 0);
        assert!(outcome.allowed_tools.is_empty());
        assert_eq!(outcome.hidden_tools.len(), 1);
        assert_eq!(outcome.hidden_tools[0].name, "Bash");
    }

    #[test]
    fn openai_bash_only_with_empty_policy_tool_choice_auto() {
        use crate::orchestration_policy::{EditPolicy, GitPolicy, RuntimePolicy};

        let policy = crate::orchestration_policy::OrchestrationPolicy {
            context_sources: vec![],
            allowed_tools: vec![],
            required_tools: vec![],
            blocked_tools: vec![],
            edit_policy: EditPolicy::ReadOnly,
            validation_policy: crate::orchestration_policy::ValidationPolicy::None,
            git_policy: GitPolicy::NoGitChanges,
            runtime_policy: RuntimePolicy::NoRestart,
            scope_policy: vec![],
            prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
            risk_policy: vec![],
        };

        let mut req = json!({
            "messages": [{"role": "user", "content": "run a command"}],
            "tools": [
                {"type": "function", "function": {"name": "Bash"}}
            ],
            "tool_choice": {"type": "function", "function": {"name": "Bash"}}
        });

        let outcome = shape_openai_request_with_policy(&mut req, "run a command", Some(&policy));

        assert_eq!(req["tool_choice"], "auto");
        assert!(outcome.tool_choice_changed);
    }

    // -----------------------------------------------------------------------
    // policy_filtered_all_tools — Anthropic
    // -----------------------------------------------------------------------

    #[test]
    fn anthropic_bash_only_with_empty_policy_hides_all() {
        use crate::orchestration_policy::{EditPolicy, GitPolicy, RuntimePolicy};

        let policy = crate::orchestration_policy::OrchestrationPolicy {
            context_sources: vec![],
            allowed_tools: vec![],
            required_tools: vec![],
            blocked_tools: vec![],
            edit_policy: EditPolicy::ReadOnly,
            validation_policy: crate::orchestration_policy::ValidationPolicy::None,
            git_policy: GitPolicy::NoGitChanges,
            runtime_policy: RuntimePolicy::NoRestart,
            scope_policy: vec![],
            prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
            risk_policy: vec![],
        };

        let mut req = json!({
            "messages": [],
            "tools": [
                {"name": "bash", "input_schema": {}}
            ],
            "tool_choice": {"type": "tool", "name": "bash"}
        });

        let outcome = shape_anthropic_request_with_policy(&mut req, "run a command", Some(&policy));

        assert_eq!(outcome.decision, "shape");
        assert_eq!(outcome.reason, "policy_filtered_all_tools");
        assert_eq!(req["tools"].as_array().unwrap().len(), 0);
        assert!(outcome.allowed_tools.is_empty());
        assert_eq!(outcome.hidden_tools.len(), 1);
        assert_eq!(outcome.hidden_tools[0].name, "bash");
    }

    // -----------------------------------------------------------------------
    // ShellMutation policy tests
    // -----------------------------------------------------------------------

    #[test]
    fn policy_blocks_shell_mutation_denies_mutating_bash() {
        use crate::orchestration_policy::{
            EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
        };

        let policy = crate::orchestration_policy::OrchestrationPolicy {
            context_sources: vec![],
            allowed_tools: vec![],
            required_tools: vec![],
            blocked_tools: vec![PolicyCap::ShellMutation],
            edit_policy: EditPolicy::ReadOnly,
            validation_policy: crate::orchestration_policy::ValidationPolicy::None,
            git_policy: GitPolicy::NoGitChanges,
            runtime_policy: RuntimePolicy::NoRestart,
            scope_policy: vec![],
            prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
            risk_policy: vec![],
        };

        let req = ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("remove a file".to_string()),
            tool_name: "Bash".to_string(),
            arguments: json!({"command": "rm -rf /tmp/example"}),
            available_tools: vec!["Bash".to_string()],
        };

        let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

        assert_eq!(response.decision, "deny");
        assert_eq!(response.reason, "policy_blocked_tool");
        assert_eq!(response.capability, "shell_mutation");
        assert_eq!(response.attempted_tool, "Bash");
        assert_eq!(response.preferred_tool, None);
        assert_eq!(
            response.message,
            "Tool call denied by orchestration policy."
        );
    }

    #[test]
    fn policy_shell_read_only_does_not_allow_mutating_bash() {
        use crate::orchestration_policy::{
            EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
        };

        let policy = crate::orchestration_policy::OrchestrationPolicy {
            context_sources: vec![],
            allowed_tools: vec![PolicyCap::ShellRead],
            required_tools: vec![],
            blocked_tools: vec![],
            edit_policy: EditPolicy::ReadOnly,
            validation_policy: crate::orchestration_policy::ValidationPolicy::None,
            git_policy: GitPolicy::NoGitChanges,
            runtime_policy: RuntimePolicy::NoRestart,
            scope_policy: vec![],
            prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
            risk_policy: vec![],
        };

        let req = ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("remove a file".to_string()),
            tool_name: "Bash".to_string(),
            arguments: json!({"command": "rm -rf /tmp/example"}),
            available_tools: vec!["Bash".to_string()],
        };

        let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

        assert_eq!(response.decision, "deny");
        assert_eq!(response.reason, "policy_tool_not_allowed");
        assert_eq!(response.capability, "shell_mutation");
        assert_eq!(response.attempted_tool, "Bash");
        assert_eq!(response.preferred_tool, None);
        assert_eq!(
            response.message,
            "Tool call is not allowed by orchestration policy."
        );
    }

    #[test]
    fn policy_repo_read_allows_rg_shell_as_text_search() {
        use crate::orchestration_policy::{
            EditPolicy, GitPolicy, RuntimePolicy, ToolCapability as PolicyCap,
        };

        let policy = crate::orchestration_policy::OrchestrationPolicy {
            context_sources: vec![],
            allowed_tools: vec![PolicyCap::RepoRead],
            required_tools: vec![],
            blocked_tools: vec![],
            edit_policy: EditPolicy::ReadOnly,
            validation_policy: crate::orchestration_policy::ValidationPolicy::None,
            git_policy: GitPolicy::NoGitChanges,
            runtime_policy: RuntimePolicy::NoRestart,
            scope_policy: vec![],
            prompt_refinement_policy: crate::orchestration_policy::PromptRefinementPolicy::None,
            risk_policy: vec![],
        };

        let req = ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("search for pattern".to_string()),
            tool_name: "Bash".to_string(),
            arguments: json!({"command": "rg pattern src"}),
            available_tools: vec!["Bash".to_string()],
        };

        let response = authorize_tool_call_with_policy(&req, true, Some(&policy));

        assert_eq!(response.decision, "allow");
        assert_eq!(response.capability, "text_search");
        assert_ne!(response.capability, "shell_mutation");
    }
}
