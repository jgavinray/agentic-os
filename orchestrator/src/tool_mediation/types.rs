use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub use crate::tool_mediation_authorization_types::{ToolAuthorizeRequest, ToolAuthorizeResponse};

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

pub fn missing_implementation_tool_capabilities(outcome: &ToolMenuOutcome) -> Vec<&'static str> {
    const REQUIRED: &[&str] = &["file_read", "text_search", "file_list", "file_edit"];

    REQUIRED
        .iter()
        .copied()
        .filter(|required| {
            !outcome
                .allowed_tools
                .iter()
                .any(|tool| tool.capability == *required)
        })
        .collect()
}

pub fn policy_requires_implementation_tool_surface(
    policy: &crate::orchestration_policy::OrchestrationPolicy,
) -> bool {
    let edit_policy_requires_files = matches!(
        policy.edit_policy,
        crate::orchestration_policy::EditPolicy::ExplicitFileOnly
            | crate::orchestration_policy::EditPolicy::SingleFileEdit
            | crate::orchestration_policy::EditPolicy::ScopedEdit
            | crate::orchestration_policy::EditPolicy::MultiFileEdit
    );
    let file_edit_allowed = policy
        .allowed_tools
        .contains(&crate::orchestration_policy::ToolCapability::FileEdit);
    let file_edit_blocked = policy
        .blocked_tools
        .contains(&crate::orchestration_policy::ToolCapability::FileEdit);

    edit_policy_requires_files && file_edit_allowed && !file_edit_blocked
}
