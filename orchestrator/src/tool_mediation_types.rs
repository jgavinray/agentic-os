use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
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
