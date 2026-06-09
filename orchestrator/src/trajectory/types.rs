use uuid::Uuid;

pub const DEFAULT_TRAJECTORY_IDLE_TIMEOUT_SEC: u64 = 600;

pub const EVENT_TYPE_CONTEXT_PACK: &str = "context_pack";
pub const EVENT_TYPE_TRAJECTORY_RESULT: &str = "trajectory_result";

pub const EVENT_ROLES: [&str; 10] = [
    "request",
    "context_pack",
    "model_response",
    "tool_call",
    "tool_result",
    "validation",
    "patch",
    "failure",
    "remediation",
    "trajectory_result",
];

pub const FINAL_STATUSES: [&str; 4] = ["succeeded", "abandoned", "unresolved", "reverted"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventRole {
    Request,
    ContextPack,
    ModelResponse,
    ToolCall,
    ToolResult,
    Validation,
    Patch,
    Failure,
    Remediation,
    TrajectoryResult,
}

impl EventRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::ContextPack => "context_pack",
            Self::ModelResponse => "model_response",
            Self::ToolCall => "tool_call",
            Self::ToolResult => "tool_result",
            Self::Validation => "validation",
            Self::Patch => "patch",
            Self::Failure => "failure",
            Self::Remediation => "remediation",
            Self::TrajectoryResult => "trajectory_result",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(value: &str) -> Option<Self> {
        Some(match value {
            "request" => Self::Request,
            "context_pack" => Self::ContextPack,
            "model_response" => Self::ModelResponse,
            "tool_call" => Self::ToolCall,
            "tool_result" => Self::ToolResult,
            "validation" => Self::Validation,
            "patch" => Self::Patch,
            "failure" => Self::Failure,
            "remediation" => Self::Remediation,
            "trajectory_result" => Self::TrajectoryResult,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinalStatus {
    Succeeded,
    Abandoned,
    Unresolved,
    Reverted,
}

impl FinalStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Abandoned => "abandoned",
            Self::Unresolved => "unresolved",
            Self::Reverted => "reverted",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryReason {
    NewUserMessage,
    IdleTimeout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrajectoryContext {
    pub trajectory_id: Uuid,
    pub attempt_index: i32,
}
