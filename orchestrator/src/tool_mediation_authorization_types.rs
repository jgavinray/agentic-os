use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

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
