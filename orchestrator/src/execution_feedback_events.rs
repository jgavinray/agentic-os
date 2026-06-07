use crate::db::AgentEvent;
use crate::execution_feedback_fingerprints::{
    inline_signature_payload, outcome_raw_text_from_payload, summarize_text,
};
use crate::execution_feedback_payloads::event_summary;
use serde_json::{json, Value};
use uuid::Uuid;

// Keep the public event vocabulary centralized so storage, metrics, docs, and
// tests do not drift into subtly different names for the same artifact.
pub const EVENT_TYPE_TOOL_RESULT: &str = "tool_result";
pub const EVENT_TYPE_COMPILE_RESULT: &str = "compile_result";
pub const EVENT_TYPE_TEST_RESULT: &str = "test_result";
pub const EVENT_TYPE_LINT_RESULT: &str = "lint_result";
pub const EVENT_TYPE_VALIDATION_RESULT: &str = "validation_result";
pub const EVENT_TYPE_PATCH_RESULT: &str = "patch_result";
pub const EVENT_TYPE_REMEDIATION: &str = "remediation";

pub const EXECUTION_EVENT_TYPES: [&str; 7] = [
    EVENT_TYPE_TOOL_RESULT,
    EVENT_TYPE_COMPILE_RESULT,
    EVENT_TYPE_TEST_RESULT,
    EVENT_TYPE_LINT_RESULT,
    EVENT_TYPE_VALIDATION_RESULT,
    EVENT_TYPE_PATCH_RESULT,
    EVENT_TYPE_REMEDIATION,
];

pub const FAILURE_OUTCOME_EVENT_TYPES: [&str; 5] = [
    EVENT_TYPE_TOOL_RESULT,
    EVENT_TYPE_COMPILE_RESULT,
    EVENT_TYPE_TEST_RESULT,
    EVENT_TYPE_LINT_RESULT,
    EVENT_TYPE_VALIDATION_RESULT,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionEventKind {
    ToolResult,
    CompileResult,
    TestResult,
    LintResult,
    ValidationResult,
    PatchResult,
    Remediation,
}

impl ExecutionEventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ToolResult => EVENT_TYPE_TOOL_RESULT,
            Self::CompileResult => EVENT_TYPE_COMPILE_RESULT,
            Self::TestResult => EVENT_TYPE_TEST_RESULT,
            Self::LintResult => EVENT_TYPE_LINT_RESULT,
            Self::ValidationResult => EVENT_TYPE_VALIDATION_RESULT,
            Self::PatchResult => EVENT_TYPE_PATCH_RESULT,
            Self::Remediation => EVENT_TYPE_REMEDIATION,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(value: &str) -> Option<Self> {
        Some(match value {
            EVENT_TYPE_TOOL_RESULT => Self::ToolResult,
            EVENT_TYPE_COMPILE_RESULT => Self::CompileResult,
            EVENT_TYPE_TEST_RESULT => Self::TestResult,
            EVENT_TYPE_LINT_RESULT => Self::LintResult,
            EVENT_TYPE_VALIDATION_RESULT => Self::ValidationResult,
            EVENT_TYPE_PATCH_RESULT => Self::PatchResult,
            EVENT_TYPE_REMEDIATION => Self::Remediation,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ExecutionEventContext {
    pub session_id: String,
    pub repo: String,
    pub task: String,
    pub actor: String,
    pub correlation_id: Uuid,
    pub parent_event_id: Option<Uuid>,
    pub trajectory: Option<crate::trajectory::TrajectoryContext>,
}

impl ExecutionEventContext {
    /// Reuse the same correlation ID while advancing the parent pointer.
    /// This is the chain model: the patch is reconstructed from linked events.
    pub fn child_of(&self, parent_event_id: Uuid) -> Self {
        Self {
            parent_event_id: Some(parent_event_id),
            ..self.clone()
        }
    }
}

pub fn is_failure_outcome_event_type(event_type: &str) -> bool {
    FAILURE_OUTCOME_EVENT_TYPES.contains(&event_type)
}

pub fn build_execution_event(
    ctx: &ExecutionEventContext,
    kind: ExecutionEventKind,
    success: bool,
    mut payload: Value,
) -> AgentEvent {
    let event_id = Uuid::new_v4();
    let event_type = kind.as_str();
    if is_failure_outcome_event_type(event_type) {
        let has_signature_shape = ["signature", "signature_category", "fingerprint_version"]
            .iter()
            .all(|key| payload.get(*key).is_some());
        if !has_signature_shape {
            let raw = outcome_raw_text_from_payload(&payload, None);
            payload = inline_signature_payload(payload, success, &raw);
        }
    }
    let event_role = ctx
        .trajectory
        .and_then(|_| crate::trajectory::default_role_for_event_type(event_type));
    let trajectory_id = ctx.trajectory.map(|trajectory| trajectory.trajectory_id);
    let attempt_index = ctx.trajectory.map(|trajectory| trajectory.attempt_index);
    // The structured envelope is duplicated into metadata while the chain IDs
    // are also real columns. That keeps old retrieval/indexing behavior intact
    // and gives SQL exact-match queries fast access to chain fields.
    let mut metadata = json!({
        "event_type": event_type,
        "success": success,
        "correlation_id": ctx.correlation_id,
        "parent_event_id": ctx.parent_event_id,
        "repo": ctx.repo,
        "task": ctx.task,
        "payload": payload,
    });
    if let Some(trajectory) = ctx.trajectory {
        metadata["trajectory_id"] = json!(trajectory.trajectory_id);
        metadata["attempt_index"] = json!(trajectory.attempt_index);
    }
    if let Some(role) = event_role {
        metadata["event_role"] = json!(role.as_str());
    }
    metadata = crate::feature_extraction::annotate_event_metadata(
        event_type,
        &event_summary(event_type, success, &metadata["payload"]),
        Some(&summarize_text(&metadata["payload"].to_string(), 1000)),
        metadata,
    );

    AgentEvent {
        id: event_id.to_string(),
        session_id: ctx.session_id.clone(),
        repo: ctx.repo.clone(),
        actor: ctx.actor.clone(),
        event_type: event_type.to_string(),
        summary: event_summary(event_type, success, &metadata["payload"]),
        evidence: Some(summarize_text(&metadata["payload"].to_string(), 1000)),
        metadata,
        correlation_id: Some(ctx.correlation_id),
        parent_event_id: ctx.parent_event_id,
        trajectory_id,
        attempt_index,
        event_role: event_role.map(|role| role.as_str().to_string()),
        created_at: chrono::Utc::now(),
        summary_level: 0,
    }
}
