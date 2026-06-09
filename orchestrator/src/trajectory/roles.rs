use crate::trajectory::types::{EventRole, EVENT_TYPE_CONTEXT_PACK, EVENT_TYPE_TRAJECTORY_RESULT};

pub fn validate_event_role(role: Option<&str>) -> Result<(), anyhow::Error> {
    if let Some(role) = role {
        if EventRole::from_str(role).is_none() {
            anyhow::bail!("invalid event_role `{role}`");
        }
    }
    Ok(())
}

pub fn is_trajectory_participating_event_type(event_type: &str) -> bool {
    matches!(
        event_type,
        "user_message"
            | "assistant_message"
            | "failed_attempt"
            | EVENT_TYPE_CONTEXT_PACK
            | EVENT_TYPE_TRAJECTORY_RESULT
            | crate::execution_feedback::EVENT_TYPE_TOOL_RESULT
            | crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT
            | crate::execution_feedback::EVENT_TYPE_TEST_RESULT
            | crate::execution_feedback::EVENT_TYPE_LINT_RESULT
            | crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT
            | crate::execution_feedback::EVENT_TYPE_PATCH_RESULT
            | crate::execution_feedback::EVENT_TYPE_REMEDIATION
    )
}

pub fn default_role_for_event_type(event_type: &str) -> Option<EventRole> {
    Some(match event_type {
        "user_message" => EventRole::Request,
        "assistant_message" => EventRole::ModelResponse,
        "failed_attempt" => EventRole::Failure,
        EVENT_TYPE_CONTEXT_PACK => EventRole::ContextPack,
        EVENT_TYPE_TRAJECTORY_RESULT => EventRole::TrajectoryResult,
        crate::execution_feedback::EVENT_TYPE_TOOL_RESULT => EventRole::ToolResult,
        crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT
        | crate::execution_feedback::EVENT_TYPE_TEST_RESULT
        | crate::execution_feedback::EVENT_TYPE_LINT_RESULT
        | crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT => EventRole::Validation,
        crate::execution_feedback::EVENT_TYPE_PATCH_RESULT => EventRole::Patch,
        crate::execution_feedback::EVENT_TYPE_REMEDIATION => EventRole::Remediation,
        _ => return None,
    })
}
