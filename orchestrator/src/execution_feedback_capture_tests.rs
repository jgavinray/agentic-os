use super::*;

fn ctx() -> ExecutionEventContext {
    ExecutionEventContext {
        session_id: "session".to_string(),
        repo: "repo".to_string(),
        task: "task".to_string(),
        actor: "agent".to_string(),
        correlation_id: Uuid::new_v4(),
        parent_event_id: None,
        trajectory: None,
    }
}

#[path = "execution_feedback_tool_capture_tests.rs"]
mod tool_capture_tests;

#[path = "execution_feedback_validation_report_tests.rs"]
mod validation_report_tests;

#[path = "execution_feedback_capture_performance_tests.rs"]
mod performance_tests;
