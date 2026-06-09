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

#[path = "tool_capture.rs"]
mod tool_capture_tests;

#[path = "validation_report.rs"]
mod validation_report_tests;

#[path = "capture_performance.rs"]
mod performance_tests;
