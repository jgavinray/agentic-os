use deadpool_postgres::Pool;
use uuid::Uuid;

use crate::db_types::AgentEvent;

/// Store an execution artifact through the same event log and best-effort
/// vector indexing path used by ordinary memory events.
pub async fn append_execution_event(
    pool: &Pool,
    embedder: &crate::embedder::Embedder,
    qdrant_url: &str,
    event: AgentEvent,
) -> Result<(String, bool), anyhow::Error> {
    let id = insert_execution_artifact_event(pool, &event).await?;
    let qdrant_indexed = match crate::qdrant::store_event(embedder, qdrant_url, &event).await {
        Ok(_) => true,
        Err(e) => {
            tracing::warn!(
                event_id = %id,
                event_type = %event.event_type,
                "execution event stored in postgres but qdrant indexing failed: {e}"
            );
            false
        }
    };
    Ok((id, qdrant_indexed))
}

pub async fn insert_execution_artifact_event(
    pool: &Pool,
    event: &AgentEvent,
) -> Result<String, anyhow::Error> {
    crate::db::insert_event(pool, event).await?;
    // Metrics are updated only after the DB write succeeds so counters describe
    // persisted outcome memory, not attempted capture.
    crate::telemetry::record_execution_artifact(event);
    Ok(event.id.clone())
}

#[allow(dead_code)]
pub async fn insert_tool_result_event(
    pool: &Pool,
    ctx: &crate::execution_feedback::ExecutionEventContext,
    result: &crate::execution_feedback::CapturedToolResult,
) -> Result<String, anyhow::Error> {
    let event = crate::execution_feedback::build_execution_event(
        ctx,
        crate::execution_feedback::ExecutionEventKind::ToolResult,
        result.exit_code == 0,
        crate::execution_feedback::tool_result_payload(result),
    );
    insert_execution_artifact_event(pool, &event).await
}

#[allow(dead_code)]
pub async fn insert_compile_result_event(
    pool: &Pool,
    ctx: &crate::execution_feedback::ExecutionEventContext,
    language: &str,
    target: &str,
    exit_code: i32,
    content: &str,
) -> Result<String, anyhow::Error> {
    let payload =
        crate::execution_feedback::compile_result_payload(language, target, exit_code, content);
    let success = exit_code == 0 && payload["error_count"].as_u64().unwrap_or_default() == 0;
    let event = crate::execution_feedback::build_execution_event(
        ctx,
        crate::execution_feedback::ExecutionEventKind::CompileResult,
        success,
        payload,
    );
    insert_execution_artifact_event(pool, &event).await
}

#[allow(dead_code)]
pub async fn insert_test_result_event(
    pool: &Pool,
    ctx: &crate::execution_feedback::ExecutionEventContext,
    framework: &str,
    content: &str,
) -> Result<String, anyhow::Error> {
    let payload = crate::execution_feedback::test_result_payload(framework, content);
    let success = payload["failed"].as_u64().unwrap_or_default() == 0;
    let event = crate::execution_feedback::build_execution_event(
        ctx,
        crate::execution_feedback::ExecutionEventKind::TestResult,
        success,
        payload,
    );
    insert_execution_artifact_event(pool, &event).await
}

#[allow(dead_code)]
pub async fn insert_lint_result_event(
    pool: &Pool,
    ctx: &crate::execution_feedback::ExecutionEventContext,
    tool_name: &str,
    content: &str,
) -> Result<String, anyhow::Error> {
    let payload = crate::execution_feedback::lint_result_payload(tool_name, content);
    let success = payload["error_count"].as_u64().unwrap_or_default() == 0;
    let event = crate::execution_feedback::build_execution_event(
        ctx,
        crate::execution_feedback::ExecutionEventKind::LintResult,
        success,
        payload,
    );
    insert_execution_artifact_event(pool, &event).await
}

#[allow(dead_code)]
pub async fn insert_validation_result_event(
    pool: &Pool,
    ctx: &crate::execution_feedback::ExecutionEventContext,
    validator_name: &str,
    pass: bool,
    failure_reason: &str,
) -> Result<String, anyhow::Error> {
    let event = crate::execution_feedback::build_execution_event(
        ctx,
        crate::execution_feedback::ExecutionEventKind::ValidationResult,
        pass,
        crate::execution_feedback::validation_result_payload(validator_name, pass, failure_reason),
    );
    insert_execution_artifact_event(pool, &event).await
}

#[allow(dead_code)]
pub async fn insert_patch_result_event(
    pool: &Pool,
    ctx: &crate::execution_feedback::ExecutionEventContext,
    files_touched: Vec<String>,
    outcome: &str,
    validation_event_ids: Vec<Uuid>,
) -> Result<String, anyhow::Error> {
    let event = crate::execution_feedback::build_execution_event(
        ctx,
        crate::execution_feedback::ExecutionEventKind::PatchResult,
        outcome == "applied",
        crate::execution_feedback::patch_result_payload(
            files_touched,
            outcome,
            validation_event_ids,
        ),
    );
    insert_execution_artifact_event(pool, &event).await
}

#[allow(dead_code)]
pub async fn insert_remediation_event(
    pool: &Pool,
    ctx: &crate::execution_feedback::ExecutionEventContext,
    failure_event_id: Uuid,
    resolving_event_id: Uuid,
    signature: &str,
) -> Result<String, anyhow::Error> {
    let event = crate::execution_feedback::build_execution_event(
        ctx,
        crate::execution_feedback::ExecutionEventKind::Remediation,
        true,
        crate::execution_feedback::remediation_payload(
            failure_event_id,
            resolving_event_id,
            signature,
        ),
    );
    insert_execution_artifact_event(pool, &event).await
}
