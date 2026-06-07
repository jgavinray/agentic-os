use deadpool_postgres::Pool;

pub use crate::context_rendering::{build_context, build_layered_context, estimate_tokens};
pub use crate::db_connection::{acquire_single_writer_lock, create_pool, SingleWriterGuard};
pub use crate::db_context_artifacts::{
    get_active_context_artifacts, get_context_artifacts_for_repo,
    get_recent_instruction_candidates, get_recent_repo_decision_candidates,
    get_recent_session_events, upsert_context_artifact,
};
pub use crate::db_context_compiler_ledger::{
    get_context_compiler_ledger, insert_context_compiler_ledger_entry,
};
pub use crate::db_context_events::{
    count_events_for_repo, get_context_events_for_repo, get_context_evidence_for_policy,
    get_events_for_repo, hydrate_active_search_hits, preferred_summary_levels, search_events_fts,
};
pub use crate::db_execution_events::{
    append_execution_event, insert_compile_result_event, insert_execution_artifact_event,
    insert_lint_result_event, insert_patch_result_event, insert_remediation_event,
    insert_test_result_event, insert_tool_result_event, insert_validation_result_event,
};
pub use crate::db_failures::{
    get_active_errors, get_failure_history_for_signatures, get_recent_failure_history,
    insert_error_record, warn_if_legacy_signature_backfill_pending,
};
pub use crate::db_sessions::{
    append_event_from_request, create_session, event_from_append_request, find_or_create_session,
    start_session_from_request,
};
pub use crate::db_trajectory::{
    get_event_chain_by_event_id, get_trajectory, get_trajectory_attempts, get_trajectory_result,
    idle_trajectory_ids, latest_trajectory_event_for_session, order_event_chain,
};
pub use crate::db_trajectory_results::emit_trajectory_result_once;
pub use crate::db_types::{
    AgentEvent, ContextCompilerLedgerEntry, ContextEvidence, FailureHistoryItem,
    VllmCacheObservationInput, VllmCacheStats,
};
pub use crate::db_usage::{
    get_vllm_cache_stats, insert_vllm_cache_observation, record_token_usage,
};

// ── DB query functions ────────────────────────────────────────

pub async fn insert_event(pool: &Pool, event: &AgentEvent) -> Result<(), anyhow::Error> {
    crate::trajectory::validate_event_role(event.event_role.as_deref())?;
    if matches!(event.attempt_index, Some(attempt) if attempt < 1) {
        anyhow::bail!("attempt_index must be positive");
    }
    let lineage_fields = [
        event.trajectory_id.is_some(),
        event.attempt_index.is_some(),
        event.event_role.is_some(),
    ];
    if lineage_fields.iter().any(|present| *present)
        && !lineage_fields.iter().all(|present| *present)
    {
        anyhow::bail!("trajectory_id, attempt_index, and event_role must be written together");
    }
    let metadata = crate::harness_feedback::annotate_event_metadata(
        &event.event_type,
        &event.summary,
        event.evidence.as_deref(),
        event.metadata.clone(),
    );
    let metadata = crate::feature_extraction::annotate_event_metadata(
        &event.event_type,
        &event.summary,
        event.evidence.as_deref(),
        metadata,
    );
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        conn.execute(
            "INSERT INTO agent_events
             (id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, summary_level)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)",
            &[
                &event.id,
                &event.session_id,
                &event.repo,
                &event.actor,
                &event.event_type,
                &event.summary,
                &event.evidence,
                &metadata,
                &event.correlation_id,
                &event.parent_event_id,
                &event.trajectory_id,
                &event.attempt_index,
                &event.event_role,
                &event.summary_level,
            ],
        )
        .await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("insert_event", started.elapsed(), result.is_ok());
    if result.is_ok() {
        crate::harness_feedback::record_metadata_metrics(&metadata);
        let classification_event = AgentEvent {
            metadata,
            ..event.clone()
        };
        if let Err(e) =
            crate::request_classification::classify_and_persist_event(pool, &classification_event)
                .await
        {
            tracing::warn!(
                target: "request_classification",
                event_id = %event.id,
                event_type = %event.event_type,
                "failed to persist request classification for new event: {e}"
            );
        }
        tracing::info!(
            target: "execution_feedback",
            event_type = %event.event_type,
            event_id = %event.id,
            correlation_id = ?event.correlation_id,
            "event written"
        );
    }
    result
}

// ── Context pack builder ──────────────────────────────────────

pub async fn check_ready(pool: &deadpool_postgres::Pool) -> Result<(), anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let _ = conn.query_one("SELECT 1", &[]).await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("check_ready", started.elapsed(), result.is_ok());
    result
}

#[cfg(test)]
#[path = "db_tests.rs"]
mod tests;
