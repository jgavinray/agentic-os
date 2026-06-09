use crate::feature_extraction_persistence_sql::UPSERT_FEATURE_RECORD_SQL;
use crate::feature_extraction_types::FeatureRecord;
use deadpool_postgres::Pool;

pub async fn persist_feature_record(
    pool: &Pool,
    record: &FeatureRecord,
) -> Result<(), anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let failure_classes = serde_json::to_value(&record.failure_classes)?;
        let recommended_constraints = serde_json::to_value(&record.recommended_constraints)?;
        let suppressed_constraints = serde_json::to_value(&record.suppressed_constraints)?;
        conn.execute(
            UPSERT_FEATURE_RECORD_SQL,
            &[
                &record.feature_id,
                &record.trajectory_id,
                &record.session_id,
                &record.repo,
                &record.window_start,
                &record.window_end,
                &record.event_count,
                &record.user_message_count,
                &record.assistant_message_count,
                &record.failed_attempt_count,
                &record.summary_count,
                &record.tool_loop_count,
                &record.repeated_read_loop_count,
                &record.repeated_bash_loop_count,
                &record.user_interruption_count,
                &record.missing_auth_count,
                &record.wrong_endpoint_count,
                &record.summarization_failure_count,
                &record.migration_failure_count,
                &record.other_failure_count,
                &record.context_pack_empty_count,
                &record.context_pack_truncated_count,
                &record.high_input_token_count,
                &record.slow_upstream_model_count,
                &record.empty_tool_use_message_count,
                &record.abandoned_before_model_count,
                &record.single_model_abandoned_no_tools_count,
                &record.summarizer_shared_upstream_count,
                &record.known_endpoint,
                &record.known_auth_header,
                &record.known_migration_fix,
                &record.loop_detected,
                &record.user_interrupted,
                &failure_classes,
                &recommended_constraints,
                &suppressed_constraints,
            ],
        )
        .await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("feature_persist_record", started.elapsed(), result.is_ok());
    result
}
