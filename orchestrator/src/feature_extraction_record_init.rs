use crate::db::AgentEvent;
use crate::feature_extraction_grouping::FeatureGroupKey;
use crate::feature_extraction_types::{ExtractionConfig, FeatureRecord};

pub(crate) fn initialize_feature_record(
    key: &FeatureGroupKey,
    events: &[AgentEvent],
    config: &ExtractionConfig,
) -> FeatureRecord {
    let first = events
        .first()
        .expect("initialize_feature_record requires non-empty events");
    let (trajectory_id, window_start, window_end, feature_id) = match key {
        FeatureGroupKey::Trajectory(trajectory_id) => {
            let window_start = events
                .iter()
                .map(|event| event.created_at)
                .min()
                .unwrap_or(first.created_at);
            let window_end = events
                .iter()
                .map(|event| event.created_at)
                .max()
                .unwrap_or(first.created_at);
            (
                Some(*trajectory_id),
                window_start,
                window_end,
                format!("feature:trajectory:{trajectory_id}"),
            )
        }
        FeatureGroupKey::SessionWindow {
            repo: _,
            session_id,
            window_start,
            window_end,
        } => (
            None,
            *window_start,
            *window_end,
            format!("feature:session:{session_id}:{}", window_start.timestamp()),
        ),
    };

    FeatureRecord {
        feature_id,
        trajectory_id,
        session_id: first.session_id.clone(),
        repo: first.repo.clone(),
        window_start,
        window_end,
        event_count: events.len() as i64,
        user_message_count: 0,
        assistant_message_count: 0,
        failed_attempt_count: 0,
        summary_count: 0,
        tool_loop_count: 0,
        repeated_read_loop_count: 0,
        repeated_bash_loop_count: 0,
        user_interruption_count: 0,
        missing_auth_count: 0,
        wrong_endpoint_count: 0,
        summarization_failure_count: 0,
        migration_failure_count: 0,
        other_failure_count: 0,
        context_pack_empty_count: 0,
        context_pack_truncated_count: 0,
        high_input_token_count: 0,
        slow_upstream_model_count: 0,
        empty_tool_use_message_count: 0,
        abandoned_before_model_count: 0,
        single_model_abandoned_no_tools_count: 0,
        summarizer_shared_upstream_count: 0,
        known_endpoint: None,
        known_auth_header: None,
        known_migration_fix: None,
        loop_detected: false,
        user_interrupted: false,
        failure_classes: vec![],
        recommended_constraints: vec![],
        suppressed_constraints: vec![],
        created_at: config.evaluation_time,
        updated_at: config.evaluation_time,
    }
}
