use crate::db::AgentEvent;
use crate::feature_constraints::{record_latest_recovery, FailureKey};
use crate::feature_extraction_metadata::{
    context_pack_empty, context_pack_token_count, context_pack_truncated_value, event_endpoint,
    event_input_tokens_from_metadata, event_latency_ms_from_metadata, event_success,
    event_tool_name, is_successful_authenticated_request, is_successful_migration,
    is_successful_summarization, summarizer_has_dedicated_upstream, HIGH_INPUT_TOKEN_THRESHOLD,
    SLOW_UPSTREAM_MODEL_MS_THRESHOLD,
};
use chrono::{DateTime, Utc};
use std::collections::BTreeMap;

pub(crate) fn record_fresh_recovery_signals(
    event: &AgentEvent,
    freshness_cutoff: DateTime<Utc>,
    latest_recovery: &mut BTreeMap<FailureKey, DateTime<Utc>>,
    endpoint_recoveries: &mut Vec<(DateTime<Utc>, String)>,
) {
    if event.created_at < freshness_cutoff {
        return;
    }

    if event_success(event) {
        record_successful_event_recovery(event, latest_recovery, endpoint_recoveries);
    }
    record_metadata_recovery(event, latest_recovery);
}

pub(crate) fn reconcile_endpoint_recoveries(
    known_endpoint: Option<&str>,
    endpoint_recoveries: Vec<(DateTime<Utc>, String)>,
    latest_recovery: &mut BTreeMap<FailureKey, DateTime<Utc>>,
) {
    let Some(endpoint) = known_endpoint else {
        return;
    };
    for (time, recovered_endpoint) in endpoint_recoveries {
        if recovered_endpoint == endpoint {
            record_latest_recovery(latest_recovery, FailureKey::WrongEndpoint, time);
        }
    }
}

fn record_successful_event_recovery(
    event: &AgentEvent,
    latest_recovery: &mut BTreeMap<FailureKey, DateTime<Utc>>,
    endpoint_recoveries: &mut Vec<(DateTime<Utc>, String)>,
) {
    if event.event_type == crate::execution_feedback::EVENT_TYPE_TOOL_RESULT {
        if let Some(tool) = event_tool_name(event) {
            record_latest_recovery(
                latest_recovery,
                FailureKey::ToolLoop(tool.to_string()),
                event.created_at,
            );
        }
    }
    if is_successful_authenticated_request(event) {
        record_latest_recovery(latest_recovery, FailureKey::MissingAuth, event.created_at);
    }
    if let Some(endpoint) = event_endpoint(event) {
        endpoint_recoveries.push((event.created_at, endpoint.to_string()));
    }
    if is_successful_summarization(event) {
        record_latest_recovery(
            latest_recovery,
            FailureKey::SummarizationFailure,
            event.created_at,
        );
    }
    if is_successful_migration(event) {
        record_latest_recovery(
            latest_recovery,
            FailureKey::MigrationFailure,
            event.created_at,
        );
    }
}

fn record_metadata_recovery(
    event: &AgentEvent,
    latest_recovery: &mut BTreeMap<FailureKey, DateTime<Utc>>,
) {
    if event.event_type == "context_pack" {
        if context_pack_token_count(&event.metadata).is_some()
            && !context_pack_empty(&event.metadata)
        {
            record_latest_recovery(
                latest_recovery,
                FailureKey::ContextPackEmpty,
                event.created_at,
            );
        }
        if context_pack_truncated_value(&event.metadata) == Some(false) {
            record_latest_recovery(
                latest_recovery,
                FailureKey::ContextPackTruncated,
                event.created_at,
            );
        }
    }
    if matches!(
        event.event_type.as_str(),
        "assistant_message" | "trajectory_result"
    ) {
        if event_input_tokens_from_metadata(&event.metadata)
            .is_some_and(|tokens| tokens > 0 && tokens < HIGH_INPUT_TOKEN_THRESHOLD)
        {
            record_latest_recovery(
                latest_recovery,
                FailureKey::HighInputTokens,
                event.created_at,
            );
        }
        if event_latency_ms_from_metadata(&event.metadata).is_some_and(|latency_ms| {
            latency_ms > 0 && latency_ms < SLOW_UPSTREAM_MODEL_MS_THRESHOLD
        }) {
            record_latest_recovery(
                latest_recovery,
                FailureKey::SlowUpstreamModel,
                event.created_at,
            );
        }
    }
    if summarizer_has_dedicated_upstream(&event.metadata) {
        record_latest_recovery(
            latest_recovery,
            FailureKey::SummarizerSharedUpstream,
            event.created_at,
        );
    }
}
