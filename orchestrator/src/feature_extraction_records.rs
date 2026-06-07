use crate::db::AgentEvent;
use crate::feature_constraints::{
    build_constraints, failure_key_for_tag, record_latest_recovery, FailureKey,
};
use crate::feature_extraction_grouping::FeatureGroupKey;
use crate::feature_extraction_metadata::{
    context_pack_empty, context_pack_token_count, context_pack_truncated_value, event_endpoint,
    event_input_tokens_from_metadata, event_latency_ms_from_metadata, event_success,
    event_tool_name, is_successful_authenticated_request, is_successful_migration,
    is_successful_summarization, summarizer_has_dedicated_upstream, HIGH_INPUT_TOKEN_THRESHOLD,
    SLOW_UPSTREAM_MODEL_MS_THRESHOLD,
};
use crate::feature_extraction_record_init::initialize_feature_record;
use crate::feature_extraction_tags::{
    apply_tag_to_record, extraction_condition_key, recognized_tags,
};
use crate::feature_extraction_types::{ExtractionConfig, FeatureRecord};
use chrono::{DateTime, Duration, Utc};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn extract_record_for_group(
    key: &FeatureGroupKey,
    events: &[AgentEvent],
    config: &ExtractionConfig,
) -> (FeatureRecord, u64) {
    let mut record = initialize_feature_record(key, events, config);

    let freshness_cutoff =
        config.evaluation_time - Duration::seconds(config.constraint_freshness_window_sec);
    let mut unknown_versions = 0u64;
    let mut failure_classes = BTreeSet::new();
    let mut latest_detection: BTreeMap<FailureKey, DateTime<Utc>> = BTreeMap::new();
    let mut stale_detection_keys = BTreeSet::new();
    let mut latest_recovery: BTreeMap<FailureKey, DateTime<Utc>> = BTreeMap::new();
    let mut endpoint_recoveries: Vec<(DateTime<Utc>, String)> = Vec::new();

    for event in events {
        match event.event_type.as_str() {
            "user_message" => record.user_message_count += 1,
            "assistant_message" => record.assistant_message_count += 1,
            "failed_attempt" => record.failed_attempt_count += 1,
            "summary" => record.summary_count += 1,
            _ => {}
        }

        let parsed = recognized_tags(event);
        unknown_versions += parsed.unknown_schema_versions;
        let mut condition_keys = BTreeSet::new();
        let recognized_count = parsed.tags.len();
        for tag in parsed.tags {
            let condition_key = extraction_condition_key(&tag);
            if !condition_keys.insert(condition_key) {
                continue;
            }
            apply_tag_to_record(event, &tag, &mut record, &mut failure_classes);
            if let Some(key) = failure_key_for_tag(&tag) {
                if event.created_at >= freshness_cutoff {
                    latest_detection
                        .entry(key)
                        .and_modify(|time| {
                            if event.created_at > *time {
                                *time = event.created_at;
                            }
                        })
                        .or_insert(event.created_at);
                } else {
                    stale_detection_keys.insert(key);
                }
            }
            if tag.tag_type == "correction_acknowledged" && event.created_at >= freshness_cutoff {
                latest_recovery
                    .entry(FailureKey::UserInterruption)
                    .and_modify(|time| {
                        if event.created_at > *time {
                            *time = event.created_at;
                        }
                    })
                    .or_insert(event.created_at);
            }
        }

        if event.event_type == "failed_attempt" && recognized_count == 0 {
            record.other_failure_count += 1;
            failure_classes.insert("other".to_string());
        }

        if event.created_at >= freshness_cutoff && event_success(event) {
            if event.event_type == crate::execution_feedback::EVENT_TYPE_TOOL_RESULT {
                if let Some(tool) = event_tool_name(event) {
                    latest_recovery
                        .entry(FailureKey::ToolLoop(tool.to_string()))
                        .and_modify(|time| {
                            if event.created_at > *time {
                                *time = event.created_at;
                            }
                        })
                        .or_insert(event.created_at);
                }
            }
            if is_successful_authenticated_request(event) {
                latest_recovery
                    .entry(FailureKey::MissingAuth)
                    .and_modify(|time| {
                        if event.created_at > *time {
                            *time = event.created_at;
                        }
                    })
                    .or_insert(event.created_at);
            }
            if let Some(endpoint) = event_endpoint(event) {
                endpoint_recoveries.push((event.created_at, endpoint.to_string()));
            }
            if is_successful_summarization(event) {
                latest_recovery
                    .entry(FailureKey::SummarizationFailure)
                    .and_modify(|time| {
                        if event.created_at > *time {
                            *time = event.created_at;
                        }
                    })
                    .or_insert(event.created_at);
            }
            if is_successful_migration(event) {
                latest_recovery
                    .entry(FailureKey::MigrationFailure)
                    .and_modify(|time| {
                        if event.created_at > *time {
                            *time = event.created_at;
                        }
                    })
                    .or_insert(event.created_at);
            }
        }

        if event.created_at >= freshness_cutoff {
            if event.event_type == "context_pack" {
                if context_pack_token_count(&event.metadata).is_some()
                    && !context_pack_empty(&event.metadata)
                {
                    record_latest_recovery(
                        &mut latest_recovery,
                        FailureKey::ContextPackEmpty,
                        event.created_at,
                    );
                }
                if context_pack_truncated_value(&event.metadata) == Some(false) {
                    record_latest_recovery(
                        &mut latest_recovery,
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
                        &mut latest_recovery,
                        FailureKey::HighInputTokens,
                        event.created_at,
                    );
                }
                if event_latency_ms_from_metadata(&event.metadata).is_some_and(|latency_ms| {
                    latency_ms > 0 && latency_ms < SLOW_UPSTREAM_MODEL_MS_THRESHOLD
                }) {
                    record_latest_recovery(
                        &mut latest_recovery,
                        FailureKey::SlowUpstreamModel,
                        event.created_at,
                    );
                }
            }
            if summarizer_has_dedicated_upstream(&event.metadata) {
                record_latest_recovery(
                    &mut latest_recovery,
                    FailureKey::SummarizerSharedUpstream,
                    event.created_at,
                );
            }
        }
    }

    if let Some(endpoint) = record.known_endpoint.as_deref() {
        for (time, recovered_endpoint) in endpoint_recoveries {
            if recovered_endpoint == endpoint {
                latest_recovery
                    .entry(FailureKey::WrongEndpoint)
                    .and_modify(|existing| {
                        if time > *existing {
                            *existing = time;
                        }
                    })
                    .or_insert(time);
            }
        }
    }

    record.failure_classes = failure_classes.into_iter().collect();
    build_constraints(
        &mut record,
        &latest_detection,
        &latest_recovery,
        &stale_detection_keys,
        config.max_operational_constraints,
    );
    (record, unknown_versions)
}
