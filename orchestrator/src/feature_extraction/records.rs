use crate::db::AgentEvent;
use crate::feature_constraints::{build_constraints, failure_key_for_tag, FailureKey};
use crate::feature_extraction_grouping::FeatureGroupKey;
use crate::feature_extraction_record_init::initialize_feature_record;
use crate::feature_extraction_recovery::{
    reconcile_endpoint_recoveries, record_fresh_recovery_signals,
};
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

        record_fresh_recovery_signals(
            event,
            freshness_cutoff,
            &mut latest_recovery,
            &mut endpoint_recoveries,
        );
    }

    reconcile_endpoint_recoveries(
        record.known_endpoint.as_deref(),
        endpoint_recoveries,
        &mut latest_recovery,
    );

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
