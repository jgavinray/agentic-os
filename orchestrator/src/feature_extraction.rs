//! Deterministic trajectory feature extraction and operational constraints.
//!
//! The source of truth remains `agent_events`. This module consumes structured
//! detection tags from event metadata, aggregates compact feature rows, and
//! renders bounded operational constraints without LLM calls or probabilistic
//! classification. Free-form prose matching is intentionally limited to the
//! bootstrap tagging path for historical events that predate structured tags.

use crate::db::AgentEvent;
use chrono::{DateTime, Duration, TimeZone, Utc};
use deadpool_postgres::Pool;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::DerefMut;
use uuid::Uuid;

use crate::feature_extraction_metadata::{
    bool_path, context_pack_empty, context_pack_token_count, context_pack_truncated,
    context_pack_truncated_value, event_endpoint, event_input_tokens_from_metadata,
    event_latency_ms_from_metadata, event_success, event_tool_name,
    is_successful_authenticated_request, is_successful_migration, is_successful_summarization,
    known_fact, string_path, summarizer_has_dedicated_upstream, summarizer_shares_litellm_upstream,
    trajectory_abandoned_before_model, trajectory_single_model_abandoned_no_tools,
};
pub use crate::feature_extraction_types::{
    constraint_freshness_window_sec_from_env, feature_extraction_enabled_from_env,
    feature_window_sec_from_env, max_operational_constraints_from_env,
    operational_constraints_token_budget_from_env, BackfillOptions, BackfillReport, DetectionTag,
    ExtractionConfig, ExtractionReport, ExtractionScope, FeatureRecord, OperationalConstraint,
    SuppressedConstraint, DEFAULT_CONSTRAINT_FRESHNESS_WINDOW_SEC, DEFAULT_FEATURE_WINDOW_SEC,
    DEFAULT_MAX_OPERATIONAL_CONSTRAINTS, DEFAULT_OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET,
    DETECTION_TAG_SOURCES, DETECTION_TAG_TYPES, FEATURE_FAILURE_CLASSES,
    OPERATIONAL_CONSTRAINT_TYPES, OPERATIONAL_SUPPRESSION_REASONS, TAG_SCHEMA_VERSION,
};

pub use crate::feature_extraction_metadata::{
    CONTEXT_PACK_EMPTY_TOKEN_THRESHOLD, HIGH_INPUT_TOKEN_THRESHOLD,
    SLOW_UPSTREAM_MODEL_MS_THRESHOLD,
};

const CONSTRAINT_PRIORITY: [&str; 9] = [
    "use_known_auth",
    "use_known_endpoint",
    "use_known_migration_fix",
    "avoid_tool_loop",
    "fix_context_retrieval",
    "reduce_context_bloat",
    "separate_summarizer_upstream",
    "handle_user_interruption",
    "handle_summarization_failure",
];

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum FeatureGroupKey {
    Trajectory(Uuid),
    SessionWindow {
        repo: String,
        session_id: String,
        window_start: DateTime<Utc>,
        window_end: DateTime<Utc>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum FailureKey {
    ToolLoop(String),
    UserInterruption,
    MissingAuth,
    WrongEndpoint,
    SummarizationFailure,
    MigrationFailure,
    ContextPackEmpty,
    ContextPackTruncated,
    HighInputTokens,
    SlowUpstreamModel,
    EmptyToolUseMessage,
    AbandonedBeforeModel,
    SingleModelAbandonedNoTools,
    SummarizerSharedUpstream,
}

pub fn annotate_event_metadata(
    event_type: &str,
    summary: &str,
    evidence: Option<&str>,
    metadata: Value,
) -> Value {
    merge_detection_tags(
        metadata.clone(),
        live_detection_tags_for_event(event_type, summary, evidence, &metadata),
    )
}

pub fn bootstrap_annotate_event_metadata(event: &AgentEvent) -> Value {
    merge_detection_tags(
        event.metadata.clone(),
        bootstrap_detection_tags_for_event(event),
    )
}

pub fn live_detection_tags_for_event(
    event_type: &str,
    summary: &str,
    evidence: Option<&str>,
    metadata: &Value,
) -> Vec<DetectionTag> {
    deterministic_tags_for_event(event_type, summary, evidence, metadata, false, None)
}

pub fn bootstrap_detection_tags_for_event(event: &AgentEvent) -> Vec<DetectionTag> {
    deterministic_tags_for_event(
        &event.event_type,
        &event.summary,
        event.evidence.as_deref(),
        &event.metadata,
        true,
        Some("bootstrap_migration"),
    )
}

fn deterministic_tags_for_event(
    event_type: &str,
    summary: &str,
    evidence: Option<&str>,
    metadata: &Value,
    allow_text_bootstrap: bool,
    source_override: Option<&str>,
) -> Vec<DetectionTag> {
    let mut tags = Vec::new();

    tags.extend(explicit_producer_signal_tags(metadata, source_override));

    if bool_path(metadata, &["hook", "tool_loop"]) {
        if let Some(tool) = string_path(metadata, &["hook", "tool"]) {
            tags.push(DetectionTag::tool_loop(
                source_override.unwrap_or("hook_parser"),
                tool,
            ));
        }
    }
    if let Some(tool) = string_path(metadata, &["hook", "tool_loop_tool"]) {
        tags.push(DetectionTag::tool_loop(
            source_override.unwrap_or("hook_parser"),
            tool,
        ));
    }

    if bool_path(metadata, &["tool_loop_detected"])
        || bool_path(metadata, &["loop_detected"])
        || string_path(metadata, &["failure_class"]) == Some("tool_loop")
    {
        let tool = string_path(metadata, &["tool"])
            .or_else(|| string_path(metadata, &["tool_name"]))
            .or_else(|| string_path(metadata, &["payload", "tool_name"]))
            .unwrap_or("unknown");
        tags.push(DetectionTag::tool_loop(
            source_override.unwrap_or("tool_loop_detector"),
            tool,
        ));
    }

    if let Some(class) = string_path(metadata, &["failure_class"])
        .or_else(|| string_path(metadata, &["payload", "failure_class"]))
        .or_else(|| string_path(metadata, &["validation_failure"]))
    {
        match class {
            "missing_auth" | "wrong_endpoint" | "summarization_failure" | "migration_failure" => {
                tags.push(DetectionTag::new(
                    class,
                    source_override.unwrap_or("validation_parser"),
                ));
            }
            "user_interruption" if event_type == "failed_attempt" => {
                tags.push(DetectionTag::new(
                    "user_interruption",
                    source_override.unwrap_or("failed_attempt_classifier"),
                ));
            }
            _ => {}
        }
    }

    if event_type == "failed_attempt" {
        if bool_path(metadata, &["user_interruption"]) {
            tags.push(DetectionTag::new(
                "user_interruption",
                source_override.unwrap_or("failed_attempt_classifier"),
            ));
        }
        if let Some(tool) = string_path(metadata, &["repeated_tool"]) {
            tags.push(DetectionTag::tool_loop(
                source_override.unwrap_or("failed_attempt_classifier"),
                tool,
            ));
        }
    }

    if event_type == "summary"
        && (bool_path(metadata, &["summarization_failure"])
            || metadata.get("success").and_then(Value::as_bool) == Some(false))
    {
        tags.push(DetectionTag::new(
            "summarization_failure",
            source_override.unwrap_or("summarizer"),
        ));
    }

    if event_type == "context_pack" {
        if context_pack_empty(metadata) {
            tags.push(DetectionTag::new(
                "context_pack_empty",
                source_override.unwrap_or("context_builder"),
            ));
        }
        if context_pack_truncated(metadata) {
            tags.push(DetectionTag::new(
                "context_pack_truncated",
                source_override.unwrap_or("context_builder"),
            ));
        }
    }

    if event_type == "assistant_message" {
        if event_input_tokens_from_metadata(metadata)
            .is_some_and(|tokens| tokens >= HIGH_INPUT_TOKEN_THRESHOLD)
        {
            tags.push(DetectionTag::new(
                "high_input_tokens",
                source_override.unwrap_or("model_response_parser"),
            ));
        }
        if event_latency_ms_from_metadata(metadata)
            .is_some_and(|latency_ms| latency_ms >= SLOW_UPSTREAM_MODEL_MS_THRESHOLD)
        {
            tags.push(DetectionTag::new(
                "slow_upstream_model",
                source_override.unwrap_or("model_response_parser"),
            ));
        }
        if string_path(metadata, &["finish_reason"])
            .or_else(|| string_path(metadata, &["payload", "finish_reason"]))
            == Some("tool_use")
            && summary.trim().is_empty()
        {
            tags.push(DetectionTag::new(
                "empty_tool_use_message",
                source_override.unwrap_or("model_response_parser"),
            ));
        }
    }

    if event_type == "trajectory_result" {
        if event_input_tokens_from_metadata(metadata)
            .is_some_and(|tokens| tokens >= HIGH_INPUT_TOKEN_THRESHOLD)
        {
            tags.push(DetectionTag::new(
                "high_input_tokens",
                source_override.unwrap_or("trajectory_analyzer"),
            ));
        }
        if event_latency_ms_from_metadata(metadata)
            .is_some_and(|latency_ms| latency_ms >= SLOW_UPSTREAM_MODEL_MS_THRESHOLD)
        {
            tags.push(DetectionTag::new(
                "slow_upstream_model",
                source_override.unwrap_or("trajectory_analyzer"),
            ));
        }
        if trajectory_abandoned_before_model(metadata) {
            tags.push(DetectionTag::new(
                "abandoned_before_model",
                source_override.unwrap_or("trajectory_analyzer"),
            ));
        }
        if trajectory_single_model_abandoned_no_tools(metadata) {
            tags.push(DetectionTag::new(
                "single_model_abandoned_no_tools",
                source_override.unwrap_or("trajectory_analyzer"),
            ));
        }
    }

    if summarizer_shares_litellm_upstream(metadata) {
        tags.push(DetectionTag::new(
            "summarizer_shared_upstream",
            source_override.unwrap_or("config_validator"),
        ));
    }

    if bool_path(metadata, &["correction_acknowledged"])
        || bool_path(metadata, &["payload", "correction_acknowledged"])
    {
        tags.push(DetectionTag::new(
            "correction_acknowledged",
            source_override.unwrap_or("remediation_parser"),
        ));
    }

    if allow_text_bootstrap {
        tags.extend(bootstrap_text_tags(
            summary,
            evidence,
            metadata,
            source_override,
        ));
    }

    dedupe_tags_within_producer(tags)
}

fn explicit_producer_signal_tags(
    metadata: &Value,
    source_override: Option<&str>,
) -> Vec<DetectionTag> {
    let mut tags = Vec::new();
    let Some(signals) = metadata.get("producer_signals").and_then(Value::as_object) else {
        return tags;
    };
    for (source, value) in signals {
        let source = source_override.unwrap_or(source.as_str());
        match value {
            Value::Array(values) => {
                for value in values {
                    if let Some(tag) = tag_from_signal_value(source, value) {
                        tags.push(tag);
                    }
                }
            }
            Value::Object(_) => {
                if let Some(tag) = tag_from_signal_value(source, value) {
                    tags.push(tag);
                }
            }
            Value::String(tag_type) => tags.push(DetectionTag::new(tag_type, source)),
            _ => {}
        }
    }
    tags
}

fn tag_from_signal_value(source: &str, value: &Value) -> Option<DetectionTag> {
    let obj = value.as_object()?;
    let tag_type = obj.get("type").and_then(Value::as_str)?;
    if tag_type == "tool_loop" {
        let tool = obj.get("tool").and_then(Value::as_str).unwrap_or("unknown");
        Some(DetectionTag::tool_loop(source, tool))
    } else {
        Some(DetectionTag::new(tag_type, source))
    }
}

fn bootstrap_text_tags(
    summary: &str,
    evidence: Option<&str>,
    metadata: &Value,
    source_override: Option<&str>,
) -> Vec<DetectionTag> {
    let source = source_override.unwrap_or("bootstrap_migration");
    let haystack = format!(
        "{}\n{}\n{}",
        summary,
        evidence.unwrap_or_default(),
        metadata
    )
    .to_ascii_lowercase();
    let mut tags = Vec::new();

    if haystack.contains("repeated read")
        || haystack.contains("read loop")
        || haystack.contains("read tool loop")
        || haystack.contains("tool 'read' called")
        || haystack.contains("tool \"read\" called")
        || haystack.contains("loop warning: tool 'read'")
    {
        tags.push(DetectionTag::tool_loop(source, "Read"));
    } else if haystack.contains("repeated bash")
        || haystack.contains("bash loop")
        || haystack.contains("bash tool loop")
        || haystack.contains("tool 'bash' called")
        || haystack.contains("tool \"bash\" called")
        || haystack.contains("loop warning: tool 'bash'")
    {
        tags.push(DetectionTag::tool_loop(source, "Bash"));
    } else if haystack.contains("tool loop") || haystack.contains("repeat identical tool") {
        tags.push(DetectionTag::tool_loop(source, "unknown"));
    }

    if haystack.contains("user interruption")
        || haystack.contains("user interrupted")
        || haystack.contains("user correction")
        || haystack.contains("request interrupted by user")
        || haystack.contains("interrupted by user for tool use")
    {
        tags.push(DetectionTag::new("user_interruption", source));
    }
    if haystack.contains("missing authorization")
        || haystack.contains("missing auth")
        || haystack.contains("authorization header")
        || haystack.contains("unauthorized")
        || haystack.contains("without authorization header")
        || haystack.contains("without the authorization header")
        || (haystack.contains("without") && haystack.contains("bearer"))
    {
        tags.push(DetectionTag::new("missing_auth", source));
    }
    if (haystack.contains("wrong endpoint") || haystack.contains("incorrect endpoint"))
        || (haystack.contains("localhost") && haystack.contains("correct endpoint"))
        || (haystack.contains("localhost") && haystack.contains("should be using"))
        || haystack.contains("trying localhost")
        || haystack.contains("not localhost")
    {
        tags.push(DetectionTag::new("wrong_endpoint", source));
    }
    if haystack.contains("summarization failure")
        || haystack.contains("summary failure")
        || haystack.contains("empty summary")
        || (haystack.contains("summarization") && haystack.contains("empty response"))
    {
        tags.push(DetectionTag::new("summarization_failure", source));
    }
    if haystack.contains("migration failure")
        || haystack.contains("baseline migration")
        || haystack.contains("extension creation")
    {
        tags.push(DetectionTag::new("migration_failure", source));
    }
    if haystack.contains("acknowledge")
        && (haystack.contains("correction") || haystack.contains("interruption"))
    {
        tags.push(DetectionTag::new("correction_acknowledged", source));
    }

    tags
}

fn merge_detection_tags(mut metadata: Value, new_tags: Vec<DetectionTag>) -> Value {
    if new_tags.is_empty() && metadata.get("detection_tags").is_none() {
        return metadata;
    }
    if !metadata.is_object() {
        metadata = json!({});
    }

    let mut tags = metadata
        .get("detection_tags")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let mut seen_by_source_type = BTreeSet::new();
    tags.retain(|value| {
        let Some(source) = value.get("source").and_then(Value::as_str) else {
            return true;
        };
        let Some(tag_type) = value.get("type").and_then(Value::as_str) else {
            return true;
        };
        seen_by_source_type.insert((source.to_string(), tag_type.to_string()))
    });

    for tag in new_tags {
        if !DETECTION_TAG_TYPES.contains(&tag.tag_type.as_str()) {
            continue;
        }
        if !DETECTION_TAG_SOURCES.contains(&tag.source.as_str()) {
            continue;
        }
        let key = (tag.source.clone(), tag.tag_type.clone());
        if !seen_by_source_type.insert(key) {
            tracing::debug!(
                source = %tag.source,
                tag_type = %tag.tag_type,
                "dropping duplicate detection tag from same producer for event"
            );
            continue;
        }
        if let Ok(value) = serde_json::to_value(tag) {
            tracing::debug!(
                target: "feature_extraction",
                tag_type = %value.get("type").and_then(|v| v.as_str()).unwrap_or("unknown"),
                source = %value.get("source").and_then(|v| v.as_str()).unwrap_or("unknown"),
                tool = %bounded_tool_label(value.get("tool").and_then(|v| v.as_str())),
                "detection tag emitted"
            );
            tags.push(value);
        }
    }

    if let Some(obj) = metadata.as_object_mut() {
        obj.insert("detection_tags".to_string(), Value::Array(tags));
    }
    metadata
}

fn bounded_tool_label(value: Option<&str>) -> &'static str {
    match value {
        Some("Read") => "Read",
        Some("Bash") => "Bash",
        Some("unknown") | None => "unknown",
        Some(_) => "other",
    }
}

fn dedupe_tags_within_producer(tags: Vec<DetectionTag>) -> Vec<DetectionTag> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for tag in tags {
        let key = (tag.source.clone(), tag.tag_type.clone());
        if seen.insert(key) {
            out.push(tag);
        } else {
            tracing::debug!(
                source = %tag.source,
                tag_type = %tag.tag_type,
                "dropping duplicate detection tag from same producer"
            );
        }
    }
    out
}

pub fn extract_records(events: &[AgentEvent], config: &ExtractionConfig) -> ExtractionReport {
    let mut groups: BTreeMap<FeatureGroupKey, Vec<AgentEvent>> = BTreeMap::new();
    for event in events {
        let key = if let Some(trajectory_id) = event.trajectory_id {
            FeatureGroupKey::Trajectory(trajectory_id)
        } else {
            let window_start = floor_to_window(event.created_at, config.feature_window_sec);
            let window_end = window_start + Duration::seconds(config.feature_window_sec);
            FeatureGroupKey::SessionWindow {
                repo: event.repo.clone(),
                session_id: event.session_id.clone(),
                window_start,
                window_end,
            }
        };
        groups.entry(key).or_default().push(event.clone());
    }

    let mut report = ExtractionReport::default();
    for (key, mut grouped_events) in groups {
        grouped_events.sort_by(|a, b| {
            a.created_at
                .cmp(&b.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });
        let (record, unknown) = extract_record_for_group(&key, &grouped_events, config);
        report.unknown_tag_schema_versions += unknown;
        report.records.push(record);
    }
    report
}

fn extract_record_for_group(
    key: &FeatureGroupKey,
    events: &[AgentEvent],
    config: &ExtractionConfig,
) -> (FeatureRecord, u64) {
    let first = events
        .first()
        .expect("extract_record_for_group requires non-empty events");
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

    let mut record = FeatureRecord {
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
    };

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

fn apply_tag_to_record(
    event: &AgentEvent,
    tag: &DetectionTag,
    record: &mut FeatureRecord,
    failure_classes: &mut BTreeSet<String>,
) {
    match tag.tag_type.as_str() {
        "tool_loop" => {
            record.tool_loop_count += 1;
            record.loop_detected = true;
            failure_classes.insert("tool_loop".to_string());
            match tag.tool.as_deref() {
                Some("Read") => record.repeated_read_loop_count += 1,
                Some("Bash") => record.repeated_bash_loop_count += 1,
                _ => {}
            }
        }
        "user_interruption" => {
            record.user_interruption_count += 1;
            record.user_interrupted = true;
            failure_classes.insert("user_interruption".to_string());
        }
        "missing_auth" => {
            record.missing_auth_count += 1;
            failure_classes.insert("missing_auth".to_string());
            if let Some(value) = known_fact(event, "auth_header") {
                record.known_auth_header = Some(value.to_string());
            }
        }
        "wrong_endpoint" => {
            record.wrong_endpoint_count += 1;
            failure_classes.insert("wrong_endpoint".to_string());
            if let Some(value) = known_fact(event, "endpoint") {
                record.known_endpoint = Some(value.to_string());
            }
        }
        "summarization_failure" => {
            record.summarization_failure_count += 1;
            failure_classes.insert("summarization_failure".to_string());
        }
        "migration_failure" => {
            record.migration_failure_count += 1;
            failure_classes.insert("migration_failure".to_string());
            if let Some(value) = known_fact(event, "migration_fix") {
                record.known_migration_fix = Some(value.to_string());
            }
        }
        "context_pack_empty" => {
            record.context_pack_empty_count += 1;
            failure_classes.insert("context_pack_empty".to_string());
        }
        "context_pack_truncated" => {
            record.context_pack_truncated_count += 1;
            failure_classes.insert("context_pack_truncated".to_string());
        }
        "high_input_tokens" => {
            record.high_input_token_count += 1;
            failure_classes.insert("high_input_tokens".to_string());
        }
        "slow_upstream_model" => {
            record.slow_upstream_model_count += 1;
            failure_classes.insert("slow_upstream_model".to_string());
        }
        "empty_tool_use_message" => {
            record.empty_tool_use_message_count += 1;
            failure_classes.insert("empty_tool_use_message".to_string());
        }
        "abandoned_before_model" => {
            record.abandoned_before_model_count += 1;
            failure_classes.insert("abandoned_before_model".to_string());
        }
        "single_model_abandoned_no_tools" => {
            record.single_model_abandoned_no_tools_count += 1;
            failure_classes.insert("single_model_abandoned_no_tools".to_string());
        }
        "summarizer_shared_upstream" => {
            record.summarizer_shared_upstream_count += 1;
            failure_classes.insert("summarizer_shared_upstream".to_string());
        }
        "correction_acknowledged" => {}
        _ => {}
    }
}

fn build_constraints(
    record: &mut FeatureRecord,
    latest_detection: &BTreeMap<FailureKey, DateTime<Utc>>,
    latest_recovery: &BTreeMap<FailureKey, DateTime<Utc>>,
    stale_detection_keys: &BTreeSet<FailureKey>,
    max_constraints: usize,
) {
    let mut candidates = Vec::new();
    let mut suppressed = Vec::new();

    let mut active_tool_loop_tools = BTreeSet::new();
    let mut had_fresh_tool_loop = false;
    let mut recovered_tool_loop = false;
    for (key, detected_at) in latest_detection {
        let FailureKey::ToolLoop(tool) = key else {
            continue;
        };
        had_fresh_tool_loop = true;
        if recovery_after_detection(key, *detected_at, latest_recovery) {
            recovered_tool_loop = true;
        } else {
            active_tool_loop_tools.insert(tool.clone());
        }
    }

    if active_detection(
        &FailureKey::MissingAuth,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "use_known_auth",
    ) && record.known_auth_header.is_some()
    {
        candidates.push(OperationalConstraint {
            constraint_type: "use_known_auth".to_string(),
            text: format!(
                "Use `{}` when calling protected orchestrator endpoints.",
                record.known_auth_header.as_deref().unwrap()
            ),
        });
    }
    if active_detection(
        &FailureKey::WrongEndpoint,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "use_known_endpoint",
    ) && record.known_endpoint.is_some()
    {
        candidates.push(OperationalConstraint {
            constraint_type: "use_known_endpoint".to_string(),
            text: format!(
                "Do not use `localhost` for host-side orchestrator testing. The correct endpoint for this environment is `{}`.",
                record.known_endpoint.as_deref().unwrap()
            ),
        });
    }
    if active_detection(
        &FailureKey::MigrationFailure,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "use_known_migration_fix",
    ) && record.known_migration_fix.is_some()
    {
        candidates.push(OperationalConstraint {
            constraint_type: "use_known_migration_fix".to_string(),
            text: format!(
                "When retrying the baseline migration, use `{}` to make extension creation idempotent.",
                record.known_migration_fix.as_deref().unwrap()
            ),
        });
    }

    if !active_tool_loop_tools.is_empty() {
        let tools = active_tool_loop_tools
            .into_iter()
            .collect::<Vec<_>>()
            .join(" and ");
        candidates.push(OperationalConstraint {
            constraint_type: "avoid_tool_loop".to_string(),
            text: format!(
                "Do not repeat identical {tools} tool calls within this trajectory. Summarize the previous result and choose a different action before reusing the same tool with identical parameters."
            ),
        });
    } else if had_fresh_tool_loop && recovered_tool_loop {
        suppressed.push(SuppressedConstraint {
            constraint_type: "avoid_tool_loop".to_string(),
            reason: "recovery_detected".to_string(),
        });
    }

    if active_detection(
        &FailureKey::ContextPackEmpty,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "fix_context_retrieval",
    ) {
        candidates.push(OperationalConstraint {
            constraint_type: "fix_context_retrieval".to_string(),
            text: "If the context pack is empty or near-empty, verify cache warmup and retrieval health before assuming no prior memory exists.".to_string(),
        });
    }

    let context_truncated_active = active_detection(
        &FailureKey::ContextPackTruncated,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "reduce_context_bloat",
    );
    let high_input_active = active_detection(
        &FailureKey::HighInputTokens,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "reduce_context_bloat",
    );
    if context_truncated_active || high_input_active {
        candidates.push(OperationalConstraint {
            constraint_type: "reduce_context_bloat".to_string(),
            text: "Avoid expanding context further; use concise cached memory and inspect why input tokens or context truncation are high before retrying.".to_string(),
        });
    }

    if active_detection(
        &FailureKey::SummarizerSharedUpstream,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "separate_summarizer_upstream",
    ) {
        candidates.push(OperationalConstraint {
            constraint_type: "separate_summarizer_upstream".to_string(),
            text: "Keep background summarization on the dedicated summarizer endpoint instead of sharing foreground LiteLLM capacity.".to_string(),
        });
    }

    if active_detection(
        &FailureKey::UserInterruption,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "handle_user_interruption",
    ) {
        candidates.push(OperationalConstraint {
            constraint_type: "handle_user_interruption".to_string(),
            text: "When the user interrupts due to incorrect operational behavior, explicitly acknowledge the correction and apply it before continuing.".to_string(),
        });
    }

    if active_detection(
        &FailureKey::SummarizationFailure,
        latest_detection,
        latest_recovery,
        &mut suppressed,
        "handle_summarization_failure",
    ) {
        candidates.push(OperationalConstraint {
            constraint_type: "handle_summarization_failure".to_string(),
            text: "If summarization returns an empty response, inspect the provider or LiteLLM response body before retrying.".to_string(),
        });
    }

    for key in stale_detection_keys {
        if latest_detection.contains_key(key) {
            continue;
        }
        if let Some(constraint_type) = constraint_type_for_failure_key(key) {
            suppressed.push(SuppressedConstraint {
                constraint_type: constraint_type.to_string(),
                reason: "freshness_window".to_string(),
            });
        }
    }

    candidates.sort_by_key(|constraint| constraint_priority(&constraint.constraint_type));
    if candidates.len() > max_constraints {
        for dropped in candidates.drain(max_constraints..) {
            suppressed.push(SuppressedConstraint {
                constraint_type: dropped.constraint_type,
                reason: "priority_cap".to_string(),
            });
        }
    }

    suppress_duplicates(&mut suppressed);
    record.recommended_constraints = candidates;
    record.suppressed_constraints = suppressed;
}

fn active_detection(
    key: &FailureKey,
    latest_detection: &BTreeMap<FailureKey, DateTime<Utc>>,
    latest_recovery: &BTreeMap<FailureKey, DateTime<Utc>>,
    suppressed: &mut Vec<SuppressedConstraint>,
    constraint_type: &str,
) -> bool {
    let Some(detected_at) = latest_detection.get(key) else {
        return false;
    };
    if recovery_after_detection(key, *detected_at, latest_recovery) {
        suppressed.push(SuppressedConstraint {
            constraint_type: constraint_type.to_string(),
            reason: "recovery_detected".to_string(),
        });
        return false;
    }
    true
}

fn recovery_after_detection(
    key: &FailureKey,
    detected_at: DateTime<Utc>,
    latest_recovery: &BTreeMap<FailureKey, DateTime<Utc>>,
) -> bool {
    latest_recovery
        .get(key)
        .map(|recovered_at| *recovered_at > detected_at)
        .unwrap_or(false)
}

fn record_latest_recovery(
    latest_recovery: &mut BTreeMap<FailureKey, DateTime<Utc>>,
    key: FailureKey,
    recovered_at: DateTime<Utc>,
) {
    latest_recovery
        .entry(key)
        .and_modify(|time| {
            if recovered_at > *time {
                *time = recovered_at;
            }
        })
        .or_insert(recovered_at);
}

fn constraint_type_for_failure_key(key: &FailureKey) -> Option<&'static str> {
    match key {
        FailureKey::ToolLoop(_) => Some("avoid_tool_loop"),
        FailureKey::UserInterruption => Some("handle_user_interruption"),
        FailureKey::MissingAuth => Some("use_known_auth"),
        FailureKey::WrongEndpoint => Some("use_known_endpoint"),
        FailureKey::SummarizationFailure => Some("handle_summarization_failure"),
        FailureKey::MigrationFailure => Some("use_known_migration_fix"),
        FailureKey::ContextPackEmpty => Some("fix_context_retrieval"),
        FailureKey::ContextPackTruncated | FailureKey::HighInputTokens => {
            Some("reduce_context_bloat")
        }
        FailureKey::SummarizerSharedUpstream => Some("separate_summarizer_upstream"),
        FailureKey::SlowUpstreamModel
        | FailureKey::EmptyToolUseMessage
        | FailureKey::AbandonedBeforeModel
        | FailureKey::SingleModelAbandonedNoTools => None,
    }
}

fn suppress_duplicates(suppressed: &mut Vec<SuppressedConstraint>) {
    let mut seen = BTreeSet::new();
    suppressed.retain(|item| seen.insert((item.constraint_type.clone(), item.reason.clone())));
    suppressed.sort_by_key(|item| {
        (
            constraint_priority(&item.constraint_type),
            item.reason.clone(),
        )
    });
}

fn constraint_priority(constraint_type: &str) -> usize {
    CONSTRAINT_PRIORITY
        .iter()
        .position(|candidate| *candidate == constraint_type)
        .unwrap_or(usize::MAX)
}

pub fn enforce_constraint_token_budget(
    constraints: &[OperationalConstraint],
    budget_tokens: usize,
) -> (Vec<OperationalConstraint>, Vec<SuppressedConstraint>) {
    if constraints.is_empty() || budget_tokens == 0 {
        return (
            vec![],
            constraints
                .iter()
                .map(|constraint| SuppressedConstraint {
                    constraint_type: constraint.constraint_type.clone(),
                    reason: "token_budget".to_string(),
                })
                .collect(),
        );
    }

    let mut kept = constraints.to_vec();
    kept.sort_by_key(|constraint| constraint_priority(&constraint.constraint_type));
    let mut dropped = Vec::new();
    while estimate_constraint_tokens(&kept) > budget_tokens {
        let Some(constraint) = kept.pop() else {
            break;
        };
        dropped.push(SuppressedConstraint {
            constraint_type: constraint.constraint_type,
            reason: "token_budget".to_string(),
        });
    }
    dropped.sort_by_key(|item| constraint_priority(&item.constraint_type));
    (kept, dropped)
}

fn estimate_constraint_tokens(constraints: &[OperationalConstraint]) -> usize {
    let body_len: usize = constraints
        .iter()
        .map(|constraint| constraint.text.len() + 3)
        .sum();
    ("Operational Constraints:\n".len() + body_len).div_ceil(4)
}

fn floor_to_window(ts: DateTime<Utc>, window_sec: i64) -> DateTime<Utc> {
    let seconds = ts.timestamp();
    let window_start = seconds - seconds.rem_euclid(window_sec);
    Utc.timestamp_opt(window_start, 0).single().unwrap_or(ts)
}

#[derive(Default)]
struct ParsedTags {
    tags: Vec<DetectionTag>,
    unknown_schema_versions: u64,
}

fn recognized_tags(event: &AgentEvent) -> ParsedTags {
    let mut parsed = ParsedTags::default();
    let Some(values) = event
        .metadata
        .get("detection_tags")
        .and_then(Value::as_array)
    else {
        return parsed;
    };
    for value in values {
        let schema_version = value
            .get("tag_schema_version")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if schema_version != TAG_SCHEMA_VERSION as u64 {
            parsed.unknown_schema_versions += 1;
            continue;
        }
        let Ok(tag) = serde_json::from_value::<DetectionTag>(value.clone()) else {
            continue;
        };
        if !DETECTION_TAG_TYPES.contains(&tag.tag_type.as_str()) {
            continue;
        }
        if !DETECTION_TAG_SOURCES.contains(&tag.source.as_str()) {
            continue;
        }
        if tag.tag_type == "tool_loop" && tag.tool.as_deref().unwrap_or("").is_empty() {
            continue;
        }
        parsed.tags.push(tag);
    }
    parsed
}

fn extraction_condition_key(tag: &DetectionTag) -> String {
    if tag.tag_type == "tool_loop" {
        format!("tool_loop:{}", tag.tool.as_deref().unwrap_or("unknown"))
    } else {
        tag.tag_type.clone()
    }
}

fn failure_key_for_tag(tag: &DetectionTag) -> Option<FailureKey> {
    match tag.tag_type.as_str() {
        "tool_loop" => Some(FailureKey::ToolLoop(
            tag.tool.clone().unwrap_or_else(|| "unknown".to_string()),
        )),
        "user_interruption" => Some(FailureKey::UserInterruption),
        "missing_auth" => Some(FailureKey::MissingAuth),
        "wrong_endpoint" => Some(FailureKey::WrongEndpoint),
        "summarization_failure" => Some(FailureKey::SummarizationFailure),
        "migration_failure" => Some(FailureKey::MigrationFailure),
        "context_pack_empty" => Some(FailureKey::ContextPackEmpty),
        "context_pack_truncated" => Some(FailureKey::ContextPackTruncated),
        "high_input_tokens" => Some(FailureKey::HighInputTokens),
        "slow_upstream_model" => Some(FailureKey::SlowUpstreamModel),
        "empty_tool_use_message" => Some(FailureKey::EmptyToolUseMessage),
        "abandoned_before_model" => Some(FailureKey::AbandonedBeforeModel),
        "single_model_abandoned_no_tools" => Some(FailureKey::SingleModelAbandonedNoTools),
        "summarizer_shared_upstream" => Some(FailureKey::SummarizerSharedUpstream),
        _ => None,
    }
}

pub async fn load_events_for_scope(
    pool: &Pool,
    scope: &ExtractionScope,
    limit: Option<i64>,
) -> Result<Vec<AgentEvent>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, session_id, repo, actor, event_type, summary, evidence, metadata,
                        correlation_id, parent_event_id, trajectory_id, attempt_index, event_role,
                        created_at, summary_level
                 FROM agent_events
                 WHERE ($1::text IS NULL OR repo = $1)
                   AND ($2::text IS NULL OR session_id = $2)
                   AND ($3::uuid IS NULL OR trajectory_id = $3)
                   AND ($4::timestamptz IS NULL OR created_at >= $4)
                   AND ($5::timestamptz IS NULL OR created_at < $5)
                 ORDER BY created_at ASC, id ASC
                 LIMIT COALESCE($6, 9223372036854775807)",
                &[
                    &scope.repo,
                    &scope.session_id,
                    &scope.trajectory_id,
                    &scope.since,
                    &scope.until,
                    &limit,
                ],
            )
            .await?;
        Ok(rows
            .into_iter()
            .map(|row| AgentEvent {
                id: row.get("id"),
                session_id: row.get("session_id"),
                repo: row.get("repo"),
                actor: row.get("actor"),
                event_type: row.get("event_type"),
                summary: row.get("summary"),
                evidence: row.get("evidence"),
                metadata: row.get("metadata"),
                correlation_id: row.get("correlation_id"),
                parent_event_id: row.get("parent_event_id"),
                trajectory_id: row.get("trajectory_id"),
                attempt_index: row.get("attempt_index"),
                event_role: row.get("event_role"),
                created_at: row.get("created_at"),
                summary_level: row.get("summary_level"),
            })
            .collect())
    }
    .await;
    crate::telemetry::record_db_query("feature_load_events", started.elapsed(), result.is_ok());
    result
}

pub async fn extract_and_persist_scope(
    pool: &Pool,
    scope: &ExtractionScope,
    config: &ExtractionConfig,
    dry_run: bool,
) -> Result<ExtractionReport, anyhow::Error> {
    let events = load_events_for_scope(pool, scope, None).await?;
    let report = extract_records(&events, config);
    if !dry_run {
        for record in &report.records {
            persist_feature_record(pool, record).await?;
        }
    }
    Ok(report)
}

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
            "INSERT INTO agent_feature_records (
                feature_id, trajectory_id, session_id, repo, window_start, window_end,
                event_count, user_message_count, assistant_message_count, failed_attempt_count,
                summary_count, tool_loop_count, repeated_read_loop_count, repeated_bash_loop_count,
                user_interruption_count, missing_auth_count, wrong_endpoint_count,
                summarization_failure_count, migration_failure_count, other_failure_count,
                context_pack_empty_count, context_pack_truncated_count, high_input_token_count,
                slow_upstream_model_count, empty_tool_use_message_count,
                abandoned_before_model_count, single_model_abandoned_no_tools_count,
                summarizer_shared_upstream_count,
                known_endpoint, known_auth_header, known_migration_fix, loop_detected,
                user_interrupted, failure_classes, recommended_constraints, suppressed_constraints,
                created_at, updated_at
             )
             VALUES (
                $1, $2, $3, $4, $5, $6,
                $7, $8, $9, $10,
                $11, $12, $13, $14,
                $15, $16, $17,
                $18, $19, $20,
                $21, $22, $23, $24,
                $25, $26, $27, $28,
                $29, $30, $31, $32,
                $33, $34, $35, $36,
                now(), now()
             )
             ON CONFLICT (feature_id)
             DO UPDATE SET
                trajectory_id = EXCLUDED.trajectory_id,
                session_id = EXCLUDED.session_id,
                repo = EXCLUDED.repo,
                window_start = EXCLUDED.window_start,
                window_end = EXCLUDED.window_end,
                event_count = EXCLUDED.event_count,
                user_message_count = EXCLUDED.user_message_count,
                assistant_message_count = EXCLUDED.assistant_message_count,
                failed_attempt_count = EXCLUDED.failed_attempt_count,
                summary_count = EXCLUDED.summary_count,
                tool_loop_count = EXCLUDED.tool_loop_count,
                repeated_read_loop_count = EXCLUDED.repeated_read_loop_count,
                repeated_bash_loop_count = EXCLUDED.repeated_bash_loop_count,
                user_interruption_count = EXCLUDED.user_interruption_count,
                missing_auth_count = EXCLUDED.missing_auth_count,
                wrong_endpoint_count = EXCLUDED.wrong_endpoint_count,
                summarization_failure_count = EXCLUDED.summarization_failure_count,
                migration_failure_count = EXCLUDED.migration_failure_count,
                other_failure_count = EXCLUDED.other_failure_count,
                context_pack_empty_count = EXCLUDED.context_pack_empty_count,
                context_pack_truncated_count = EXCLUDED.context_pack_truncated_count,
                high_input_token_count = EXCLUDED.high_input_token_count,
                slow_upstream_model_count = EXCLUDED.slow_upstream_model_count,
                empty_tool_use_message_count = EXCLUDED.empty_tool_use_message_count,
                abandoned_before_model_count = EXCLUDED.abandoned_before_model_count,
                single_model_abandoned_no_tools_count = EXCLUDED.single_model_abandoned_no_tools_count,
                summarizer_shared_upstream_count = EXCLUDED.summarizer_shared_upstream_count,
                known_endpoint = EXCLUDED.known_endpoint,
                known_auth_header = EXCLUDED.known_auth_header,
                known_migration_fix = EXCLUDED.known_migration_fix,
                loop_detected = EXCLUDED.loop_detected,
                user_interrupted = EXCLUDED.user_interrupted,
                failure_classes = EXCLUDED.failure_classes,
                recommended_constraints = EXCLUDED.recommended_constraints,
                suppressed_constraints = EXCLUDED.suppressed_constraints,
                updated_at = CASE
                    WHEN (
                        agent_feature_records.trajectory_id,
                        agent_feature_records.session_id,
                        agent_feature_records.repo,
                        agent_feature_records.window_start,
                        agent_feature_records.window_end,
                        agent_feature_records.event_count,
                        agent_feature_records.user_message_count,
                        agent_feature_records.assistant_message_count,
                        agent_feature_records.failed_attempt_count,
                        agent_feature_records.summary_count,
                        agent_feature_records.tool_loop_count,
                        agent_feature_records.repeated_read_loop_count,
                        agent_feature_records.repeated_bash_loop_count,
                        agent_feature_records.user_interruption_count,
                        agent_feature_records.missing_auth_count,
                        agent_feature_records.wrong_endpoint_count,
                        agent_feature_records.summarization_failure_count,
                        agent_feature_records.migration_failure_count,
                        agent_feature_records.other_failure_count,
                        agent_feature_records.context_pack_empty_count,
                        agent_feature_records.context_pack_truncated_count,
                        agent_feature_records.high_input_token_count,
                        agent_feature_records.slow_upstream_model_count,
                        agent_feature_records.empty_tool_use_message_count,
                        agent_feature_records.abandoned_before_model_count,
                        agent_feature_records.single_model_abandoned_no_tools_count,
                        agent_feature_records.summarizer_shared_upstream_count,
                        agent_feature_records.known_endpoint,
                        agent_feature_records.known_auth_header,
                        agent_feature_records.known_migration_fix,
                        agent_feature_records.loop_detected,
                        agent_feature_records.user_interrupted,
                        agent_feature_records.failure_classes,
                        agent_feature_records.recommended_constraints,
                        agent_feature_records.suppressed_constraints
                    ) IS DISTINCT FROM (
                        EXCLUDED.trajectory_id,
                        EXCLUDED.session_id,
                        EXCLUDED.repo,
                        EXCLUDED.window_start,
                        EXCLUDED.window_end,
                        EXCLUDED.event_count,
                        EXCLUDED.user_message_count,
                        EXCLUDED.assistant_message_count,
                        EXCLUDED.failed_attempt_count,
                        EXCLUDED.summary_count,
                        EXCLUDED.tool_loop_count,
                        EXCLUDED.repeated_read_loop_count,
                        EXCLUDED.repeated_bash_loop_count,
                        EXCLUDED.user_interruption_count,
                        EXCLUDED.missing_auth_count,
                        EXCLUDED.wrong_endpoint_count,
                        EXCLUDED.summarization_failure_count,
                        EXCLUDED.migration_failure_count,
                        EXCLUDED.other_failure_count,
                        EXCLUDED.context_pack_empty_count,
                        EXCLUDED.context_pack_truncated_count,
                        EXCLUDED.high_input_token_count,
                        EXCLUDED.slow_upstream_model_count,
                        EXCLUDED.empty_tool_use_message_count,
                        EXCLUDED.abandoned_before_model_count,
                        EXCLUDED.single_model_abandoned_no_tools_count,
                        EXCLUDED.summarizer_shared_upstream_count,
                        EXCLUDED.known_endpoint,
                        EXCLUDED.known_auth_header,
                        EXCLUDED.known_migration_fix,
                        EXCLUDED.loop_detected,
                        EXCLUDED.user_interrupted,
                        EXCLUDED.failure_classes,
                        EXCLUDED.recommended_constraints,
                        EXCLUDED.suppressed_constraints
                    )
                    THEN now()
                    ELSE agent_feature_records.updated_at
                END",
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

pub async fn operational_constraints_for_context(
    pool: &Pool,
    repo: &str,
    session_id: Option<&str>,
    trajectory_id: Option<Uuid>,
    token_budget: usize,
) -> Result<(Vec<OperationalConstraint>, Vec<SuppressedConstraint>), anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let row = if let Some(trajectory_id) = trajectory_id {
            conn.query_opt(
                "SELECT recommended_constraints
                 FROM agent_feature_records
                 WHERE repo = $1 AND trajectory_id = $2
                 ORDER BY updated_at DESC
                 LIMIT 1",
                &[&repo, &trajectory_id],
            )
            .await?
        } else if let Some(session_id) = session_id {
            conn.query_opt(
                "SELECT recommended_constraints
                 FROM agent_feature_records
                 WHERE repo = $1 AND session_id = $2
                 ORDER BY window_end DESC, updated_at DESC
                 LIMIT 1",
                &[&repo, &session_id],
            )
            .await?
        } else {
            conn.query_opt(
                "SELECT recommended_constraints
                 FROM agent_feature_records
                 WHERE repo = $1
                 ORDER BY updated_at DESC
                 LIMIT 1",
                &[&repo],
            )
            .await?
        };

        let constraints = row
            .map(|row| row.get::<_, Value>("recommended_constraints"))
            .and_then(|value| serde_json::from_value::<Vec<OperationalConstraint>>(value).ok())
            .unwrap_or_default();
        Ok(enforce_constraint_token_budget(&constraints, token_budget))
    }
    .await;
    crate::telemetry::record_db_query(
        "feature_context_constraints",
        started.elapsed(),
        result.is_ok(),
    );
    result
}

pub async fn bootstrap_tag_events(
    pool: &Pool,
    scope: &ExtractionScope,
    batch_size: i64,
    dry_run: bool,
) -> Result<i64, anyhow::Error> {
    let events = load_events_for_scope(pool, scope, None).await?;
    let mut updates = Vec::new();
    for event in events {
        let updated = bootstrap_annotate_event_metadata(&event);
        if updated != event.metadata {
            updates.push((event.id, updated));
        }
    }
    if dry_run || updates.is_empty() {
        return Ok(updates.len() as i64);
    }

    let mut conn = pool.get().await?;
    let mut changed = 0i64;
    for chunk in updates.chunks(batch_size.max(1) as usize) {
        let tx = conn.deref_mut().deref_mut().transaction().await?;
        for (id, metadata) in chunk {
            changed += tx
                .execute(
                    "UPDATE agent_events
                     SET metadata = $2
                     WHERE id = $1 AND metadata IS DISTINCT FROM $2",
                    &[id, metadata],
                )
                .await? as i64;
        }
        tx.commit().await?;
    }
    Ok(changed)
}

pub async fn run_backfill(
    pool: &Pool,
    opts: &BackfillOptions,
    config: &ExtractionConfig,
) -> Result<BackfillReport, anyhow::Error> {
    let scope = ExtractionScope {
        repo: opts.repo.clone(),
        session_id: opts.session_id.clone(),
        trajectory_id: opts.trajectory_id,
        since: opts.since,
        until: None,
    };
    let mut report = BackfillReport {
        dry_run: opts.dry_run,
        ..Default::default()
    };
    if !opts.skip_bootstrap_tagging {
        report.bootstrap_updated =
            bootstrap_tag_events(pool, &scope, opts.batch_size, opts.dry_run).await?;
    }

    let mut events = load_events_for_scope(pool, &scope, None).await?;
    if opts.dry_run && !opts.skip_bootstrap_tagging {
        for event in &mut events {
            event.metadata = bootstrap_annotate_event_metadata(event);
        }
    }
    report.events_scanned = events.len();
    let extraction = extract_records(&events, config);
    report.feature_records = extraction.records.len();
    if !opts.dry_run {
        for record in extraction.records {
            persist_feature_record(pool, &record).await?;
            report.persisted_records += 1;
        }
    }
    Ok(report)
}

pub async fn run_inline_extraction_best_effort(
    pool: Pool,
    metrics: crate::telemetry::MetricsRegistry,
    repo: String,
    session_id: String,
    trajectory_id: Option<Uuid>,
) {
    let started = std::time::Instant::now();
    let config = ExtractionConfig {
        feature_window_sec: feature_window_sec_from_env(),
        constraint_freshness_window_sec: constraint_freshness_window_sec_from_env(),
        max_operational_constraints: max_operational_constraints_from_env(),
        evaluation_time: Utc::now(),
    };
    let scope = ExtractionScope {
        repo: Some(repo.clone()),
        session_id: Some(session_id),
        trajectory_id,
        since: None,
        until: None,
    };

    let events = match load_events_for_scope(&pool, &scope, None).await {
        Ok(events) => events,
        Err(e) => {
            tracing::warn!(
                target: "feature_extraction",
                repo = %repo,
                stage = "extraction",
                "inline feature extraction failed to load events: {e}"
            );
            crate::telemetry::record_feature_extraction_failure("extraction");
            return;
        }
    };
    let report = extract_records(&events, &config);
    for _ in 0..report.unknown_tag_schema_versions {
        crate::telemetry::record_unknown_tag_schema_version();
    }
    for record in &report.records {
        for class in &record.failure_classes {
            crate::telemetry::record_feature_failure_class(class);
        }
        if let Err(e) = persist_feature_record(&pool, record).await {
            tracing::warn!(
                target: "feature_extraction",
                repo = %repo,
                stage = "persistence",
                "inline feature extraction failed to persist record: {e}"
            );
            crate::telemetry::record_feature_extraction_failure("persistence");
            continue;
        }
        crate::telemetry::record_trajectory_feature_written(&metrics);
    }
    crate::telemetry::record_feature_extraction_duration(started.elapsed());
    drop(metrics);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(id: &str, event_type: &str, metadata: Value, created_at: DateTime<Utc>) -> AgentEvent {
        AgentEvent {
            id: id.to_string(),
            session_id: "s1".to_string(),
            repo: "repo".to_string(),
            actor: "agent".to_string(),
            event_type: event_type.to_string(),
            summary: event_type.to_string(),
            evidence: None,
            metadata,
            correlation_id: None,
            parent_event_id: None,
            trajectory_id: None,
            attempt_index: None,
            event_role: None,
            created_at,
            summary_level: 0,
        }
    }

    fn config(now: DateTime<Utc>) -> ExtractionConfig {
        ExtractionConfig {
            evaluation_time: now,
            feature_window_sec: 3600,
            constraint_freshness_window_sec: 1800,
            max_operational_constraints: 5,
        }
    }

    #[test]
    fn unknown_schema_tags_are_not_consumed() {
        let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
        let events = vec![event(
            "e1",
            "failed_attempt",
            json!({
                "detection_tags": [{
                    "type": "wrong_endpoint",
                    "source": "hook_parser",
                    "tag_schema_version": 99
                }]
            }),
            now,
        )];

        let report = extract_records(&events, &config(now));
        let record = &report.records[0];
        assert_eq!(report.unknown_tag_schema_versions, 1);
        assert_eq!(record.wrong_endpoint_count, 0);
        assert_eq!(record.other_failure_count, 1);
    }

    #[test]
    fn same_condition_from_two_producers_is_retained_but_not_double_counted() {
        let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
        let events = vec![event(
            "e1",
            "failed_attempt",
            json!({
                "detection_tags": [
                    {"type": "tool_loop", "tool": "Read", "source": "hook_parser", "tag_schema_version": 1},
                    {"type": "tool_loop", "tool": "Read", "source": "summarizer", "tag_schema_version": 1}
                ]
            }),
            now,
        )];

        let report = extract_records(&events, &config(now));
        let record = &report.records[0];
        assert_eq!(record.tool_loop_count, 1);
        assert_eq!(record.repeated_read_loop_count, 1);
        assert_eq!(record.other_failure_count, 0);
        assert_eq!(record.recommended_constraints.len(), 1);
        assert_eq!(
            record.recommended_constraints[0].constraint_type,
            "avoid_tool_loop"
        );
    }

    #[test]
    fn known_facts_feed_constraint_templates() {
        let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
        let events = vec![
            event(
                "e1",
                "failed_attempt",
                json!({
                    "known_facts": {"auth_header": "Bearer test"},
                    "detection_tags": [{"type": "missing_auth", "source": "validation_parser", "tag_schema_version": 1}]
                }),
                now,
            ),
            event(
                "e2",
                "failed_attempt",
                json!({
                    "known_facts": {"endpoint": "http://host.docker.internal:8088"},
                    "detection_tags": [{"type": "wrong_endpoint", "source": "validation_parser", "tag_schema_version": 1}]
                }),
                now + Duration::seconds(1),
            ),
        ];

        let report = extract_records(&events, &config(now + Duration::seconds(1)));
        let record = &report.records[0];
        assert_eq!(record.missing_auth_count, 1);
        assert_eq!(record.wrong_endpoint_count, 1);
        assert_eq!(record.known_auth_header.as_deref(), Some("Bearer test"));
        assert_eq!(
            record.known_endpoint.as_deref(),
            Some("http://host.docker.internal:8088")
        );
        assert!(record
            .recommended_constraints
            .iter()
            .any(|constraint| constraint.constraint_type == "use_known_auth"));
        assert!(record
            .recommended_constraints
            .iter()
            .any(|constraint| constraint.constraint_type == "use_known_endpoint"));
    }

    #[test]
    fn stale_detection_suppresses_by_freshness_window() {
        let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
        let old = now - Duration::seconds(4000);
        let events = vec![event(
            "e1",
            "failed_attempt",
            json!({
                "detection_tags": [{"type": "summarization_failure", "source": "summarizer", "tag_schema_version": 1}]
            }),
            old,
        )];

        let report = extract_records(&events, &config(now));
        let record = &report.records[0];
        assert_eq!(record.summarization_failure_count, 1);
        assert!(record.recommended_constraints.is_empty());
        assert!(record.suppressed_constraints.iter().any(|item| {
            item.constraint_type == "handle_summarization_failure"
                && item.reason == "freshness_window"
        }));
    }

    #[test]
    fn fresh_recovery_suppresses_then_new_detection_resets() {
        let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
        let detection = event(
            "e1",
            "failed_attempt",
            json!({
                "detection_tags": [{"type": "tool_loop", "tool": "Bash", "source": "tool_loop_detector", "tag_schema_version": 1}]
            }),
            now,
        );
        let recovery = event(
            "e2",
            crate::execution_feedback::EVENT_TYPE_TOOL_RESULT,
            json!({"success": true, "payload": {"tool_name": "Bash"}}),
            now + Duration::seconds(10),
        );

        let recovered = extract_records(
            &[detection.clone(), recovery.clone()],
            &config(now + Duration::seconds(20)),
        );
        assert!(recovered.records[0].recommended_constraints.is_empty());
        assert!(recovered.records[0]
            .suppressed_constraints
            .iter()
            .any(|item| item.constraint_type == "avoid_tool_loop"
                && item.reason == "recovery_detected"));

        let new_detection = event(
            "e3",
            "failed_attempt",
            json!({
                "detection_tags": [{"type": "tool_loop", "tool": "Bash", "source": "tool_loop_detector", "tag_schema_version": 1}]
            }),
            now + Duration::seconds(30),
        );
        let reset = extract_records(
            &[detection, recovery, new_detection],
            &config(now + Duration::seconds(40)),
        );
        assert!(reset.records[0]
            .recommended_constraints
            .iter()
            .any(|item| item.constraint_type == "avoid_tool_loop"));
    }

    #[test]
    fn stale_recovery_does_not_suppress_fresh_failure() {
        let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
        let stale_recovery = event(
            "e1",
            crate::execution_feedback::EVENT_TYPE_TOOL_RESULT,
            json!({"success": true, "payload": {"tool_name": "Read"}}),
            now - Duration::seconds(4000),
        );
        let detection = event(
            "e2",
            "failed_attempt",
            json!({
                "detection_tags": [{"type": "tool_loop", "tool": "Read", "source": "tool_loop_detector", "tag_schema_version": 1}]
            }),
            now,
        );

        let report = extract_records(&[stale_recovery, detection], &config(now));
        let record = report
            .records
            .iter()
            .find(|record| record.tool_loop_count == 1)
            .unwrap();
        assert!(record
            .recommended_constraints
            .iter()
            .any(|item| item.constraint_type == "avoid_tool_loop"));
    }

    #[test]
    fn priority_cap_keeps_highest_priority_constraints() {
        let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
        let mut cfg = config(now);
        cfg.max_operational_constraints = 4;
        let events = vec![
            event(
                "e1",
                "failed_attempt",
                json!({"known_facts": {"auth_header": "Bearer t"}, "detection_tags": [{"type": "missing_auth", "source": "validation_parser", "tag_schema_version": 1}]}),
                now,
            ),
            event(
                "e2",
                "failed_attempt",
                json!({"known_facts": {"endpoint": "http://host:8088"}, "detection_tags": [{"type": "wrong_endpoint", "source": "validation_parser", "tag_schema_version": 1}]}),
                now,
            ),
            event(
                "e3",
                "failed_attempt",
                json!({"known_facts": {"migration_fix": "CREATE EXTENSION IF NOT EXISTS vector"}, "detection_tags": [{"type": "migration_failure", "source": "validation_parser", "tag_schema_version": 1}]}),
                now,
            ),
            event(
                "e4",
                "failed_attempt",
                json!({"detection_tags": [{"type": "tool_loop", "tool": "Read", "source": "tool_loop_detector", "tag_schema_version": 1}]}),
                now,
            ),
            event(
                "e5",
                "failed_attempt",
                json!({"detection_tags": [{"type": "user_interruption", "source": "failed_attempt_classifier", "tag_schema_version": 1}]}),
                now,
            ),
            event(
                "e6",
                "failed_attempt",
                json!({"detection_tags": [{"type": "summarization_failure", "source": "summarizer", "tag_schema_version": 1}]}),
                now,
            ),
        ];

        let record = &extract_records(&events, &cfg).records[0];
        let types: Vec<_> = record
            .recommended_constraints
            .iter()
            .map(|constraint| constraint.constraint_type.as_str())
            .collect();
        assert_eq!(
            types,
            vec![
                "use_known_auth",
                "use_known_endpoint",
                "use_known_migration_fix",
                "avoid_tool_loop"
            ]
        );
        assert!(record.suppressed_constraints.iter().any(|item| {
            item.constraint_type == "handle_user_interruption" && item.reason == "priority_cap"
        }));
        assert!(record.suppressed_constraints.iter().any(|item| {
            item.constraint_type == "handle_summarization_failure" && item.reason == "priority_cap"
        }));
    }

    #[test]
    fn read_and_bash_loops_combine_before_priority_cap() {
        let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
        let events = vec![
            event(
                "e1",
                "failed_attempt",
                json!({"detection_tags": [{"type": "tool_loop", "tool": "Read", "source": "tool_loop_detector", "tag_schema_version": 1}]}),
                now,
            ),
            event(
                "e2",
                "failed_attempt",
                json!({"detection_tags": [{"type": "tool_loop", "tool": "Bash", "source": "tool_loop_detector", "tag_schema_version": 1}]}),
                now,
            ),
        ];
        let record = &extract_records(&events, &config(now)).records[0];
        assert_eq!(record.recommended_constraints.len(), 1);
        let text = &record.recommended_constraints[0].text;
        assert!(text.contains("Read"));
        assert!(text.contains("Bash"));
    }

    #[test]
    fn live_operational_health_tags_populate_counters() {
        let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
        let events = vec![
            event(
                "e1",
                "context_pack",
                annotate_event_metadata(
                    "context_pack",
                    "",
                    None,
                    json!({
                        "total_context_tokens": 26,
                        "retrieved_event_ids": [],
                        "memory_levels_used": [],
                        "truncated": true
                    }),
                ),
                now,
            ),
            event(
                "e2",
                "assistant_message",
                annotate_event_metadata(
                    "assistant_message",
                    "",
                    None,
                    json!({
                        "input_tokens": 120000,
                        "latency_ms": 65000,
                        "finish_reason": "tool_use"
                    }),
                ),
                now + Duration::seconds(1),
            ),
            event(
                "e3",
                "trajectory_result",
                annotate_event_metadata(
                    "trajectory_result",
                    "abandoned before model",
                    None,
                    json!({
                        "final_status": "abandoned",
                        "total_model_calls": 0,
                        "total_input_tokens": 125000,
                        "total_latency_ms": 70000
                    }),
                ),
                now + Duration::seconds(2),
            ),
            event(
                "e4",
                "trajectory_result",
                annotate_event_metadata(
                    "trajectory_result",
                    "single call abandoned",
                    None,
                    json!({
                        "final_status": "abandoned",
                        "total_model_calls": 1,
                        "total_tool_calls": 0,
                        "total_validations": 0
                    }),
                ),
                now + Duration::seconds(3),
            ),
            event(
                "e5",
                "config_check",
                annotate_event_metadata(
                    "config_check",
                    "shared summarizer upstream",
                    None,
                    json!({
                        "summarizer_base_url": "http://litellm:4000",
                        "litellm_url": "http://litellm:4000/"
                    }),
                ),
                now + Duration::seconds(4),
            ),
        ];

        let record = &extract_records(&events, &config(now + Duration::seconds(4))).records[0];
        assert_eq!(record.context_pack_empty_count, 1);
        assert_eq!(record.context_pack_truncated_count, 1);
        assert_eq!(record.high_input_token_count, 2);
        assert_eq!(record.slow_upstream_model_count, 2);
        assert_eq!(record.empty_tool_use_message_count, 1);
        assert_eq!(record.abandoned_before_model_count, 1);
        assert_eq!(record.single_model_abandoned_no_tools_count, 1);
        assert_eq!(record.summarizer_shared_upstream_count, 1);
        assert!(record
            .recommended_constraints
            .iter()
            .any(|item| item.constraint_type == "fix_context_retrieval"));
        assert!(record
            .recommended_constraints
            .iter()
            .any(|item| item.constraint_type == "reduce_context_bloat"));
        assert!(record
            .recommended_constraints
            .iter()
            .any(|item| item.constraint_type == "separate_summarizer_upstream"));
    }

    #[test]
    fn bootstrap_catches_loop_warning_and_bracketed_interruption() {
        let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
        let read_loop = AgentEvent {
            summary: "LOOP WARNING: Tool 'Read' called 3 times with identical params".to_string(),
            ..event("e1", "failed_attempt", json!({}), now)
        };
        let bash_loop = AgentEvent {
            summary: "LOOP WARNING: Tool 'Bash' called 3 times with identical params".to_string(),
            ..event("e2", "failed_attempt", json!({}), now)
        };
        let interrupted = AgentEvent {
            summary: "[Request interrupted by user for tool use]".to_string(),
            ..event("e3", "failed_attempt", json!({}), now)
        };

        assert!(bootstrap_detection_tags_for_event(&read_loop)
            .iter()
            .any(|tag| tag.tag_type == "tool_loop" && tag.tool.as_deref() == Some("Read")));
        assert!(bootstrap_detection_tags_for_event(&bash_loop)
            .iter()
            .any(|tag| tag.tag_type == "tool_loop" && tag.tool.as_deref() == Some("Bash")));
        assert!(bootstrap_detection_tags_for_event(&interrupted)
            .iter()
            .any(|tag| tag.tag_type == "user_interruption"));
    }

    #[test]
    fn token_budget_drops_lowest_priority_first() {
        let constraints = vec![
            OperationalConstraint { constraint_type: "use_known_auth".to_string(), text: "Use `Bearer t` when calling protected orchestrator endpoints.".to_string() },
            OperationalConstraint { constraint_type: "handle_summarization_failure".to_string(), text: "If summarization returns an empty response, inspect the provider or LiteLLM response body before retrying.".to_string() },
        ];
        let (kept, suppressed) = enforce_constraint_token_budget(&constraints, 30);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].constraint_type, "use_known_auth");
        assert_eq!(
            suppressed[0].constraint_type,
            "handle_summarization_failure"
        );
        assert_eq!(suppressed[0].reason, "token_budget");
    }

    #[test]
    fn bootstrap_structured_detection_matches_live_except_source() {
        let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
        let metadata = json!({
            "failure_class": "wrong_endpoint",
            "known_facts": {"endpoint": "http://host.docker.internal:8088"}
        });
        let live = live_detection_tags_for_event("failed_attempt", "bad endpoint", None, &metadata)
            .into_iter()
            .map(|mut tag| {
                tag.source = "bootstrap_migration".to_string();
                tag
            })
            .collect::<Vec<_>>();
        let historical = event("e1", "failed_attempt", metadata, now);
        let bootstrap = bootstrap_detection_tags_for_event(&historical);
        assert_eq!(live, bootstrap);
    }

    #[test]
    fn backward_compatible_extra_fields_do_not_break_extraction() {
        let now = Utc.with_ymd_and_hms(2026, 5, 19, 12, 0, 0).unwrap();
        let events = vec![event(
            "e1",
            "failed_attempt",
            json!({
                "known_facts": {"endpoint": "http://host.docker.internal:8088"},
                "detection_tags": [{
                    "type": "wrong_endpoint",
                    "source": "validation_parser",
                    "tag_schema_version": 1,
                    "new_optional_field": "ok"
                }]
            }),
            now,
        )];

        let record = &extract_records(&events, &config(now)).records[0];
        assert_eq!(record.wrong_endpoint_count, 1);
        assert!(record
            .recommended_constraints
            .iter()
            .any(|item| item.constraint_type == "use_known_endpoint"));
    }
}
