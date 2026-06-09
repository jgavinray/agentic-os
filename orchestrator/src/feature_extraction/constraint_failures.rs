use crate::feature_extraction_types::DetectionTag;
use chrono::{DateTime, Utc};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FailureKey {
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

pub(crate) fn recovery_after_detection(
    key: &FailureKey,
    detected_at: DateTime<Utc>,
    latest_recovery: &BTreeMap<FailureKey, DateTime<Utc>>,
) -> bool {
    latest_recovery
        .get(key)
        .map(|recovered_at| *recovered_at > detected_at)
        .unwrap_or(false)
}

pub(crate) fn record_latest_recovery(
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

pub(crate) fn constraint_type_for_failure_key(key: &FailureKey) -> Option<&'static str> {
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

pub(crate) fn failure_key_for_tag(tag: &DetectionTag) -> Option<FailureKey> {
    match tag.tag_type.as_str() {
        "tool_loop" => Some(FailureKey::ToolLoop(
            tag.tool.as_deref().unwrap_or("unknown").to_string(),
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
        "correction_acknowledged" => None,
        _ => None,
    }
}
