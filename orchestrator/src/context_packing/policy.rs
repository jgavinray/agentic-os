use crate::db;
use crate::state::{ContextPolicy, SearchHit, TaskCategory};

pub(crate) fn context_task_category(
    task: &str,
    classification: Option<&crate::request_classification::RequestClassification>,
) -> TaskCategory {
    use crate::request_classification::{RequestComplexity, RequestIntent};
    let Some(classification) = classification else {
        return TaskCategory::from_task(task);
    };
    match classification.complexity {
        RequestComplexity::L0Trivial | RequestComplexity::L1Simple => TaskCategory::Narrow,
        RequestComplexity::L2Moderate => {
            if matches!(
                classification.intent,
                RequestIntent::Explain | RequestIntent::Search | RequestIntent::Classify
            ) {
                TaskCategory::Narrow
            } else {
                TaskCategory::Moderate
            }
        }
        RequestComplexity::L3Complex | RequestComplexity::L4ToolRequired => TaskCategory::Moderate,
        RequestComplexity::L5HighRisk | RequestComplexity::Unknown => TaskCategory::Narrow,
    }
}

pub(crate) fn policy_context_cache_key(
    policy: &crate::orchestration_policy::OrchestrationPolicy,
) -> String {
    let context = policy
        .context_sources
        .iter()
        .map(|source| source.as_str())
        .collect::<Vec<_>>()
        .join(".");
    format!(
        "{}.{}.{}",
        policy.edit_policy.as_str(),
        policy.validation_policy.as_str(),
        context
    )
}

pub(crate) fn orchestration_context_source_allowed(
    policy: Option<&crate::orchestration_policy::OrchestrationPolicy>,
    source: crate::orchestration_policy::ContextSource,
) -> bool {
    policy
        .map(|policy| policy.context_sources.contains(&source))
        .unwrap_or(true)
}

pub(crate) fn apply_orchestration_context_limits(
    policy: &mut ContextPolicy,
    orchestration_policy: &crate::orchestration_policy::OrchestrationPolicy,
) {
    if !orchestration_policy
        .context_sources
        .contains(&crate::orchestration_policy::ContextSource::PostgresEvents)
    {
        policy.l0_recent_limit = 0;
        policy.l1_limit = 0;
        policy.l2_limit = 0;
        policy.l3_limit = 0;
        policy.failure_limit = 0;
    }
    if !orchestration_policy
        .context_sources
        .contains(&crate::orchestration_policy::ContextSource::CompiledSummaries)
    {
        policy.l1_limit = policy.l1_limit.min(1);
        policy.l2_limit = 0;
        policy.l3_limit = 0;
    }
    if !orchestration_policy
        .context_sources
        .contains(&crate::orchestration_policy::ContextSource::ContextLedger)
    {
        policy.failure_limit = 0;
    }
}

pub(crate) fn retrieved_event_ids(
    evidence: &db::ContextEvidence,
    hits: &[SearchHit],
) -> Vec<String> {
    evidence
        .l0_recent
        .iter()
        .chain(evidence.l1_matching.iter())
        .chain(evidence.l2_repo.iter())
        .chain(evidence.l3_project.iter())
        .chain(evidence.failures.iter())
        .chain(
            evidence
                .failure_history
                .iter()
                .flat_map(|item| std::iter::once(&item.failure).chain(item.remediation.iter())),
        )
        .map(|event| event.id.clone())
        .chain(hits.iter().map(|hit| hit.event_id.clone()))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn memory_levels_used(evidence: &db::ContextEvidence) -> Vec<String> {
    let mut levels = Vec::new();
    if !evidence.l0_recent.is_empty() {
        levels.push("l0".to_string());
    }
    if !evidence.l1_matching.is_empty() {
        levels.push("l1".to_string());
    }
    if !evidence.l2_repo.is_empty() {
        levels.push("l2".to_string());
    }
    if !evidence.l3_project.is_empty() {
        levels.push("l3".to_string());
    }
    if !evidence.failures.is_empty() {
        levels.push("failures".to_string());
    }
    if !evidence.failure_history.is_empty() {
        levels.push("failure_history".to_string());
    }
    if !evidence.operational_constraints.is_empty() {
        levels.push("operational_constraints".to_string());
    }
    levels
}
