use crate::context_artifacts::{self, ContextArtifact};
use crate::context_compiler::CompilerRequest;
use crate::db;

pub(crate) fn compiler_source_allowed(
    request: &CompilerRequest,
    source: crate::orchestration_policy::ContextSource,
) -> bool {
    request
        .policy
        .as_ref()
        .map(|policy| policy.context_sources.contains(&source))
        .unwrap_or(true)
}

pub(crate) fn compile_service_topology(request: &CompilerRequest) -> ContextArtifact {
    context_artifacts::service_topology_artifact(context_artifacts::ServiceTopologyInput {
        repo: request.repo.clone(),
        default_model: request.runtime.default_model.clone(),
        litellm_url: request.runtime.litellm_url.clone(),
        qdrant_url: request.runtime.qdrant_url.clone(),
        summarizer_url: request.runtime.summarizer_url.clone(),
        summarizer_model: request.runtime.summarizer_model.clone(),
    })
}

pub(crate) fn compile_repo_map(request: &CompilerRequest) -> ContextArtifact {
    context_artifacts::repo_map_artifact(context_artifacts::RepoMapInput {
        repo: request.repo.clone(),
    })
}

pub(crate) async fn total_recall_candidates(
    http: &reqwest::Client,
    total_recall_url: &str,
    repo: &str,
    task: &str,
) -> Result<Vec<crate::total_recall::MemoryNote>, anyhow::Error> {
    let query = format!("{repo} {task}");
    let (recent, searched) = tokio::join!(
        crate::total_recall::recent_notes(http, total_recall_url, 30, 10),
        crate::total_recall::search_notes(http, total_recall_url, &query, 10),
    );
    let mut notes = recent?;
    notes.extend(searched?);
    let mut seen = std::collections::HashSet::new();
    notes.retain(|note| seen.insert(note.id.clone()));
    Ok(notes)
}

pub(crate) fn artifact_source_event_ids(
    artifact: &ContextArtifact,
) -> std::collections::HashSet<String> {
    artifact
        .source_event_ids
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_string)
        .collect()
}

pub(crate) async fn record_ledger(
    pool: &deadpool_postgres::Pool,
    repo: &str,
    artifact: &ContextArtifact,
    candidate_source: &str,
    candidate_id: Option<&str>,
    decision: &str,
    reason: &str,
    metadata: serde_json::Value,
) {
    if let Err(e) = db::insert_context_compiler_ledger_entry(
        pool,
        repo,
        &artifact.artifact_type,
        candidate_source,
        candidate_id,
        decision,
        reason,
        Some(artifact.id),
        metadata,
    )
    .await
    {
        tracing::warn!(
            target: "context_compiler",
            repo = %repo,
            artifact_type = %artifact.artifact_type,
            "failed to write compiler ledger entry: {e}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_compiler::RuntimeContext;

    #[test]
    fn service_topology_compilation_is_deterministic() {
        let request = CompilerRequest {
            repo: "agentic-os".to_string(),
            task: "debug compiler".to_string(),
            session_id: None,
            policy: None,
            runtime: RuntimeContext {
                default_model: "qwen36-35b-a3b".to_string(),
                litellm_url: "http://litellm:4000/v1".to_string(),
                qdrant_url: "http://qdrant:6333".to_string(),
                summarizer_url: "http://summarizer:8080/v1".to_string(),
                summarizer_model: "qwen2.5-3b".to_string(),
                total_recall_url: None,
            },
        };

        let a = compile_service_topology(&request);
        let b = compile_service_topology(&request);
        assert_eq!(a.invalidation_key, b.invalidation_key);
        assert_eq!(a.content_hash, b.content_hash);
    }
}
