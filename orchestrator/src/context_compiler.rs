use crate::context_artifacts::{self, ContextArtifact};
use crate::db;

#[derive(Clone, Debug)]
pub struct CompilerRequest {
    pub repo: String,
    pub task: String,
    pub runtime: RuntimeContext,
}

#[derive(Clone, Debug)]
pub struct RuntimeContext {
    pub default_model: String,
    pub litellm_url: String,
    pub qdrant_url: String,
    pub summarizer_url: String,
    pub summarizer_model: String,
    pub total_recall_url: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct CompilerOutput {
    pub active_artifacts: Vec<ContextArtifact>,
}

pub struct ContextCompiler<'a> {
    pool: &'a deadpool_postgres::Pool,
    http: &'a reqwest::Client,
}

impl<'a> ContextCompiler<'a> {
    pub fn new(pool: &'a deadpool_postgres::Pool, http: &'a reqwest::Client) -> Self {
        Self { pool, http }
    }

    pub async fn compile(&self, request: CompilerRequest) -> CompilerOutput {
        let mut artifacts = vec![
            compile_service_topology(&request),
            compile_repo_map(&request),
        ];

        for artifact in &artifacts {
            record_ledger(
                self.pool,
                &request.repo,
                artifact,
                "runtime_config",
                None,
                "included",
                "static compiler artifact included for stack and repo awareness",
                serde_json::json!({}),
            )
            .await;
        }

        match db::get_recent_instruction_candidates(self.pool, &request.repo, 25).await {
            Ok(events) => {
                if let Some(artifact) =
                    context_artifacts::active_instruction_artifact(request.repo.clone(), &events)
                {
                    let included_event_ids = artifact_source_event_ids(&artifact);
                    let mut seen_subjects = std::collections::HashSet::new();
                    for event in &events {
                        let text = event
                            .evidence
                            .as_deref()
                            .filter(|value| !value.trim().is_empty())
                            .unwrap_or(&event.summary);
                        let subject = context_artifacts::instruction_subject(text);
                        let is_first_subject = seen_subjects.insert(subject);
                        let included = included_event_ids.contains(&event.id);
                        let (decision, reason) = if included && is_first_subject {
                            (
                                "included",
                                "promoted newest explicit user instruction for this subject",
                            )
                        } else {
                            (
                                "superseded",
                                "newer explicit user instruction for this subject was promoted",
                            )
                        };
                        record_ledger(
                            self.pool,
                            &request.repo,
                            &artifact,
                            "agent_events",
                            Some(&event.id),
                            decision,
                            reason,
                            serde_json::json!({
                                "event_type": event.event_type,
                                "subject": subject,
                            }),
                        )
                        .await;
                    }
                    artifacts.push(artifact);
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "context_compiler",
                    repo = %request.repo,
                    "failed to fetch instruction candidates: {e}"
                );
            }
        }

        match db::get_recent_failure_history(self.pool, &request.repo, 12).await {
            Ok(items) => {
                if let Some(artifact) =
                    context_artifacts::failure_history_artifact(request.repo.clone(), &items)
                {
                    for item in &items {
                        let (decision, reason) = if item.remediation.is_some() {
                            (
                                "included",
                                "promoted resolved failure/remediation pair into failure history artifact",
                            )
                        } else {
                            (
                                "suppressed",
                                "failure has no known remediation and remains raw recent evidence",
                            )
                        };
                        record_ledger(
                            self.pool,
                            &request.repo,
                            &artifact,
                            "agent_events",
                            Some(&item.failure.id),
                            decision,
                            reason,
                            serde_json::json!({
                                "signature": item.signature,
                                "category": item.category,
                                "remediation_event_id": item.remediation.as_ref().map(|event| event.id.clone()),
                            }),
                        )
                        .await;
                    }
                    artifacts.push(artifact);
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "context_compiler",
                    repo = %request.repo,
                    "failed to fetch recent failure history: {e}"
                );
            }
        }

        if let Some(total_recall_url) = request.runtime.total_recall_url.as_deref() {
            match total_recall_candidates(self.http, total_recall_url, &request.repo, &request.task)
                .await
            {
                Ok(notes) => {
                    if let Some(artifact) = context_artifacts::durable_project_memory_artifact(
                        request.repo.clone(),
                        &notes,
                    ) {
                        for note in &notes {
                            record_ledger(
                                self.pool,
                                &request.repo,
                                &artifact,
                                "total_recall",
                                Some(&note.id),
                                "included",
                                "promoted external episodic note into durable project memory artifact",
                                serde_json::json!({
                                    "date": note.date,
                                    "title": note.title,
                                    "updated_at": note.updated_at,
                                }),
                            )
                            .await;
                        }
                        artifacts.push(artifact);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        target: "context_compiler",
                        repo = %request.repo,
                        total_recall_url = %total_recall_url,
                        "failed to fetch Total Recall episodic memory: {e}"
                    );
                }
            }
        }

        for artifact in artifacts {
            if let Err(e) = db::upsert_context_artifact(self.pool, &artifact).await {
                tracing::warn!(
                    target: "context_compiler",
                    repo = %request.repo,
                    artifact_type = %artifact.artifact_type,
                    "failed to upsert context artifact: {e}"
                );
            }
        }

        let active_artifacts = db::get_active_context_artifacts(self.pool, &request.repo, 12)
            .await
            .unwrap_or_else(|e| {
                tracing::warn!(
                    target: "context_compiler",
                    repo = %request.repo,
                    "failed to fetch active context artifacts: {e}"
                );
                vec![]
            });

        CompilerOutput { active_artifacts }
    }
}

fn compile_service_topology(request: &CompilerRequest) -> ContextArtifact {
    context_artifacts::service_topology_artifact(context_artifacts::ServiceTopologyInput {
        repo: request.repo.clone(),
        default_model: request.runtime.default_model.clone(),
        litellm_url: request.runtime.litellm_url.clone(),
        qdrant_url: request.runtime.qdrant_url.clone(),
        summarizer_url: request.runtime.summarizer_url.clone(),
        summarizer_model: request.runtime.summarizer_model.clone(),
    })
}

fn compile_repo_map(request: &CompilerRequest) -> ContextArtifact {
    context_artifacts::repo_map_artifact(context_artifacts::RepoMapInput {
        repo: request.repo.clone(),
    })
}

async fn total_recall_candidates(
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

fn artifact_source_event_ids(artifact: &ContextArtifact) -> std::collections::HashSet<String> {
    artifact
        .source_event_ids
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_string)
        .collect()
}

async fn record_ledger(
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

pub fn render(output: &CompilerOutput) -> String {
    context_artifacts::render_artifacts(&output.active_artifacts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_topology_compilation_is_deterministic() {
        let request = CompilerRequest {
            repo: "agentic-os".to_string(),
            task: "debug compiler".to_string(),
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
