use crate::context_artifacts::{self, ContextArtifact};
use crate::context_compiler_event_artifacts::{
    compile_failure_history_artifact, compile_repo_decisions_artifact,
    compile_session_state_artifact,
};
use crate::context_compiler_instructions::compile_active_instruction_artifact;
use crate::context_compiler_support::{
    compile_repo_map, compile_service_topology, compiler_source_allowed, record_ledger,
    total_recall_candidates,
};
use crate::db;

#[derive(Clone, Debug)]
pub struct CompilerRequest {
    pub repo: String,
    pub task: String,
    pub session_id: Option<String>,
    pub policy: Option<crate::orchestration_policy::OrchestrationPolicy>,
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
        let mut artifacts = Vec::new();

        if compiler_source_allowed(
            &request,
            crate::orchestration_policy::ContextSource::CompiledSummaries,
        ) {
            artifacts.push(compile_service_topology(&request));
            artifacts.push(compile_repo_map(&request));

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
        }

        if compiler_source_allowed(
            &request,
            crate::orchestration_policy::ContextSource::PostgresEvents,
        ) && compiler_source_allowed(
            &request,
            crate::orchestration_policy::ContextSource::CompiledSummaries,
        ) {
            if let Some(artifact) = compile_active_instruction_artifact(self.pool, &request).await {
                artifacts.push(artifact);
            }
        }

        if compiler_source_allowed(
            &request,
            crate::orchestration_policy::ContextSource::ContextLedger,
        ) {
            if let Some(artifact) = compile_failure_history_artifact(self.pool, &request).await {
                artifacts.push(artifact);
            }
        }

        if compiler_source_allowed(
            &request,
            crate::orchestration_policy::ContextSource::CompiledSummaries,
        ) {
            if let Some(artifact) = compile_repo_decisions_artifact(self.pool, &request).await {
                artifacts.push(artifact);
            }
        }

        if compiler_source_allowed(
            &request,
            crate::orchestration_policy::ContextSource::CompiledSummaries,
        ) {
            if let Some(session_id) = request.session_id.as_deref() {
                if let Some(artifact) =
                    compile_session_state_artifact(self.pool, &request, session_id).await
                {
                    artifacts.push(artifact);
                }
            }
        }

        if compiler_source_allowed(
            &request,
            crate::orchestration_policy::ContextSource::TotalRecall,
        ) {
            if let Some(total_recall_url) = request.runtime.total_recall_url.as_deref() {
                match total_recall_candidates(
                    self.http,
                    total_recall_url,
                    &request.repo,
                    &request.task,
                )
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

pub fn render(output: &CompilerOutput) -> String {
    context_artifacts::render_artifacts(&output.active_artifacts)
}
