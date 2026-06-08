use crate::context_artifacts::{self, ContextArtifact};
use crate::context_compiler_event_artifacts::{
    compile_failure_history_artifact, compile_repo_decisions_artifact,
    compile_session_state_artifact,
};
use crate::context_compiler_instructions::compile_active_instruction_artifact;
use crate::context_compiler_support::{
    compile_repo_map, compile_service_topology, compiler_source_allowed, record_ledger,
};
use crate::context_compiler_total_recall::compile_total_recall_artifact;
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

    pub(crate) fn pool(&self) -> &'a deadpool_postgres::Pool {
        self.pool
    }

    pub(crate) fn http(&self) -> &'a reqwest::Client {
        self.http
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
            if let Some(artifact) = compile_total_recall_artifact(self, &request).await {
                artifacts.push(artifact);
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
