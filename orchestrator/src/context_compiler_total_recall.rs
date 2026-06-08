use crate::context_artifacts::{self, ContextArtifact};
use crate::context_compiler::{CompilerRequest, ContextCompiler};
use crate::context_compiler_support::{record_ledger, total_recall_candidates};

pub(crate) async fn compile_total_recall_artifact(
    compiler: &ContextCompiler<'_>,
    request: &CompilerRequest,
) -> Option<ContextArtifact> {
    let total_recall_url = request.runtime.total_recall_url.as_deref()?;
    match total_recall_candidates(
        compiler.http(),
        total_recall_url,
        &request.repo,
        &request.task,
    )
    .await
    {
        Ok(notes) => {
            let artifact =
                context_artifacts::durable_project_memory_artifact(request.repo.clone(), &notes)?;
            for note in &notes {
                record_ledger(
                    compiler.pool(),
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
            Some(artifact)
        }
        Err(e) => {
            tracing::warn!(
                target: "context_compiler",
                repo = %request.repo,
                total_recall_url = %total_recall_url,
                "failed to fetch Total Recall episodic memory: {e}"
            );
            None
        }
    }
}
