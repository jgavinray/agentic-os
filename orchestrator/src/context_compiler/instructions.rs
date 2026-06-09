use crate::context_artifacts::{self, ContextArtifact};
use crate::context_compiler::support::{artifact_source_event_ids, record_ledger};
use crate::context_compiler::CompilerRequest;
use crate::db;

pub(crate) async fn compile_active_instruction_artifact(
    pool: &deadpool_postgres::Pool,
    request: &CompilerRequest,
) -> Option<ContextArtifact> {
    match db::get_recent_instruction_candidates(pool, &request.repo, 25).await {
        Ok(events) => {
            let artifact =
                context_artifacts::active_instruction_artifact(request.repo.clone(), &events)?;
            record_instruction_candidate_decisions(pool, request, &artifact, &events).await;
            Some(artifact)
        }
        Err(e) => {
            tracing::warn!(
                target: "context_compiler",
                repo = %request.repo,
                "failed to fetch instruction candidates: {e}"
            );
            None
        }
    }
}

async fn record_instruction_candidate_decisions(
    pool: &deadpool_postgres::Pool,
    request: &CompilerRequest,
    artifact: &ContextArtifact,
    events: &[crate::db::AgentEvent],
) {
    let included_event_ids = artifact_source_event_ids(artifact);
    let mut seen_subjects = std::collections::HashSet::new();
    for event in events {
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
            pool,
            &request.repo,
            artifact,
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
}
