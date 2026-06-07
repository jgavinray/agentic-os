use crate::context_artifacts::{self, ContextArtifact};
use crate::context_compiler::CompilerRequest;
use crate::context_compiler_support::record_ledger;
use crate::db;

pub(crate) async fn compile_failure_history_artifact(
    pool: &deadpool_postgres::Pool,
    request: &CompilerRequest,
) -> Option<ContextArtifact> {
    match db::get_recent_failure_history(pool, &request.repo, 12).await {
        Ok(items) => {
            let artifact =
                context_artifacts::failure_history_artifact(request.repo.clone(), &items)?;
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
                    pool,
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
            Some(artifact)
        }
        Err(e) => {
            tracing::warn!(
                target: "context_compiler",
                repo = %request.repo,
                "failed to fetch recent failure history: {e}"
            );
            None
        }
    }
}

pub(crate) async fn compile_repo_decisions_artifact(
    pool: &deadpool_postgres::Pool,
    request: &CompilerRequest,
) -> Option<ContextArtifact> {
    match db::get_recent_repo_decision_candidates(pool, &request.repo, 12).await {
        Ok(events) => {
            let artifact =
                context_artifacts::repo_decisions_artifact(request.repo.clone(), &events)?;
            for event in &events {
                record_ledger(
                    pool,
                    &request.repo,
                    &artifact,
                    "agent_events",
                    Some(&event.id),
                    "included",
                    "promoted repository decision candidate into repo_decisions artifact",
                    serde_json::json!({"event_type": event.event_type}),
                )
                .await;
            }
            Some(artifact)
        }
        Err(e) => {
            tracing::warn!(
                target: "context_compiler",
                repo = %request.repo,
                "failed to fetch repo decision candidates: {e}"
            );
            None
        }
    }
}

pub(crate) async fn compile_session_state_artifact(
    pool: &deadpool_postgres::Pool,
    request: &CompilerRequest,
    session_id: &str,
) -> Option<ContextArtifact> {
    match db::get_recent_session_events(pool, session_id, 8).await {
        Ok(events) => {
            let artifact = context_artifacts::session_state_artifact(
                request.repo.clone(),
                session_id,
                &events,
            )?;
            for event in &events {
                record_ledger(
                    pool,
                    &request.repo,
                    &artifact,
                    "agent_events",
                    Some(&event.id),
                    "included",
                    "promoted active session event into session_state artifact",
                    serde_json::json!({
                        "event_type": event.event_type,
                        "session_id": session_id,
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
                session_id = %session_id,
                "failed to fetch session state candidates: {e}"
            );
            None
        }
    }
}
