use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub use crate::context_artifacts::builders::{
    active_instruction_artifact, durable_project_memory_artifact, instruction_subject,
    repo_decisions_artifact, session_state_artifact,
};
pub use crate::context_artifacts::failure_history_builder::failure_history_artifact;
pub use crate::context_artifacts::static_builders::{
    repo_map_artifact, service_topology_artifact, RepoMapInput, ServiceTopologyInput,
};

mod builders;
mod failure_history_builder;
mod static_builders;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextArtifact {
    pub id: Uuid,
    pub repo: String,
    pub scope: String,
    pub artifact_type: String,
    pub status: String,
    pub content_raw: Option<String>,
    pub content_compact: String,
    pub content_rendered: String,
    pub content_hash: String,
    pub invalidation_key: String,
    pub source_event_ids: serde_json::Value,
    pub source_file_paths: serde_json::Value,
    pub token_estimate: i32,
}

impl ContextArtifact {
    pub fn new(
        repo: String,
        scope: &str,
        artifact_type: &str,
        content_raw: Option<String>,
        content_compact: String,
        content_rendered: String,
        invalidation_key: String,
        source_file_paths: serde_json::Value,
    ) -> Self {
        let content_hash = stable_hash(&serde_json::json!({
            "scope": scope,
            "artifact_type": artifact_type,
            "content_rendered": content_rendered,
            "invalidation_key": invalidation_key,
        }));
        let token_estimate = crate::db::estimate_tokens(&content_rendered) as i32;
        Self {
            id: Uuid::new_v4(),
            repo,
            scope: scope.to_string(),
            artifact_type: artifact_type.to_string(),
            status: "active".to_string(),
            content_raw,
            content_compact,
            content_rendered,
            content_hash,
            invalidation_key,
            source_event_ids: serde_json::json!([]),
            source_file_paths,
            token_estimate,
        }
    }

    pub fn with_source_event_ids(mut self, source_event_ids: serde_json::Value) -> Self {
        self.source_event_ids = source_event_ids;
        self
    }
}

pub fn render_artifacts(artifacts: &[ContextArtifact]) -> String {
    if artifacts.is_empty() {
        return String::new();
    }
    let mut out = String::from("== Stable Context Artifacts ==\n");
    for artifact in artifacts {
        out.push_str(&format!(
            "[{}:{}:{}]\n{}\n",
            artifact.scope, artifact.artifact_type, artifact.status, artifact.content_rendered
        ));
    }
    out.push('\n');
    out
}

pub(crate) fn stable_hash(value: &serde_json::Value) -> String {
    let bytes = serde_json::to_vec(value).expect("context artifact hash serialization cannot fail");
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

#[cfg(test)]
#[path = "context_artifacts_tests.rs"]
mod tests;
