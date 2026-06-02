use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

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

#[derive(Clone, Debug)]
pub struct ServiceTopologyInput {
    pub repo: String,
    pub default_model: String,
    pub litellm_url: String,
    pub qdrant_url: String,
    pub summarizer_url: String,
    pub summarizer_model: String,
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

pub fn service_topology_artifact(input: ServiceTopologyInput) -> ContextArtifact {
    let rendered = format!(
        "\
- orchestrator: injects context and routes model traffic through LiteLLM
- default model: {default_model}
- LiteLLM endpoint: {litellm_url}
- Qdrant endpoint: {qdrant_url}
- summarizer endpoint: {summarizer_url}
- summarizer model: {summarizer_model}
",
        default_model = input.default_model,
        litellm_url = input.litellm_url,
        qdrant_url = input.qdrant_url,
        summarizer_url = input.summarizer_url,
        summarizer_model = input.summarizer_model,
    );
    let compact = format!(
        "Default model {model}; LiteLLM {litellm}; Qdrant {qdrant}; summarizer {summarizer}.",
        model = input.default_model,
        litellm = input.litellm_url,
        qdrant = input.qdrant_url,
        summarizer = input.summarizer_url,
    );
    let raw = serde_json::json!({
        "default_model": input.default_model,
        "litellm_url": input.litellm_url,
        "qdrant_url": input.qdrant_url,
        "summarizer_url": input.summarizer_url,
        "summarizer_model": input.summarizer_model,
    });
    let invalidation_key = stable_hash(&raw);
    ContextArtifact::new(
        input.repo,
        "repo",
        "service_topology",
        Some(raw.to_string()),
        compact,
        rendered,
        invalidation_key,
        serde_json::json!(["compose.yaml", "litellm-config.yaml"]),
    )
}

pub fn active_instruction_artifact(
    repo: String,
    events: &[crate::db::AgentEvent],
) -> Option<ContextArtifact> {
    let instructions = active_instruction_lines(events);
    if instructions.is_empty() {
        return None;
    }

    let rendered = instructions
        .iter()
        .map(|line| format!("- {line}\n"))
        .collect::<String>();
    let compact = instructions.join(" ");
    let raw = serde_json::json!({
        "source": "agent_events",
        "instructions": instructions,
        "events": events.iter().map(crate::db::AgentEvent::payload).collect::<Vec<_>>(),
    });
    let source_event_ids = serde_json::json!(events
        .iter()
        .map(|event| event.id.clone())
        .collect::<Vec<_>>());
    let invalidation_key = stable_hash(&serde_json::json!({
        "artifact_type": "active_instruction",
        "source_event_ids": source_event_ids,
        "instructions": raw["instructions"],
    }));

    Some(
        ContextArtifact::new(
            repo,
            "repo",
            "active_instruction",
            Some(raw.to_string()),
            compact,
            rendered,
            invalidation_key,
            serde_json::json!([]),
        )
        .with_source_event_ids(source_event_ids),
    )
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

fn active_instruction_lines(events: &[crate::db::AgentEvent]) -> Vec<String> {
    let mut lines = Vec::new();
    for event in events.iter().rev() {
        let text = event
            .evidence
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(&event.summary)
            .trim();
        let normalized = normalize_instruction(text);
        if normalized.is_empty() || lines.iter().any(|line| line == &normalized) {
            continue;
        }
        lines.push(normalized);
        if lines.len() >= 5 {
            break;
        }
    }
    lines
}

fn normalize_instruction(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(3)
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(320)
        .collect()
}

fn stable_hash(value: &serde_json::Value) -> String {
    let bytes = serde_json::to_vec(value).expect("context artifact hash serialization cannot fail");
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_topology_hash_is_stable_for_same_runtime_inputs() {
        let input = ServiceTopologyInput {
            repo: "agentic-os".to_string(),
            default_model: "qwen36-35b-a3b".to_string(),
            litellm_url: "http://litellm:4000/v1".to_string(),
            qdrant_url: "http://qdrant:6333".to_string(),
            summarizer_url: "http://summarizer:8080/v1".to_string(),
            summarizer_model: "qwen2.5-3b".to_string(),
        };
        let a = service_topology_artifact(input.clone());
        let b = service_topology_artifact(input);
        assert_eq!(a.invalidation_key, b.invalidation_key);
        assert_eq!(a.content_hash, b.content_hash);
    }

    #[test]
    fn active_instruction_artifact_is_bounded_and_provenanced() {
        let mut event = crate::db::AgentEvent {
            id: "event-1".to_string(),
            session_id: "session-1".to_string(),
            repo: "agentic-os".to_string(),
            actor: "user".to_string(),
            event_type: "user_message".to_string(),
            summary: "Be explicit about configuration defaults.".to_string(),
            evidence: None,
            metadata: serde_json::json!({}),
            correlation_id: None,
            parent_event_id: None,
            trajectory_id: None,
            attempt_index: None,
            event_role: Some("user".to_string()),
            created_at: chrono::Utc::now(),
            summary_level: 0,
        };
        event.evidence = Some("The config should default to the Snowflake model.".to_string());

        let artifact = active_instruction_artifact("agentic-os".to_string(), &[event]).unwrap();
        assert_eq!(artifact.artifact_type, "active_instruction");
        assert_eq!(artifact.source_event_ids, serde_json::json!(["event-1"]));
        assert!(artifact.content_rendered.contains("Snowflake"));
    }
}
