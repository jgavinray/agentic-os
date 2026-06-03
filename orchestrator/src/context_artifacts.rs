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

#[derive(Clone, Debug)]
pub struct RepoMapInput {
    pub repo: String,
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

pub fn repo_map_artifact(input: RepoMapInput) -> ContextArtifact {
    let modules = [
        (
            "orchestrator/src/handlers.rs",
            "HTTP/API handlers, context injection, request persistence",
        ),
        (
            "orchestrator/src/context_compiler.rs",
            "central compiler orchestration and artifact promotion",
        ),
        (
            "orchestrator/src/context_artifacts.rs",
            "typed working-knowledge artifact builders",
        ),
        (
            "orchestrator/src/db.rs",
            "Postgres memory ledger, context queries, artifact storage",
        ),
        ("orchestrator/src/qdrant.rs", "semantic event index access"),
        (
            "orchestrator/src/summarizer.rs",
            "background event summarization",
        ),
        (
            "orchestrator/src/execution_feedback.rs",
            "failure signatures and remediation events",
        ),
        (
            "orchestrator/src/feature_extraction.rs",
            "operational constraints and derived features",
        ),
        (
            "orchestrator/src/litellm.rs",
            "LiteLLM routing metadata and call ledger",
        ),
        ("compose.yaml", "local stack service wiring"),
        ("litellm-config.yaml", "model routing configuration"),
    ];
    let rendered = modules
        .iter()
        .map(|(path, role)| format!("- {path}: {role}\n"))
        .collect::<String>();
    let compact = "Orchestrator modules: handlers, compiler, artifacts, db, qdrant, summarizer, execution feedback, feature extraction, LiteLLM routing.".to_string();
    let raw = serde_json::json!({
        "modules": modules
            .iter()
            .map(|(path, role)| serde_json::json!({"path": path, "role": role}))
            .collect::<Vec<_>>(),
    });
    let source_file_paths = serde_json::json!(modules
        .iter()
        .map(|(path, _)| path.to_string())
        .collect::<Vec<_>>());
    let invalidation_key = stable_hash(&raw);
    ContextArtifact::new(
        input.repo,
        "repo",
        "repo_map",
        Some(raw.to_string()),
        compact,
        rendered,
        invalidation_key,
        source_file_paths,
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

pub fn failure_history_artifact(
    repo: String,
    items: &[crate::db::FailureHistoryItem],
) -> Option<ContextArtifact> {
    let items = items
        .iter()
        .filter(|item| item.remediation.is_some())
        .take(5)
        .collect::<Vec<_>>();
    if items.is_empty() {
        return None;
    }

    let mut rendered = String::new();
    let mut compact_lines = Vec::new();
    let mut source_event_ids = Vec::new();
    for item in &items {
        let remediation = item.remediation.as_ref()?;
        rendered.push_str(&format!(
            "- resolved failure [{}]: {}\n  remediation: {}\n",
            item.category,
            truncate_line(&item.failure.summary, 220),
            truncate_line(&remediation.summary, 220),
        ));
        compact_lines.push(format!(
            "Resolved {} failure with remediation: {}",
            item.category,
            truncate_line(&remediation.summary, 160)
        ));
        source_event_ids.push(item.failure.id.clone());
        source_event_ids.push(remediation.id.clone());
    }

    let raw = serde_json::json!({
        "source": "agent_events",
        "items": items
            .iter()
            .map(|item| serde_json::json!({
                "signature": item.signature,
                "category": item.category,
                "failure": item.failure.payload(),
                "remediation": item.remediation.as_ref().map(crate::db::AgentEvent::payload),
            }))
            .collect::<Vec<_>>(),
    });
    source_event_ids.sort();
    source_event_ids.dedup();
    let source_event_ids_json = serde_json::json!(source_event_ids);
    let invalidation_key = stable_hash(&serde_json::json!({
        "artifact_type": "failure_history",
        "source_event_ids": source_event_ids_json,
        "items": raw["items"],
    }));

    Some(
        ContextArtifact::new(
            repo,
            "repo",
            "failure_history",
            Some(raw.to_string()),
            compact_lines.join(" "),
            rendered,
            invalidation_key,
            serde_json::json!([]),
        )
        .with_source_event_ids(source_event_ids_json),
    )
}

pub fn durable_project_memory_artifact(
    repo: String,
    notes: &[crate::total_recall::MemoryNote],
) -> Option<ContextArtifact> {
    let notes = notes
        .iter()
        .filter(|note| !note.archived && !note.content.trim().is_empty())
        .take(5)
        .collect::<Vec<_>>();
    if notes.is_empty() {
        return None;
    }

    let rendered = notes
        .iter()
        .map(|note| {
            let title = note.title.as_deref().unwrap_or(&note.date);
            format!(
                "- {}: {}\n",
                truncate_line(title, 120),
                truncate_line(&note.content, 260)
            )
        })
        .collect::<String>();
    let compact = notes
        .iter()
        .map(|note| truncate_line(&note.content, 160))
        .collect::<Vec<_>>()
        .join(" ");
    let raw = serde_json::json!({
        "source": "total_recall",
        "notes": notes
            .iter()
            .map(|note| serde_json::json!({
                "id": note.id,
                "date": note.date,
                "title": note.title,
                "content": note.content,
                "updated_at": note.updated_at,
            }))
            .collect::<Vec<_>>(),
    });
    let invalidation_key = stable_hash(&serde_json::json!({
        "artifact_type": "durable_project_memory",
        "notes": raw["notes"],
    }));

    Some(ContextArtifact::new(
        repo,
        "repo",
        "durable_project_memory",
        Some(raw.to_string()),
        compact,
        rendered,
        invalidation_key,
        serde_json::json!([]),
    ))
}

pub fn repo_decisions_artifact(
    repo: String,
    events: &[crate::db::AgentEvent],
) -> Option<ContextArtifact> {
    let events = events.iter().take(6).collect::<Vec<_>>();
    if events.is_empty() {
        return None;
    }
    let rendered = events
        .iter()
        .map(|event| {
            let evidence = event
                .evidence
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .map(|value| format!(" Evidence: {}", truncate_line(value, 160)))
                .unwrap_or_default();
            format!("- {}{}\n", truncate_line(&event.summary, 220), evidence)
        })
        .collect::<String>();
    let raw = serde_json::json!({
        "source": "agent_events",
        "events": events.iter().map(|event| event.payload()).collect::<Vec<_>>(),
    });
    let source_event_ids = serde_json::json!(events
        .iter()
        .map(|event| event.id.clone())
        .collect::<Vec<_>>());
    let invalidation_key = stable_hash(&serde_json::json!({
        "artifact_type": "repo_decisions",
        "source_event_ids": source_event_ids,
        "events": raw["events"],
    }));
    Some(
        ContextArtifact::new(
            repo,
            "repo",
            "repo_decisions",
            Some(raw.to_string()),
            events
                .iter()
                .map(|event| truncate_line(&event.summary, 120))
                .collect::<Vec<_>>()
                .join(" "),
            rendered,
            invalidation_key,
            serde_json::json!([]),
        )
        .with_source_event_ids(source_event_ids),
    )
}

pub fn session_state_artifact(
    repo: String,
    session_id: &str,
    events: &[crate::db::AgentEvent],
) -> Option<ContextArtifact> {
    let events = events.iter().take(5).collect::<Vec<_>>();
    if events.is_empty() {
        return None;
    }
    let rendered = events
        .iter()
        .map(|event| {
            format!(
                "- [{}] {}\n",
                event.event_type,
                truncate_line(&event.summary, 220)
            )
        })
        .collect::<String>();
    let raw = serde_json::json!({
        "source": "agent_events",
        "session_id": session_id,
        "events": events.iter().map(|event| event.payload()).collect::<Vec<_>>(),
    });
    let source_event_ids = serde_json::json!(events
        .iter()
        .map(|event| event.id.clone())
        .collect::<Vec<_>>());
    let invalidation_key = stable_hash(&serde_json::json!({
        "artifact_type": "session_state",
        "session_id": session_id,
        "source_event_ids": source_event_ids,
    }));
    Some(
        ContextArtifact::new(
            repo,
            "session",
            "session_state",
            Some(raw.to_string()),
            events
                .iter()
                .map(|event| truncate_line(&event.summary, 120))
                .collect::<Vec<_>>()
                .join(" "),
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
    let mut subjects = std::collections::HashSet::new();
    for event in events {
        let text = event
            .evidence
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(&event.summary)
            .trim();
        let subject = instruction_subject(text);
        if !subjects.insert(subject) {
            continue;
        }
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

pub fn instruction_subject(text: &str) -> &'static str {
    let lower = text.to_ascii_lowercase();
    if lower.contains("model") || lower.contains("snowflake") || lower.contains("embedding") {
        "model"
    } else if lower.contains("config") || lower.contains("default") || lower.contains("explicit") {
        "configuration"
    } else if lower.contains("tool")
        || lower.contains("python")
        || lower.contains("perl")
        || lower.contains("git apply")
        || lower.contains("apply_patch")
    {
        "tooling"
    } else if lower.contains("total recall") || lower.contains("memory") {
        "memory"
    } else if lower.contains("commit") || lower.contains("push") {
        "git"
    } else {
        "general"
    }
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

fn truncate_line(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim().replace('\n', " ");
    trimmed.chars().take(max_chars).collect()
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
    fn repo_map_artifact_names_core_modules() {
        let artifact = repo_map_artifact(RepoMapInput {
            repo: "agentic-os".to_string(),
        });
        assert_eq!(artifact.artifact_type, "repo_map");
        assert!(artifact.content_rendered.contains("context_compiler.rs"));
        assert!(artifact.content_rendered.contains("db.rs"));
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

    #[test]
    fn active_instruction_artifact_keeps_newest_instruction_per_subject() {
        let old = test_event(
            "old",
            "user_message",
            "Use the old embedding model.",
            serde_json::json!({}),
        );
        let new = test_event(
            "new",
            "user_message",
            "Use the Snowflake embedding model.",
            serde_json::json!({}),
        );
        let artifact = active_instruction_artifact("agentic-os".to_string(), &[new, old]).unwrap();
        assert!(artifact.content_rendered.contains("Snowflake"));
        assert!(!artifact.content_rendered.contains("old embedding"));
    }

    #[test]
    fn failure_history_artifact_promotes_resolved_failures() {
        let failure = test_event(
            "failure-1",
            "validation_result",
            "cargo check failed with type mismatch",
            serde_json::json!({"payload": {"signature": "sig-1", "signature_category": "compile"}}),
        );
        let remediation = test_event(
            "remediation-1",
            "remediation",
            "changed generic bound to match caller",
            serde_json::json!({"payload": {"signature": "sig-1"}}),
        );
        let item = crate::db::FailureHistoryItem {
            signature: "sig-1".to_string(),
            category: "compile".to_string(),
            failure,
            remediation: Some(remediation),
        };

        let artifact = failure_history_artifact("agentic-os".to_string(), &[item]).unwrap();
        assert_eq!(artifact.artifact_type, "failure_history");
        assert!(artifact.content_rendered.contains("resolved failure"));
        assert_eq!(
            artifact.source_event_ids,
            serde_json::json!(["failure-1", "remediation-1"])
        );
    }

    #[test]
    fn durable_project_memory_artifact_promotes_total_recall_notes() {
        let note = crate::total_recall::MemoryNote {
            id: "note-1".to_string(),
            date: "06-02-2026".to_string(),
            title: Some("Compiler Strategy".to_string()),
            content: "Keep Total Recall episodic data external and promote only bounded working knowledge.".to_string(),
            updated_at: 1,
            archived: false,
        };

        let artifact = durable_project_memory_artifact("agentic-os".to_string(), &[note]).unwrap();
        assert_eq!(artifact.artifact_type, "durable_project_memory");
        assert!(artifact.content_rendered.contains("Compiler Strategy"));
    }

    #[test]
    fn repo_decisions_artifact_preserves_event_provenance() {
        let event = test_event(
            "decision-1",
            "decision",
            "Default to the Snowflake embedding model.",
            serde_json::json!({}),
        );
        let artifact = repo_decisions_artifact("agentic-os".to_string(), &[event]).unwrap();
        assert_eq!(artifact.artifact_type, "repo_decisions");
        assert_eq!(artifact.source_event_ids, serde_json::json!(["decision-1"]));
    }

    #[test]
    fn session_state_artifact_is_session_scoped() {
        let event = test_event(
            "event-1",
            "tool_result",
            "cargo check passed",
            serde_json::json!({}),
        );
        let artifact =
            session_state_artifact("agentic-os".to_string(), "session-1", &[event]).unwrap();
        assert_eq!(artifact.scope, "session");
        assert_eq!(artifact.artifact_type, "session_state");
        assert!(artifact.content_rendered.contains("cargo check passed"));
    }

    fn test_event(
        id: &str,
        event_type: &str,
        summary: &str,
        metadata: serde_json::Value,
    ) -> crate::db::AgentEvent {
        crate::db::AgentEvent {
            id: id.to_string(),
            session_id: "session-1".to_string(),
            repo: "agentic-os".to_string(),
            actor: "agent".to_string(),
            event_type: event_type.to_string(),
            summary: summary.to_string(),
            evidence: None,
            metadata,
            correlation_id: None,
            parent_event_id: None,
            trajectory_id: None,
            attempt_index: None,
            event_role: None,
            created_at: chrono::Utc::now(),
            summary_level: 0,
        }
    }
}
