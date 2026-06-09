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
        content:
            "Keep Total Recall episodic data external and promote only bounded working knowledge."
                .to_string(),
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
    let artifact = session_state_artifact("agentic-os".to_string(), "session-1", &[event]).unwrap();
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
