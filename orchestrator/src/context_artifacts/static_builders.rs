use crate::context_artifacts::{stable_hash, ContextArtifact};

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
