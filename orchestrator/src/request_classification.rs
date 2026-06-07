//! Deterministic request classification taxonomy.
//!
//! This module defines the bounded labels and table-facing structs for the
//! pre-LLM request classification layer. Feature extraction, backfill, and live
//! routing are later phases.

use chrono::Utc;
use serde_json::Value;

use crate::request_classification_composition::{
    decomposition_fragments, has_subtask_action_signal,
};
use crate::request_classification_feature_json::features_to_json;
use crate::request_classification_features::{extract_features, RequestFeatures};
use crate::request_classification_input::{event_text, has_request_text, metadata_key_text};
pub use crate::request_classification_labels::{
    bounded_complexity, bounded_domain, bounded_intent, bounded_live_policy_action,
    bounded_live_policy_bypass, bounded_live_policy_reason, bounded_risk, bounded_route,
};
pub use crate::request_classification_runtime::{
    evaluate_live_policy, record_classification_metrics,
};
pub use crate::request_classification_store::{
    classify_and_persist_event, persist_classification, request_classification_report,
    run_backfill, update_classification_if_changed,
};
pub use crate::request_classification_types::{
    enum_inventory, live_policy_config_from_env, request_classification_startup_backfill_enabled,
    request_classification_startup_batch_size, BackfillOptions, BackfillReport, LabelCount,
    LivePolicyConfig, LivePolicyDecision, PersistOutcome, RecommendedRoute, ReportOptions,
    RequestArtifactType, RequestClassification, RequestClassificationReport, RequestComplexity,
    RequestDomain, RequestIntent, RequestRisk, ResponseContract, SessionRouteCount,
    CLASSIFICATION_SCHEMA_VERSION, CLASSIFIER_SOURCE_DETERMINISTIC_RULES, FEATURE_KEYS,
    ROUTING_POLICY_VERSION,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompositeAnalysis {
    pub(crate) is_composite: bool,
    pub(crate) decomposition_candidate: bool,
    pub(crate) reason: &'static str,
    pub(crate) sub_intents: Vec<RequestIntent>,
}

/// Classify a loaded event into deterministic request-level features and labels.
///
/// This is a pure Phase 2 entry point: it does not query storage, call models,
/// touch the filesystem, emit metrics, or change live routing behavior.
pub fn classify_request_event(event: &crate::db::AgentEvent) -> RequestClassification {
    let text = event_text(event);
    let lower = text.to_ascii_lowercase();
    let metadata_keys = metadata_key_text(&event.metadata);
    let metadata_keys_lower = metadata_keys.to_ascii_lowercase();
    let features = extract_features(&text, &lower, &metadata_keys_lower);
    let detected_domains = detected_domains(&features, &lower);
    let composite = analyze_composition(&text, &lower, &event.event_type);

    let mut row = RequestClassification::deterministic(
        event.id.clone(),
        event.repo.clone(),
        event.session_id.clone(),
        event.created_at,
    );
    row.trajectory_id = event.trajectory_id;
    row.features = features_to_json(&features, &detected_domains, &composite);
    row.intent = classify_intent(&features, &lower, &event.event_type);
    row.domain = classify_domain(&features, &lower, &detected_domains);
    row.secondary_domains = detected_domains
        .iter()
        .copied()
        .filter(|domain| *domain != row.domain)
        .collect();
    row.artifact_type = classify_artifact(&features, &lower);
    row.risk = classify_risk(&features, &lower, row.domain);
    row.complexity = classify_complexity(&features, row.intent, &row.risk, detected_domains.len());
    row.recommended_route = recommend_route(row.intent, row.complexity, &row.risk, &features);
    row.response_contract = response_contract(row.intent, row.artifact_type, row.recommended_route);
    row
}

pub fn classify_request_text(
    repo: &str,
    session_id: &str,
    summary: &str,
    evidence: Option<&str>,
    event_type: &str,
) -> RequestClassification {
    let event = crate::db::AgentEvent {
        id: "live-request".to_string(),
        session_id: session_id.to_string(),
        repo: repo.to_string(),
        actor: "user".to_string(),
        event_type: event_type.to_string(),
        summary: summary.to_string(),
        evidence: evidence.map(str::to_string),
        metadata: Value::Object(Default::default()),
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: Some("request".to_string()),
        created_at: Utc::now(),
        summary_level: 0,
    };
    classify_request_event(&event)
}

pub fn is_classifiable_request_event(event: &crate::db::AgentEvent) -> bool {
    if !has_request_text(event) {
        return false;
    }

    event.event_type == "user_message"
        || event.event_role.as_deref() == Some("request")
        || (event.event_type == "context_pack"
            && event.event_role.as_deref() == Some("context_pack")
            && event.metadata.get("request").is_some())
}

fn detected_domains(features: &RequestFeatures, lower: &str) -> Vec<RequestDomain> {
    let mut domains = Vec::new();
    push_if(
        &mut domains,
        features.has_kubernetes_terms,
        RequestDomain::Kubernetes,
    );
    push_if(
        &mut domains,
        features.has_docker_terms,
        RequestDomain::Docker,
    );
    push_if(
        &mut domains,
        features.has_llm_terms,
        RequestDomain::LlmInference,
    );
    push_if(
        &mut domains,
        features.has_networking_terms,
        RequestDomain::Networking,
    );
    push_if(
        &mut domains,
        features.has_security_terms,
        RequestDomain::Security,
    );
    push_if(
        &mut domains,
        contains_any(lower, &["medical", "doctor", "diagnosis", "medicine"]),
        RequestDomain::Medical,
    );
    push_if(
        &mut domains,
        contains_any(lower, &["legal", "lawyer", "lawsuit", "contract"]),
        RequestDomain::Legal,
    );
    push_if(
        &mut domains,
        contains_any(lower, &["finance", "tax", "investment", "stock", "loan"]),
        RequestDomain::Finance,
    );
    push_if(
        &mut domains,
        features.has_shell_command,
        RequestDomain::Shell,
    );
    domains
}

fn classify_intent(features: &RequestFeatures, lower: &str, event_type: &str) -> RequestIntent {
    if features.char_count == 0 {
        return RequestIntent::Unknown;
    }
    if contains_any(lower, &["summarize", "summary", "recap"]) {
        RequestIntent::Summarize
    } else if contains_any(lower, &["classify", "categorize", "label this"]) {
        RequestIntent::Classify
    } else if contains_any(lower, &["search", "look up", "find current", "latest"]) {
        RequestIntent::Search
    } else if contains_any(lower, &["plan", "proposal", "approach", "design"]) {
        RequestIntent::Plan
    } else if contains_any(
        lower,
        &[
            "implement",
            "implementation",
            "build this",
            "build the",
            "add feature",
            "add support",
            "add functionality",
            "wire up",
            "integrate",
        ],
    ) {
        RequestIntent::Implement
    } else if contains_any(
        lower,
        &["generate config", "create yaml", "write yaml", "manifest"],
    ) {
        RequestIntent::GenerateConfig
    } else if features.has_config_shape
        && contains_any(lower, &["edit", "modify", "change", "fix", "update"])
    {
        RequestIntent::ModifyConfig
    } else if contains_any(lower, &["run ", "execute ", "deploy", "restart"])
        || event_type == "tool_call"
    {
        RequestIntent::OperateTool
    } else if features.contains_error_words || features.has_stack_trace || features.has_test_failure
    {
        RequestIntent::Debug
    } else {
        RequestIntent::Explain
    }
}

fn classify_domain(
    features: &RequestFeatures,
    lower: &str,
    detected_domains: &[RequestDomain],
) -> RequestDomain {
    for high_stakes in [
        RequestDomain::Medical,
        RequestDomain::Legal,
        RequestDomain::Finance,
    ] {
        if detected_domains.contains(&high_stakes) {
            return high_stakes;
        }
    }
    if features.char_count == 0 {
        RequestDomain::Unknown
    } else if let Some(domain) = detected_domains.first() {
        *domain
    } else if contains_any(lower, &["shell", "terminal", "bash", "zsh"]) {
        RequestDomain::Shell
    } else {
        RequestDomain::Generic
    }
}

fn classify_artifact(features: &RequestFeatures, lower: &str) -> RequestArtifactType {
    if features.char_count == 0 {
        RequestArtifactType::Unknown
    } else if features.has_logs || features.has_stack_trace || features.has_test_failure {
        RequestArtifactType::Logs
    } else if features.has_json {
        RequestArtifactType::Json
    } else if features.has_yaml {
        RequestArtifactType::Yaml
    } else if contains_any(
        lower,
        &["sql", "select ", "insert into", "update ", "delete from"],
    ) {
        RequestArtifactType::Sql
    } else if contains_any(lower, &["markdown", ".md", "# "]) {
        RequestArtifactType::Markdown
    } else if features.has_code_block || features.has_diff_or_patch {
        RequestArtifactType::Code
    } else if contains_any(lower, &["image", "screenshot", ".png", ".jpg", ".jpeg"]) {
        RequestArtifactType::Image
    } else if features.has_file_path || features.asks_for_file_generation {
        RequestArtifactType::File
    } else {
        RequestArtifactType::PlainText
    }
}

fn classify_risk(
    features: &RequestFeatures,
    lower: &str,
    domain: RequestDomain,
) -> Vec<RequestRisk> {
    if features.char_count == 0 {
        return vec![RequestRisk::Unknown];
    }

    let mut risks = Vec::new();
    push_if(
        &mut risks,
        features.has_secret_candidate,
        RequestRisk::SecretPresent,
    );
    push_if(
        &mut risks,
        features.contains_destructive_verbs,
        RequestRisk::DestructiveCommand,
    );
    push_if(
        &mut risks,
        features.asks_for_latest,
        RequestRisk::ExternalCurrentInfoRequired,
    );
    push_if(
        &mut risks,
        matches!(
            domain,
            RequestDomain::Medical | RequestDomain::Legal | RequestDomain::Finance
        ),
        RequestRisk::HighStakes,
    );
    push_if(
        &mut risks,
        contains_any(
            lower,
            &[
                "ignore previous",
                "ignore all previous",
                "system prompt",
                "developer message",
                "jailbreak",
            ],
        ),
        RequestRisk::PromptInjection,
    );
    push_if(
        &mut risks,
        contains_any(
            lower,
            &[
                "steal credentials",
                "credential dump",
                "phishing",
                "malware",
                "exploit this",
                "bypass auth",
            ],
        ),
        RequestRisk::UnsafeSecurity,
    );

    if risks.is_empty() {
        risks.push(RequestRisk::None);
    }
    risks
}

fn classify_complexity(
    features: &RequestFeatures,
    intent: RequestIntent,
    risks: &[RequestRisk],
    domain_count: usize,
) -> RequestComplexity {
    if features.char_count == 0 {
        RequestComplexity::Unknown
    } else if risks.iter().any(|risk| {
        matches!(
            risk,
            RequestRisk::HighStakes
                | RequestRisk::UnsafeSecurity
                | RequestRisk::DestructiveCommand
                | RequestRisk::SecretPresent
        )
    }) {
        RequestComplexity::L5HighRisk
    } else if matches!(intent, RequestIntent::OperateTool)
        || features.has_shell_command
        || features.asks_for_file_generation
    {
        RequestComplexity::L4ToolRequired
    } else if features.char_count > 2_000
        || features.line_count > 60
        || features.has_stack_trace
        || features.has_diff_or_patch
        || features.has_test_failure
    {
        RequestComplexity::L3Complex
    } else if domain_count > 1
        || matches!(
            intent,
            RequestIntent::Debug
                | RequestIntent::Implement
                | RequestIntent::GenerateConfig
                | RequestIntent::ModifyConfig
        )
        || features.has_config_shape
    {
        RequestComplexity::L2Moderate
    } else if features.char_count <= 40 && !features.contains_error_words {
        RequestComplexity::L0Trivial
    } else {
        RequestComplexity::L1Simple
    }
}

fn recommend_route(
    intent: RequestIntent,
    complexity: RequestComplexity,
    risks: &[RequestRisk],
    features: &RequestFeatures,
) -> RecommendedRoute {
    if risks.iter().any(|risk| {
        matches!(
            risk,
            RequestRisk::UnsafeSecurity
                | RequestRisk::HighStakes
                | RequestRisk::DestructiveCommand
                | RequestRisk::SecretPresent
        )
    }) {
        RecommendedRoute::RefuseOrGuardrail
    } else if risks.contains(&RequestRisk::ExternalCurrentInfoRequired) {
        RecommendedRoute::WebRequired
    } else if matches!(
        intent,
        RequestIntent::OperateTool | RequestIntent::Implement | RequestIntent::ModifyConfig
    ) && !features.has_file_path
        && !features.has_config_shape
        && !features.has_shell_command
    {
        RecommendedRoute::AskClarification
    } else if matches!(intent, RequestIntent::OperateTool) || features.has_shell_command {
        RecommendedRoute::ToolRequired
    } else if matches!(complexity, RequestComplexity::L0Trivial) {
        RecommendedRoute::DeterministicTemplate
    } else if matches!(
        (intent, complexity),
        (
            RequestIntent::Explain | RequestIntent::Summarize | RequestIntent::Classify,
            RequestComplexity::L1Simple | RequestComplexity::L2Moderate
        )
    ) {
        RecommendedRoute::SmallLocalModel
    } else if matches!(complexity, RequestComplexity::L3Complex) {
        RecommendedRoute::StrongLocalModel
    } else if matches!(complexity, RequestComplexity::Unknown) {
        RecommendedRoute::Unknown
    } else {
        RecommendedRoute::SmallLocalModel
    }
}

fn response_contract(
    intent: RequestIntent,
    artifact_type: RequestArtifactType,
    route: RecommendedRoute,
) -> ResponseContract {
    match route {
        RecommendedRoute::RefuseOrGuardrail => ResponseContract::Refusal,
        RecommendedRoute::AskClarification => ResponseContract::ClarificationQuestion,
        RecommendedRoute::ToolRequired => ResponseContract::ValidationRequired,
        _ if matches!(intent, RequestIntent::Classify) => ResponseContract::StructuredJson,
        _ if matches!(intent, RequestIntent::Summarize | RequestIntent::Plan) => {
            ResponseContract::MarkdownSummary
        }
        _ if matches!(intent, RequestIntent::Implement) => ResponseContract::ValidationRequired,
        _ if matches!(
            artifact_type,
            RequestArtifactType::Code | RequestArtifactType::Yaml | RequestArtifactType::Json
        ) =>
        {
            ResponseContract::ValidationRequired
        }
        _ if matches!(route, RecommendedRoute::Unknown) => ResponseContract::Unknown,
        _ => ResponseContract::DirectAnswer,
    }
}

fn analyze_composition(text: &str, lower: &str, event_type: &str) -> CompositeAnalysis {
    if text.trim().is_empty() {
        return CompositeAnalysis {
            is_composite: false,
            decomposition_candidate: false,
            reason: "none",
            sub_intents: Vec::new(),
        };
    }

    let (fragments, reason) = decomposition_fragments(text, lower);
    let mut sub_intents = Vec::new();
    for fragment in fragments.iter().take(5) {
        let fragment = fragment.trim();
        if fragment.len() < 3 {
            continue;
        }
        let fragment_lower = fragment.to_ascii_lowercase();
        if !has_subtask_action_signal(&fragment_lower) {
            continue;
        }
        let features = extract_features(fragment, &fragment_lower, "");
        sub_intents.push(classify_intent(&features, &fragment_lower, event_type));
    }

    let decomposition_candidate = sub_intents.len() >= 2;
    CompositeAnalysis {
        is_composite: decomposition_candidate,
        decomposition_candidate,
        reason: if decomposition_candidate {
            reason
        } else {
            "none"
        },
        sub_intents: if decomposition_candidate {
            sub_intents
        } else {
            Vec::new()
        },
    }
}

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

fn push_if<T: PartialEq + Copy>(items: &mut Vec<T>, condition: bool, item: T) {
    if condition && !items.contains(&item) {
        items.push(item);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::AgentEvent;
    use serde_json::json;
    use std::collections::HashSet;

    const REQUEST_CLASSIFICATION_MIGRATIONS: &str = concat!(
        include_str!("../migrations/V9__request_classifications.sql"),
        "\n",
        include_str!("../migrations/V17__add_implement_request_intent.sql")
    );

    #[test]
    fn enum_inventory_contains_unknown_for_every_enum() {
        for (name, variants) in enum_inventory() {
            assert!(
                variants.contains(&"unknown"),
                "{name} must include unknown variant"
            );
        }
    }

    #[test]
    fn unknown_labels_map_to_unknown() {
        assert_eq!(
            RequestIntent::from_label("not-a-real-intent"),
            RequestIntent::Unknown
        );
        assert_eq!(
            RequestDomain::from_label("not-a-real-domain"),
            RequestDomain::Unknown
        );
        assert_eq!(
            RequestArtifactType::from_label("not-a-real-artifact"),
            RequestArtifactType::Unknown
        );
        assert_eq!(
            RequestComplexity::from_label("not-a-real-complexity"),
            RequestComplexity::Unknown
        );
        assert_eq!(
            RequestRisk::from_label("not-a-real-risk"),
            RequestRisk::Unknown
        );
        assert_eq!(
            RecommendedRoute::from_label("not-a-real-route"),
            RecommendedRoute::Unknown
        );
        assert_eq!(
            ResponseContract::from_label("not-a-real-contract"),
            ResponseContract::Unknown
        );
    }

    #[test]
    fn migration_has_checks_for_all_closed_labels() {
        for (_name, variants) in enum_inventory() {
            for variant in *variants {
                assert!(
                    REQUEST_CLASSIFICATION_MIGRATIONS.contains(&format!("'{variant}'")),
                    "migration is missing enum label {variant}"
                );
            }
        }
        assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (intent IN"));
        assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (domain IN"));
        assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (artifact_type IN"));
        assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (complexity IN"));
        assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (recommended_route IN"));
        assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (response_contract IN"));
        assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (risk <@ ARRAY"));
        assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (secondary_domains <@ ARRAY"));
    }

    #[test]
    fn schema_privacy_allows_only_documented_unbounded_text_columns() {
        let allowed_unbounded_strings: HashSet<&str> = [
            "event_id",
            "repo",
            "session_id",
            "routing_policy_version",
            "classifier_source",
        ]
        .into_iter()
        .collect();
        let text_columns = [
            "event_id",
            "repo",
            "session_id",
            "routing_policy_version",
            "classifier_source",
        ];

        for column in text_columns {
            assert!(
                allowed_unbounded_strings.contains(column),
                "{column} must be explicitly privacy-allowlisted"
            );
        }
        assert!(!REQUEST_CLASSIFICATION_MIGRATIONS.contains("summary TEXT"));
        assert!(!REQUEST_CLASSIFICATION_MIGRATIONS.contains("evidence TEXT"));
        assert!(!REQUEST_CLASSIFICATION_MIGRATIONS.contains("request TEXT"));
        assert!(!REQUEST_CLASSIFICATION_MIGRATIONS.contains("prompt TEXT"));
    }

    #[test]
    fn deterministic_default_row_uses_current_versions() {
        let now = Utc::now();
        let row = RequestClassification::deterministic(
            "event-1".to_string(),
            "repo".to_string(),
            "session".to_string(),
            now,
        );

        assert_eq!(
            row.classification_schema_version,
            CLASSIFICATION_SCHEMA_VERSION
        );
        assert_eq!(row.routing_policy_version, ROUTING_POLICY_VERSION);
        assert_eq!(row.classifier_source, CLASSIFIER_SOURCE_DETERMINISTIC_RULES);
        assert_eq!(row.intent, RequestIntent::Unknown);
        assert_eq!(row.recommended_route, RecommendedRoute::Unknown);
    }

    #[test]
    fn feature_extraction_is_deterministic_for_identical_input() {
        let event = event(
            "e-1",
            "Please explain the Docker compose error",
            Some("ERROR failed to connect to http://localhost:8088"),
        );

        let first = classify_request_event(&event);
        let second = classify_request_event(&event);

        assert_eq!(first, second);
        assert_eq!(first.domain, RequestDomain::Docker);
        assert_eq!(first.intent, RequestIntent::Debug);
    }

    #[test]
    fn feature_keys_are_closed_and_stable() {
        let row = classify_request_event(&event(
            "e-keys",
            "Summarize this Kubernetes log",
            Some("[ERROR] pod failed"),
        ));
        let object = row.features.as_object().expect("features must be object");
        let actual: HashSet<&str> = object.keys().map(String::as_str).collect();
        let expected: HashSet<&str> = FEATURE_KEYS.iter().copied().collect();

        assert_eq!(actual, expected);
    }

    #[test]
    fn composite_requests_emit_bounded_sub_intents() {
        let row = classify_request_event(&event(
            "e-composite",
            "Search the repo for context injection; implement the fix in src/main.rs; run cargo test; summarize the result",
            None,
        ));

        assert_eq!(row.features["is_composite"], true);
        assert_eq!(row.features["decomposition_candidate"], true);
        assert_eq!(row.features["decomposition_reason"], "sequence_separator");
        assert_eq!(row.features["sub_intent_count"], 4);
        assert_eq!(
            row.features["sub_intents"],
            json!(["search", "implement", "operate_tool", "summarize"])
        );
    }

    #[test]
    fn implementation_language_maps_to_implement_intent() {
        let row = classify_request_event(&event(
            "e-implement",
            "Implement the classifier change in src/request_classification.rs",
            None,
        ));

        assert_eq!(row.intent, RequestIntent::Implement);
        assert_eq!(row.response_contract, ResponseContract::ValidationRequired);
    }

    #[test]
    fn single_intent_with_conjunction_is_not_decomposed() {
        let row = classify_request_event(&event(
            "e-not-composite",
            "Explain Docker and Kubernetes networking",
            None,
        ));

        assert_eq!(row.features["is_composite"], false);
        assert_eq!(row.features["decomposition_candidate"], false);
        assert_eq!(row.features["decomposition_reason"], "none");
        assert_eq!(row.features["sub_intent_count"], 0);
        assert_eq!(row.features["sub_intents"], json!([]));
    }

    #[test]
    fn secret_candidates_set_secret_present() {
        let row = classify_request_event(&event(
            "e-secret",
            "The request included Authorization: Bearer sk-secret-value",
            None,
        ));

        assert!(row.risk.contains(&RequestRisk::SecretPresent));
        assert_eq!(row.recommended_route, RecommendedRoute::RefuseOrGuardrail);
        assert_eq!(row.features["has_secret_candidate"], true);
    }

    #[test]
    fn destructive_commands_set_destructive_command() {
        let row = classify_request_event(&event(
            "e-destructive",
            "Run rm -rf /tmp/agentic-os-cache to clean everything",
            None,
        ));

        assert!(row.risk.contains(&RequestRisk::DestructiveCommand));
        assert_eq!(row.complexity, RequestComplexity::L5HighRisk);
        assert_eq!(row.recommended_route, RecommendedRoute::RefuseOrGuardrail);
    }

    #[test]
    fn latest_current_phrasing_sets_external_current_info_required() {
        let row = classify_request_event(&event(
            "e-latest",
            "What is the latest vLLM release today?",
            None,
        ));

        assert!(row.risk.contains(&RequestRisk::ExternalCurrentInfoRequired));
        assert_eq!(row.domain, RequestDomain::LlmInference);
        assert_eq!(row.recommended_route, RecommendedRoute::WebRequired);
    }

    #[test]
    fn high_stakes_domains_set_high_stakes_risk() {
        for (id, summary, domain) in [
            (
                "e-medical",
                "Can you diagnose this medical symptom?",
                RequestDomain::Medical,
            ),
            (
                "e-legal",
                "Is this legal contract enforceable?",
                RequestDomain::Legal,
            ),
            (
                "e-finance",
                "Should I make this tax and investment move?",
                RequestDomain::Finance,
            ),
        ] {
            let row = classify_request_event(&event(id, summary, None));
            assert_eq!(row.domain, domain);
            assert!(row.risk.contains(&RequestRisk::HighStakes));
            assert_eq!(row.recommended_route, RecommendedRoute::RefuseOrGuardrail);
        }
    }

    #[test]
    fn infrastructure_terms_map_to_expected_domains() {
        let row = classify_request_event(&event(
            "e-domains",
            "kubectl deployment, Docker container, vLLM model, DNS proxy, and JWT auth",
            None,
        ));

        assert_eq!(row.domain, RequestDomain::Kubernetes);
        assert!(row.secondary_domains.contains(&RequestDomain::Docker));
        assert!(row.secondary_domains.contains(&RequestDomain::LlmInference));
        assert!(row.secondary_domains.contains(&RequestDomain::Networking));
        assert!(row.secondary_domains.contains(&RequestDomain::Security));
        assert!(row.secondary_domains.contains(&RequestDomain::Shell));
        assert_eq!(
            row.features["detected_domain_terms"],
            json!([
                "kubernetes",
                "docker",
                "llm_inference",
                "networking",
                "security",
                "shell"
            ])
        );
    }

    #[test]
    fn unknown_or_empty_events_produce_bounded_safe_defaults() {
        let row = classify_request_event(&event("e-empty", "", None));

        assert_eq!(row.intent, RequestIntent::Unknown);
        assert_eq!(row.domain, RequestDomain::Unknown);
        assert_eq!(row.artifact_type, RequestArtifactType::Unknown);
        assert_eq!(row.risk, vec![RequestRisk::Unknown]);
        assert_eq!(row.complexity, RequestComplexity::Unknown);
        assert_eq!(row.recommended_route, RecommendedRoute::Unknown);
        assert_eq!(row.response_contract, ResponseContract::Unknown);
        assert_eq!(row.features["char_count"], 0);
        assert_eq!(row.features["estimated_tokens"], 0);
    }

    #[test]
    fn non_empty_generic_requests_use_safe_fallback_labels() {
        let row = classify_request_event(&event("e-generic", "Can you help with this?", None));

        assert_eq!(row.intent, RequestIntent::Explain);
        assert_eq!(row.domain, RequestDomain::Generic);
        assert_eq!(row.artifact_type, RequestArtifactType::PlainText);
        assert_eq!(row.risk, vec![RequestRisk::None]);
        assert_ne!(row.complexity, RequestComplexity::Unknown);
        assert_ne!(row.recommended_route, RecommendedRoute::Unknown);
        assert_ne!(row.response_contract, ResponseContract::Unknown);
    }

    #[test]
    fn features_do_not_copy_raw_text() {
        let raw_secret = "sk-raw-secret-value";
        let row = classify_request_event(&event(
            "e-privacy",
            &format!("Please classify this Authorization: Bearer {raw_secret}"),
            Some("Raw evidence body should not be copied into features"),
        ));
        let serialized = row.features.to_string();

        assert!(!serialized.contains(raw_secret));
        assert!(!serialized.contains("Raw evidence body"));
        assert!(feature_string_values_are_bounded(&row.features));
    }

    #[test]
    fn classifiable_request_event_selection_is_bounded() {
        let user_event = event("e-user", "hello", None);
        assert!(is_classifiable_request_event(&user_event));

        let empty_user_event = event("e-empty-user", "", None);
        assert!(!is_classifiable_request_event(&empty_user_event));

        let whitespace_user_event = event("e-whitespace-user", " \n\t ", None);
        assert!(!is_classifiable_request_event(&whitespace_user_event));

        let evidence_only_event = event("e-evidence-only", "", Some("hello from evidence"));
        assert!(is_classifiable_request_event(&evidence_only_event));

        let mut request_role = event("e-role", "hello", None);
        request_role.event_type = "checkpoint".to_string();
        request_role.event_role = Some("request".to_string());
        assert!(is_classifiable_request_event(&request_role));

        let mut maintenance = event("e-maint", "summary maintenance", None);
        maintenance.event_type = "summary".to_string();
        maintenance.event_role = None;
        assert!(!is_classifiable_request_event(&maintenance));
    }

    #[test]
    fn backfill_sql_does_not_treat_separator_newline_as_request_text() {
        let source = include_str!("request_classification.rs");

        assert!(source.contains("btrim(coalesce(e.summary, '') || coalesce(e.evidence, '')"));
        assert!(!source.contains("btrim(coalesce(e.summary, '') || E'\\n'"));
    }

    #[test]
    fn live_policy_is_disabled_by_default() {
        let classification = classify_request_event(&event(
            "e-disabled",
            "Can you diagnose this medical issue?",
            None,
        ));

        assert_eq!(
            evaluate_live_policy(&classification, &LivePolicyConfig::default()),
            None
        );
    }

    #[test]
    fn live_policy_handles_high_stakes_current_info_and_destructive_commands() {
        let enabled = LivePolicyConfig {
            enabled: true,
            policy_version: "v1".to_string(),
        };

        let medical = classify_request_event(&event(
            "e-live-medical",
            "Can you diagnose this medical issue?",
            None,
        ));
        let decision = evaluate_live_policy(&medical, &enabled).expect("medical should be stopped");
        assert_eq!(decision.action, "refuse_or_guardrail");
        assert_eq!(decision.reason, "objective_risk");
        assert_eq!(decision.response_contract, ResponseContract::Refusal);

        let latest = classify_request_event(&event(
            "e-live-latest",
            "What is the latest Docker release today?",
            None,
        ));
        let decision = evaluate_live_policy(&latest, &enabled).expect("latest should need web");
        assert_eq!(decision.action, "web_required");
        assert_eq!(decision.reason, "external_current_info_required");

        let destructive = classify_request_event(&event(
            "e-live-destructive",
            "Please run kubectl delete namespace production",
            None,
        ));
        let decision =
            evaluate_live_policy(&destructive, &enabled).expect("destructive should be stopped");
        assert_eq!(decision.action, "refuse_or_guardrail");
        assert_eq!(decision.reason, "objective_risk");
    }

    #[test]
    fn clarification_route_returns_bounded_contract() {
        let enabled = LivePolicyConfig {
            enabled: true,
            policy_version: "v1".to_string(),
        };
        let classification = classify_request_event(&event("e-clarify", "Please restart it", None));

        assert_eq!(
            classification.recommended_route,
            RecommendedRoute::AskClarification
        );
        let decision =
            evaluate_live_policy(&classification, &enabled).expect("clarification should apply");
        assert_eq!(decision.action, "ask_clarification");
        assert_eq!(
            decision.response_contract,
            ResponseContract::ClarificationQuestion
        );
    }

    #[test]
    fn small_and_strong_model_routes_remain_shadow_only() {
        let enabled = LivePolicyConfig {
            enabled: true,
            policy_version: "v1".to_string(),
        };
        let small = classify_request_event(&event(
            "e-small",
            "Please explain how Docker container networking works at a high level",
            None,
        ));
        assert_eq!(small.recommended_route, RecommendedRoute::SmallLocalModel);
        assert_eq!(evaluate_live_policy(&small, &enabled), None);

        let long_text = format!("Explain this architecture. {}", "detail ".repeat(400));
        let strong = classify_request_event(&event("e-strong", &long_text, None));
        assert_eq!(strong.recommended_route, RecommendedRoute::StrongLocalModel);
        assert_eq!(evaluate_live_policy(&strong, &enabled), None);
    }

    #[test]
    fn live_policy_bounding_helpers_reject_unbounded_labels() {
        assert_eq!(
            bounded_live_policy_action("something else"),
            "refuse_or_guardrail"
        );
        assert_eq!(
            bounded_live_policy_reason("raw path /tmp/foo"),
            "objective_risk"
        );
        assert_eq!(bounded_live_policy_bypass("raw unknown"), "shadow_only");
        assert_eq!(bounded_route("bad route"), "unknown");
        assert_eq!(bounded_risk("bad risk"), "unknown");
        assert_eq!(bounded_complexity("bad complexity"), "unknown");
    }

    fn event(id: &str, summary: &str, evidence: Option<&str>) -> AgentEvent {
        AgentEvent {
            id: id.to_string(),
            session_id: "session-1".to_string(),
            repo: "agent-os".to_string(),
            actor: "user".to_string(),
            event_type: "user_message".to_string(),
            summary: summary.to_string(),
            evidence: evidence.map(str::to_string),
            metadata: json!({
                "payload": {
                    "kind": "test"
                }
            }),
            correlation_id: None,
            parent_event_id: None,
            trajectory_id: None,
            attempt_index: None,
            event_role: Some("request".to_string()),
            created_at: chrono::DateTime::parse_from_rfc3339("2026-05-23T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            summary_level: 0,
        }
    }

    fn feature_string_values_are_bounded(value: &Value) -> bool {
        match value {
            Value::String(value) => enum_inventory()
                .iter()
                .any(|(_name, variants)| variants.contains(&value.as_str())),
            Value::Array(values) => values.iter().all(feature_string_values_are_bounded),
            Value::Object(values) => values.values().all(feature_string_values_are_bounded),
            _ => true,
        }
    }
}
