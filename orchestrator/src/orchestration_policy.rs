//! Deterministic orchestration policy derivation.
//!
//! Request classification answers "what kind of request is this?"  This module
//! answers the next question: "given that classification, what operating
//! envelope should the orchestrator expose?"  The envelope is intentionally
//! broader than tool mediation alone: it covers eligible context sources,
//! allowed/required/blocked tool capabilities, edit scope, validation posture,
//! git behavior, runtime behavior, prompt/spec review workflow, and risk
//! overlays.
//!
//! The derivation function is pure, deterministic, and has no I/O. Persistence
//! is a separate append-only step so policy decisions can be audited without
//! mixing database behavior into the policy rules.

pub use crate::orchestration_policy_store::{
    compact_policy_metadata, persist_orchestration_policy,
};
pub use crate::orchestration_policy_types::{
    ContextSource, EditPolicy, GitPolicy, OrchestrationPolicy, PromptRefinementPolicy, RiskPolicy,
    RuntimePolicy, ScopePolicy, ToolCapability, ValidationPolicy, POLICY_SCHEMA_VERSION,
    POLICY_SOURCE_DETERMINISTIC_RULES,
};

// ---------------------------------------------------------------------------
// derive_orchestration_policy
// ---------------------------------------------------------------------------

use crate::orchestration_policy_base::base_policy;
use crate::request_classification::{RequestClassification, RequestRisk};

/// Derive a deterministic orchestration policy from a request classification.
///
/// The derivation has four ordered phases:
///
/// 1. Pick a base policy from `RequestIntent` and `RequestArtifactType`.
/// 2. Apply risk overlays from `RequestRisk` and raw-capture state.
/// 3. Apply the prompt/spec review overlay from bounded request text matching.
/// 4. Normalize conflicts so blocked capabilities win over allowed/required.
///
/// The function is pure: no database reads, no network calls, no randomness.
/// That makes policy rows rebuildable from classification labels, request text,
/// and raw-capture configuration.
pub fn derive_orchestration_policy(
    classification: &RequestClassification,
    request_text: &str,
    raw_capture_enabled: bool,
) -> OrchestrationPolicy {
    let intent = classification.intent;
    let risk = &classification.risk;
    let artifact_type = classification.artifact_type;

    // The base policy is intentionally intent-first. Risk overlays can only
    // reduce or require capabilities after the request's ordinary operating
    // shape has been selected.
    let base = base_policy(intent, artifact_type);

    let mut allowed = base.allowed;
    let mut required = base.required;
    let mut blocked = base.blocked;
    let mut scope = base.scope;
    let mut context_sources = base.context;

    // Risk overlays are additive. They record positive risk posture and append
    // blocked/required capabilities without replacing the intent-derived base.
    let mut risk_policy: Vec<RiskPolicy> = Vec::new();

    if risk.contains(&RequestRisk::ExternalCurrentInfoRequired) {
        // A current-information request must surface web search as required so
        // callers can distinguish "web would be nice" from "web is mandatory".
        if !required.contains(&ToolCapability::WebSearch) {
            required.push(ToolCapability::WebSearch);
        }
        // Keep `required_tools` a subset of `allowed_tools` unless a later
        // blocked-tools overlay removes the same capability.
        if !allowed.contains(&ToolCapability::WebSearch) {
            allowed.push(ToolCapability::WebSearch);
        }
        push_unique(&mut risk_policy, RiskPolicy::ExternalWebRequired);
    }

    if risk.contains(&RequestRisk::HighStakes) {
        push_unique(&mut risk_policy, RiskPolicy::HighStakesGuardrail);
        // High-stakes requests keep read/context behavior available but remove
        // mutation surfaces. The existing live request policy may still refuse
        // or demand a guardrail before this policy reaches a model.
        for cap in &[
            ToolCapability::FileEdit,
            ToolCapability::ShellMutation,
            ToolCapability::DockerMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
            ToolCapability::GitWrite,
        ] {
            push_unique(&mut blocked, *cap);
        }
    }

    if risk.contains(&RequestRisk::DestructiveCommand) {
        push_unique(
            &mut risk_policy,
            RiskPolicy::DestructiveRequiresConfirmation,
        );
        for cap in &[
            ToolCapability::ShellMutation,
            ToolCapability::DockerMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
        ] {
            push_unique(&mut blocked, *cap);
        }
    }

    if raw_capture_enabled {
        push_unique(&mut risk_policy, RiskPolicy::RawCaptureEnabled);
        push_unique(&mut context_sources, ContextSource::RawCaptureFeatures);
    }

    if risk.contains(&RequestRisk::SecretPresent) {
        if raw_capture_enabled {
            push_unique(&mut risk_policy, RiskPolicy::SecretCaptureAllowed);
        }
    }

    // `no_scp` is a policy invariant. If future base policies add other scope
    // modes, this invariant still prevents accidental cross-host file copying
    // unless a separate explicit policy is introduced.
    push_unique(&mut scope, ScopePolicy::NoScp);

    // --- prompt/spec refinement overlay ---
    let mut prompt_refinement = PromptRefinementPolicy::None;
    let mut refined_allowed = allowed.clone();
    let mut refined_edit = base.edit;
    let mut refined_git = base.git;
    let mut refined_runtime = base.runtime;

    let lower = request_text.to_ascii_lowercase();
    let has_prompt_word = contains_any(
        &lower,
        &["prompt", "spec", "task", "deliverable", "constraints"],
    );
    let has_feedback_word = contains_any(
        &lower,
        &["feedback", "review", "rewrite", "refine", "is this good"],
    );

    if has_prompt_word && has_feedback_word {
        prompt_refinement = PromptRefinementPolicy::MultiPassReview;
        // Prompt/spec review is a review workflow, not an implementation
        // request. Narrow the model to repository reading and optional web
        // lookup so the review cannot silently become an edit/tool operation.
        refined_allowed.clear();
        refined_allowed.push(ToolCapability::RepoRead);
        if risk.contains(&RequestRisk::ExternalCurrentInfoRequired) {
            refined_allowed.push(ToolCapability::WebSearch);
        }
        refined_edit = EditPolicy::ReadOnly;
        refined_git = GitPolicy::NoGitChanges;
        refined_runtime = RuntimePolicy::NoRestart;
    }

    // Final normalization makes the "blocked wins" invariant explicit. This is
    // deliberately last so any earlier rule may add allowed/required tools
    // without needing to know all possible risk overlays.
    for blocked_cap in &blocked {
        refined_allowed.retain(|cap| cap != blocked_cap);
        required.retain(|cap| cap != blocked_cap);
    }

    OrchestrationPolicy {
        context_sources,
        allowed_tools: refined_allowed,
        required_tools: required,
        blocked_tools: blocked,
        edit_policy: refined_edit,
        validation_policy: base.validation,
        git_policy: refined_git,
        runtime_policy: refined_runtime,
        scope_policy: scope,
        prompt_refinement_policy: prompt_refinement,
        risk_policy,
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Push an item into a vec only if it is not already present.
///
/// Used by the risk-overlay logic to prevent `risk_policy` and
/// `blocked_tools` from accumulating duplicates as more rules are added.
fn push_unique<T: PartialEq>(vec: &mut Vec<T>, item: T) {
    if !vec.contains(&item) {
        vec.push(item);
    }
}

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_classification::{
        RequestArtifactType, RequestClassification, RequestIntent, RequestRisk,
    };
    use chrono::Utc;

    fn classification(
        intent: RequestIntent,
        risk: Vec<RequestRisk>,
        artifact_type: RequestArtifactType,
    ) -> RequestClassification {
        RequestClassification {
            event_id: "test-event".to_string(),
            repo: "test-repo".to_string(),
            session_id: "test-session".to_string(),
            trajectory_id: None,
            event_created_at: Utc::now(),
            classified_at: Utc::now(),
            classification_schema_version: 1,
            routing_policy_version: "deterministic-v1".to_string(),
            classifier_source: "deterministic_rules".to_string(),
            intent,
            domain: crate::request_classification::RequestDomain::Generic,
            secondary_domains: vec![],
            artifact_type,
            risk,
            complexity: crate::request_classification::RequestComplexity::L1Simple,
            recommended_route: crate::request_classification::RecommendedRoute::SmallLocalModel,
            response_contract: crate::request_classification::ResponseContract::DirectAnswer,
            features: serde_json::json!({}),
        }
    }

    // 1. Explain allows web_search, read_only, no_git_changes
    #[test]
    fn test_explain_allows_web_search_read_only_no_git() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::None],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this", false);

        assert!(policy.allowed_tools.contains(&ToolCapability::WebSearch));
        assert_eq!(policy.edit_policy, EditPolicy::ReadOnly);
        assert_eq!(policy.git_policy, GitPolicy::NoGitChanges);
    }

    // 2. Explain + ExternalCurrentInfoRequired requires web_search
    #[test]
    fn test_explain_external_info_requires_web_search() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::ExternalCurrentInfoRequired],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this", false);

        assert!(policy.required_tools.contains(&ToolCapability::WebSearch));
    }

    // 3. Debug Code => single_file_edit, file_edit allowed, targeted_tests
    #[test]
    fn test_debug_code_single_file_edit() {
        let c = classification(
            RequestIntent::Debug,
            vec![RequestRisk::None],
            RequestArtifactType::Code,
        );
        let policy = derive_orchestration_policy(&c, "fix the bug in main.rs", false);

        assert_eq!(policy.edit_policy, EditPolicy::SingleFileEdit);
        assert!(policy.allowed_tools.contains(&ToolCapability::FileEdit));
        assert_eq!(policy.validation_policy, ValidationPolicy::TargetedTests);
    }

    // 4. ModifyConfig Code => build
    #[test]
    fn test_modify_config_code_build() {
        let c = classification(
            RequestIntent::ModifyConfig,
            vec![RequestRisk::None],
            RequestArtifactType::Code,
        );
        let policy = derive_orchestration_policy(&c, "edit the config", false);

        assert_eq!(policy.validation_policy, ValidationPolicy::Build);
    }

    // 5. ModifyConfig Yaml => targeted_tests
    #[test]
    fn test_modify_config_yaml_targeted_tests() {
        let c = classification(
            RequestIntent::ModifyConfig,
            vec![RequestRisk::None],
            RequestArtifactType::Yaml,
        );
        let policy = derive_orchestration_policy(&c, "edit the yaml config", false);

        assert_eq!(policy.validation_policy, ValidationPolicy::TargetedTests);
    }

    // 5b. ModifyConfig Json => targeted_tests
    #[test]
    fn test_modify_config_json_targeted_tests() {
        let c = classification(
            RequestIntent::ModifyConfig,
            vec![RequestRisk::None],
            RequestArtifactType::Json,
        );
        let policy = derive_orchestration_policy(&c, "edit the json config", false);

        assert_eq!(policy.validation_policy, ValidationPolicy::TargetedTests);
    }

    // 6. DestructiveCommand blocks shell_mutation, deploy, restart_service
    #[test]
    fn test_destructive_command_blocks_mutation() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::DestructiveCommand],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "run rm -rf /tmp", false);

        assert!(policy
            .blocked_tools
            .contains(&ToolCapability::ShellMutation));
        assert!(policy.blocked_tools.contains(&ToolCapability::Deploy));
        assert!(policy
            .blocked_tools
            .contains(&ToolCapability::RestartService));
    }

    // 7. HighStakes adds high_stakes_guardrail and blocks mutations
    #[test]
    fn test_high_stakes_guardrail_and_blocks() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::HighStakes],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this medical issue", false);

        assert!(policy
            .risk_policy
            .contains(&RiskPolicy::HighStakesGuardrail));
        assert!(policy.blocked_tools.contains(&ToolCapability::FileEdit));
        assert!(policy
            .blocked_tools
            .contains(&ToolCapability::ShellMutation));
        assert!(policy
            .blocked_tools
            .contains(&ToolCapability::DockerMutation));
        assert!(policy.blocked_tools.contains(&ToolCapability::Deploy));
        assert!(policy
            .blocked_tools
            .contains(&ToolCapability::RestartService));
        assert!(policy.blocked_tools.contains(&ToolCapability::GitWrite));
    }

    #[test]
    fn test_implement_allows_edit_create_and_targeted_validation_only() {
        let c = classification(
            RequestIntent::Implement,
            vec![RequestRisk::None],
            RequestArtifactType::Code,
        );
        let policy =
            derive_orchestration_policy(&c, "implement the request classifier change", false);

        assert!(policy.allowed_tools.contains(&ToolCapability::RepoRead));
        assert!(policy.allowed_tools.contains(&ToolCapability::FileRead));
        assert!(policy.allowed_tools.contains(&ToolCapability::FileEdit));
        assert_eq!(policy.validation_policy, ValidationPolicy::TargetedTests);
        assert_eq!(policy.git_policy, GitPolicy::NoGitChanges);
        assert_eq!(policy.runtime_policy, RuntimePolicy::NoRestart);
        assert!(policy.blocked_tools.contains(&ToolCapability::ShellRead));
        assert!(policy
            .blocked_tools
            .contains(&ToolCapability::ShellMutation));
        assert!(policy.blocked_tools.contains(&ToolCapability::GitWrite));
        assert!(policy.blocked_tools.contains(&ToolCapability::Deploy));
        assert!(policy
            .blocked_tools
            .contains(&ToolCapability::RestartService));
        assert!(policy
            .scope_policy
            .contains(&ScopePolicy::IgnoreUnrelatedDirtyChanges));
    }

    // 8. Unknown minimal read-only empty allowed_tools
    #[test]
    fn test_unknown_minimal() {
        let c = classification(
            RequestIntent::Unknown,
            vec![RequestRisk::None],
            RequestArtifactType::Unknown,
        );
        let policy = derive_orchestration_policy(&c, "???", false);

        assert!(policy.allowed_tools.is_empty());
        assert_eq!(policy.edit_policy, EditPolicy::ReadOnly);
    }

    // 9. NoScp always present
    #[test]
    fn test_no_scp_always_present() {
        for intent in [
            RequestIntent::Explain,
            RequestIntent::Debug,
            RequestIntent::Implement,
            RequestIntent::ModifyConfig,
            RequestIntent::GenerateConfig,
            RequestIntent::OperateTool,
            RequestIntent::Plan,
            RequestIntent::Summarize,
            RequestIntent::Classify,
            RequestIntent::Search,
            RequestIntent::Unknown,
        ] {
            let c = classification(
                intent,
                vec![RequestRisk::None],
                RequestArtifactType::PlainText,
            );
            let policy = derive_orchestration_policy(&c, "test", false);
            assert!(
                policy.scope_policy.contains(&ScopePolicy::NoScp),
                "{intent:?} must include no_scp"
            );
        }
    }

    // 10. serde round-trip policy
    #[test]
    fn test_serde_round_trip() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::None],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this", false);

        let serialized = serde_json::to_string(&policy).expect("serialize");
        let deserialized: OrchestrationPolicy =
            serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(policy, deserialized);
    }

    // 11. prompt/spec review => multi_pass_review, read_only, narrowed tools
    #[test]
    fn test_prompt_review_overlay() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::None],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(
            &c,
            "Review the prompt spec and rewrite it — is this good?",
            false,
        );

        assert_eq!(
            policy.prompt_refinement_policy,
            PromptRefinementPolicy::MultiPassReview
        );
        assert_eq!(policy.edit_policy, EditPolicy::ReadOnly);
        assert_eq!(policy.git_policy, GitPolicy::NoGitChanges);
        assert_eq!(policy.runtime_policy, RuntimePolicy::NoRestart);
        // narrowed to repo_read only (no ExternalCurrentInfoRequired)
        assert!(policy.allowed_tools.contains(&ToolCapability::RepoRead));
        assert!(
            !policy.allowed_tools.contains(&ToolCapability::WebSearch),
            "web_search should not be present without external_web_required"
        );
    }

    #[test]
    fn test_prompt_review_overlay_with_external_info() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::ExternalCurrentInfoRequired],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(
            &c,
            "Review the prompt spec and rewrite it — is this good?",
            false,
        );

        assert_eq!(
            policy.prompt_refinement_policy,
            PromptRefinementPolicy::MultiPassReview
        );
        // web_search should be present because ExternalCurrentInfoRequired adds it
        assert!(policy.allowed_tools.contains(&ToolCapability::WebSearch));
    }

    // -----------------------------------------------------------------------
    // Phase 2 — persistence & migration tests
    // -----------------------------------------------------------------------

    // 1. vector label serialization produces snake_case JSON arrays
    #[test]
    fn test_vector_label_serialization_snake_case() {
        let c = classification(
            RequestIntent::Debug,
            vec![RequestRisk::HighStakes, RequestRisk::DestructiveCommand],
            RequestArtifactType::Code,
        );
        let policy = derive_orchestration_policy(&c, "fix the bug", false);

        // context_sources: all snake_case
        let ctx_json: Vec<&str> = policy.context_sources.iter().map(|s| s.as_str()).collect();
        assert!(ctx_json
            .iter()
            .all(|s| s.chars().all(|c| c.is_lowercase() || c == '_')));

        // allowed_tools: snake_case
        let allowed_json: Vec<&str> = policy.allowed_tools.iter().map(|t| t.as_str()).collect();
        assert!(allowed_json
            .iter()
            .all(|s| s.chars().all(|c| c.is_lowercase() || c == '_')));

        // risk_policy: snake_case
        let risk_json: Vec<&str> = policy.risk_policy.iter().map(|r| r.as_str()).collect();
        assert!(risk_json
            .iter()
            .all(|s| s.chars().all(|c| c.is_lowercase() || c == '_')));

        // scope_policy: snake_case
        let scope_json: Vec<&str> = policy.scope_policy.iter().map(|s| s.as_str()).collect();
        assert!(scope_json
            .iter()
            .all(|s| s.chars().all(|c| c.is_lowercase() || c == '_')));

        // Verify actual labels match expected snake_case values
        assert!(allowed_json.contains(&"file_read"));
        assert!(allowed_json.contains(&"repo_read"));
        assert!(risk_json.contains(&"high_stakes_guardrail"));
        assert!(risk_json.contains(&"destructive_requires_confirmation"));
        assert!(scope_json.contains(&"no_scp"));
    }

    // 2. scalar as_str labels remain snake_case
    #[test]
    fn test_scalar_as_str_labels_snake_case() {
        let c = classification(
            RequestIntent::Debug,
            vec![RequestRisk::None],
            RequestArtifactType::Code,
        );
        let policy = derive_orchestration_policy(&c, "fix the bug", false);

        // edit_policy
        let edit = policy.edit_policy.as_str();
        assert_eq!(edit, "single_file_edit");

        // validation_policy
        let val = policy.validation_policy.as_str();
        assert_eq!(val, "targeted_tests");

        // git_policy
        let git = policy.git_policy.as_str();
        assert_eq!(git, "no_git_changes");

        // runtime_policy
        let rt = policy.runtime_policy.as_str();
        assert_eq!(rt, "no_restart");

        // prompt_refinement_policy
        let pr = policy.prompt_refinement_policy.as_str();
        assert_eq!(pr, "none");
    }

    // 3. migration file contains expected columns and structure
    #[test]
    fn test_migration_has_expected_structure() {
        let migration = include_str!("../migrations/V16__orchestration_policies.sql");

        // Table name
        assert!(
            migration.contains("agent_orchestration_policies"),
            "migration must reference agent_orchestration_policies"
        );

        // Key columns
        assert!(
            migration.contains("policy_schema_version"),
            "must have policy_schema_version"
        );
        assert!(
            migration.contains("prompt_refinement_policy"),
            "must have prompt_refinement_policy"
        );
        assert!(
            migration.contains("context_sources"),
            "must have context_sources"
        );
        assert!(
            migration.contains("allowed_tools"),
            "must have allowed_tools"
        );
        assert!(
            migration.contains("required_tools"),
            "must have required_tools"
        );
        assert!(
            migration.contains("blocked_tools"),
            "must have blocked_tools"
        );
        assert!(migration.contains("risk_policy"), "must have risk_policy");
        assert!(migration.contains("scope_policy"), "must have scope_policy");

        // JSONB columns
        assert!(
            migration.contains("JSONB"),
            "must use JSONB for array columns"
        );

        // Indexes
        assert!(
            migration.contains("session_id"),
            "must have session_id index"
        );
        assert!(migration.contains("repo"), "must have repo index");
        assert!(migration.contains("event_id"), "must have event_id index");
        assert!(migration.contains("intent"), "must have intent index");
        assert!(
            migration.contains("prompt_refinement_policy"),
            "must have prompt_refinement index"
        );
    }

    // 4. compact_policy_metadata produces correct JSON shape
    #[test]
    fn test_compact_policy_metadata() {
        let c = classification(
            RequestIntent::Debug,
            vec![RequestRisk::None],
            RequestArtifactType::Code,
        );
        let policy = derive_orchestration_policy(&c, "fix the bug", false);

        let meta = compact_policy_metadata(&c, &policy);

        // scalar fields
        assert!(meta["intent"].is_string());
        assert!(meta["recommended_route"].is_string());
        assert!(meta["edit_policy"].is_string());
        assert!(meta["validation_policy"].is_string());
        assert!(meta["git_policy"].is_string());
        assert!(meta["runtime_policy"].is_string());
        assert!(meta["prompt_refinement_policy"].is_string());

        // array fields
        assert!(meta["context_sources"].is_array());
        assert!(meta["allowed_tools"].is_array());
        assert!(meta["required_tools"].is_array());
        assert!(meta["blocked_tools"].is_array());
        assert!(meta["scope_policy"].is_array());
        assert!(meta["risk_policy"].is_array());

        // schema version & source
        assert_eq!(meta["policy_schema_version"], 1);
        assert_eq!(meta["source"], "deterministic_rules");

        // snake_case labels
        let allowed: Vec<&str> = meta["allowed_tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(allowed.contains(&"repo_read"));

        let scope: Vec<&str> = meta["scope_policy"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(scope.contains(&"no_scp"));
    }

    // -----------------------------------------------------------------------
    // Phase 3 — blocked_tools authority & required⊆allowed invariants
    // -----------------------------------------------------------------------

    // 1. Summarize + ExternalCurrentInfoRequired => web_search is both allowed and required.
    #[test]
    fn test_summarize_external_info_requires_and_allows_web_search() {
        let c = classification(
            RequestIntent::Summarize,
            vec![RequestRisk::ExternalCurrentInfoRequired],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "summarize the latest release", false);

        // web_search must be in allowed_tools
        assert!(
            policy.allowed_tools.contains(&ToolCapability::WebSearch),
            "web_search must be allowed for Summarize+ExternalCurrentInfoRequired"
        );
        // web_search must be in required_tools
        assert!(
            policy.required_tools.contains(&ToolCapability::WebSearch),
            "web_search must be required for Summarize+ExternalCurrentInfoRequired"
        );
    }

    // 2. Debug + Code + HighStakes => file_edit is blocked and not allowed.
    #[test]
    fn test_debug_code_high_stakes_blocks_file_edit() {
        let c = classification(
            RequestIntent::Debug,
            vec![RequestRisk::HighStakes],
            RequestArtifactType::Code,
        );
        let policy = derive_orchestration_policy(&c, "fix the bug in main.rs", false);

        // file_edit must be blocked
        assert!(
            policy.blocked_tools.contains(&ToolCapability::FileEdit),
            "file_edit must be blocked under HighStakes"
        );
        // file_edit must NOT be in allowed_tools
        assert!(
            !policy.allowed_tools.contains(&ToolCapability::FileEdit),
            "file_edit must not be allowed when blocked"
        );
        // file_edit must NOT be in required_tools
        assert!(
            !policy.required_tools.contains(&ToolCapability::FileEdit),
            "file_edit must not be required when blocked"
        );
    }

    // 3. DestructiveCommand on OperateTool => shell_mutation/deploy/restart_service blocked and absent.
    #[test]
    fn test_destructive_command_operate_tool_blocks_mutation_tools() {
        let c = classification(
            RequestIntent::OperateTool,
            vec![RequestRisk::DestructiveCommand],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "run rm -rf /tmp", false);

        let blocked_caps = [
            ToolCapability::ShellMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
        ];

        for cap in &blocked_caps {
            assert!(
                policy.blocked_tools.contains(cap),
                "{cap:?} must be blocked under DestructiveCommand"
            );
            assert!(
                !policy.allowed_tools.contains(cap),
                "{cap:?} must not be allowed when blocked"
            );
            assert!(
                !policy.required_tools.contains(cap),
                "{cap:?} must not be required when blocked"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Phase 4 — raw-capture state in deterministic policy
    // -----------------------------------------------------------------------

    // 1. raw_capture_enabled true with RequestRisk::None emits RawCaptureEnabled
    //    and does not emit LowRisk.
    #[test]
    fn test_raw_capture_enabled_true_emits_raw_capture_enabled_no_low_risk() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::None],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this", true);

        assert!(
            policy.risk_policy.contains(&RiskPolicy::RawCaptureEnabled),
            "raw_capture_enabled=true must emit RawCaptureEnabled"
        );
        assert!(
            !policy.risk_policy.contains(&RiskPolicy::LowRisk),
            "LowRisk must never be emitted by derive_orchestration_policy"
        );
    }

    // 2. SecretPresent with raw_capture_enabled true emits both
    //    RawCaptureEnabled and SecretCaptureAllowed.
    #[test]
    fn test_secret_present_raw_capture_emits_both() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::SecretPresent],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "show me the secrets", true);

        assert!(
            policy.risk_policy.contains(&RiskPolicy::RawCaptureEnabled),
            "SecretPresent+raw_capture must emit RawCaptureEnabled"
        );
        assert!(
            policy
                .risk_policy
                .contains(&RiskPolicy::SecretCaptureAllowed),
            "SecretPresent+raw_capture must emit SecretCaptureAllowed"
        );
        // RawCaptureEnabled should appear exactly once (from the unconditional block above)
        assert_eq!(
            policy
                .risk_policy
                .iter()
                .filter(|&&r| r == RiskPolicy::RawCaptureEnabled)
                .count(),
            1,
            "RawCaptureEnabled must appear exactly once"
        );
    }

    // 3. SecretPresent with raw_capture_enabled false emits neither
    //    RawCaptureEnabled nor SecretCaptureAllowed.
    #[test]
    fn test_secret_present_no_raw_capture_emits_neither() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::SecretPresent],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "show me the secrets", false);

        assert!(
            !policy.risk_policy.contains(&RiskPolicy::RawCaptureEnabled),
            "SecretPresent+no raw_capture must not emit RawCaptureEnabled"
        );
        assert!(
            !policy
                .risk_policy
                .contains(&RiskPolicy::SecretCaptureAllowed),
            "SecretPresent+no raw_capture must not emit SecretCaptureAllowed"
        );
    }

    // 4. raw_capture_enabled true with RequestIntent::Explain includes
    //    ContextSource::RawCaptureFeatures without widening durable memory.
    #[test]
    fn test_raw_capture_explain_includes_raw_capture_features_only() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::None],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this", true);

        assert!(
            policy
                .context_sources
                .contains(&ContextSource::RawCaptureFeatures),
            "raw_capture_enabled=true must include RawCaptureFeatures"
        );
        assert!(
            !policy.context_sources.contains(&ContextSource::TotalRecall),
            "raw capture must not widen narrow explain requests into TotalRecall"
        );
        assert!(
            !policy
                .context_sources
                .contains(&ContextSource::CompiledSummaries),
            "raw capture must not inject compiled summaries into narrow explain requests"
        );
    }

    // 5. raw_capture_enabled false with RequestIntent::Explain does not
    //    include ContextSource::RawCaptureFeatures.
    #[test]
    fn test_no_raw_capture_explain_no_raw_capture_features() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::None],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this", false);

        assert!(
            !policy
                .context_sources
                .contains(&ContextSource::RawCaptureFeatures),
            "raw_capture_enabled=false must not include RawCaptureFeatures"
        );
    }

    // 6. raw_capture_enabled true emits RawCaptureFeatures exactly once.
    #[test]
    fn test_raw_capture_features_no_duplicates() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::None],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this", true);

        let count = policy
            .context_sources
            .iter()
            .filter(|&&s| s == ContextSource::RawCaptureFeatures)
            .count();
        assert_eq!(count, 1, "RawCaptureFeatures must appear exactly once");
    }
}
