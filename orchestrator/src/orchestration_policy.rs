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
#[path = "orchestration_policy_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "orchestration_policy_contract_tests.rs"]
mod contract_tests;

#[cfg(test)]
#[path = "orchestration_policy_raw_capture_tests.rs"]
mod raw_capture_tests;
