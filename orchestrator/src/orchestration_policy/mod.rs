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

pub mod base;
pub mod context_types;
pub mod intents;
pub mod intents_mutating;
pub mod intents_read;
pub mod overlay_types;
pub mod overlays;
pub mod posture_types;
pub mod store;
pub mod tool_types;
pub mod types;

pub use store::{compact_policy_metadata, persist_orchestration_policy};
pub use types::{
    ContextSource, EditPolicy, GitPolicy, OrchestrationPolicy, PromptRefinementPolicy, RiskPolicy,
    RuntimePolicy, ScopePolicy, ToolCapability, ValidationPolicy, POLICY_SCHEMA_VERSION,
    POLICY_SOURCE_DETERMINISTIC_RULES,
};

// ---------------------------------------------------------------------------
// derive_orchestration_policy
// ---------------------------------------------------------------------------

use crate::request_classification::RequestClassification;
use base::base_policy;
use overlays::{
    apply_prompt_refinement_overlay, apply_risk_overlays, normalize_blocked_capabilities,
};

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

    apply_risk_overlays(
        risk,
        raw_capture_enabled,
        &mut allowed,
        &mut required,
        &mut blocked,
        &mut scope,
        &mut context_sources,
        &mut risk_policy,
    );

    // --- prompt/spec refinement overlay ---
    let mut refined_allowed = allowed.clone();
    let mut refined_edit = base.edit;
    let mut refined_git = base.git;
    let mut refined_runtime = base.runtime;
    let prompt_refinement = apply_prompt_refinement_overlay(
        request_text,
        risk,
        &mut refined_allowed,
        &mut refined_edit,
        &mut refined_git,
        &mut refined_runtime,
    );

    // Final normalization makes the "blocked wins" invariant explicit. This is
    // deliberately last so any earlier rule may add allowed/required tools
    // without needing to know all possible risk overlays.
    normalize_blocked_capabilities(&mut refined_allowed, &mut required, &blocked);

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
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    pub(crate) use super::*;
    pub(crate) mod support;
    pub(crate) use support as test_support;

    mod base;
    mod contract;
    mod prompt;
    mod raw_capture;
    mod risk;
}
