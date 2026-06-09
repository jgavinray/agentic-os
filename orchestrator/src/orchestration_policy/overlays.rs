use crate::orchestration_policy_types::{
    ContextSource, EditPolicy, GitPolicy, PromptRefinementPolicy, RiskPolicy, RuntimePolicy,
    ScopePolicy, ToolCapability,
};
use crate::request_classification::RequestRisk;

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_risk_overlays(
    risk: &[RequestRisk],
    raw_capture_enabled: bool,
    allowed: &mut Vec<ToolCapability>,
    required: &mut Vec<ToolCapability>,
    blocked: &mut Vec<ToolCapability>,
    scope: &mut Vec<ScopePolicy>,
    context_sources: &mut Vec<ContextSource>,
    risk_policy: &mut Vec<RiskPolicy>,
) {
    if risk.contains(&RequestRisk::ExternalCurrentInfoRequired) {
        if !required.contains(&ToolCapability::WebSearch) {
            required.push(ToolCapability::WebSearch);
        }
        if !allowed.contains(&ToolCapability::WebSearch) {
            allowed.push(ToolCapability::WebSearch);
        }
        push_unique(risk_policy, RiskPolicy::ExternalWebRequired);
    }

    if risk.contains(&RequestRisk::HighStakes) {
        push_unique(risk_policy, RiskPolicy::HighStakesGuardrail);
        for cap in &[
            ToolCapability::FileEdit,
            ToolCapability::ShellMutation,
            ToolCapability::DockerMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
            ToolCapability::GitWrite,
        ] {
            push_unique(blocked, *cap);
        }
    }

    if risk.contains(&RequestRisk::DestructiveCommand) {
        push_unique(risk_policy, RiskPolicy::DestructiveRequiresConfirmation);
        for cap in &[
            ToolCapability::ShellMutation,
            ToolCapability::DockerMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
        ] {
            push_unique(blocked, *cap);
        }
    }

    if raw_capture_enabled {
        push_unique(risk_policy, RiskPolicy::RawCaptureEnabled);
        push_unique(context_sources, ContextSource::RawCaptureFeatures);
    }

    if risk.contains(&RequestRisk::SecretPresent) && raw_capture_enabled {
        push_unique(risk_policy, RiskPolicy::SecretCaptureAllowed);
    }

    push_unique(scope, ScopePolicy::NoScp);
}

pub(crate) fn apply_prompt_refinement_overlay(
    request_text: &str,
    risk: &[RequestRisk],
    allowed: &mut Vec<ToolCapability>,
    edit: &mut EditPolicy,
    git: &mut GitPolicy,
    runtime: &mut RuntimePolicy,
) -> PromptRefinementPolicy {
    let lower = request_text.to_ascii_lowercase();
    let has_prompt_word = contains_any(
        &lower,
        &["prompt", "spec", "task", "deliverable", "constraints"],
    );
    let has_feedback_word = contains_any(
        &lower,
        &["feedback", "review", "rewrite", "refine", "is this good"],
    );

    if !(has_prompt_word && has_feedback_word) {
        return PromptRefinementPolicy::None;
    }

    allowed.clear();
    allowed.push(ToolCapability::RepoRead);
    if risk.contains(&RequestRisk::ExternalCurrentInfoRequired) {
        allowed.push(ToolCapability::WebSearch);
    }
    *edit = EditPolicy::ReadOnly;
    *git = GitPolicy::NoGitChanges;
    *runtime = RuntimePolicy::NoRestart;
    PromptRefinementPolicy::MultiPassReview
}

pub(crate) fn normalize_blocked_capabilities(
    allowed: &mut Vec<ToolCapability>,
    required: &mut Vec<ToolCapability>,
    blocked: &[ToolCapability],
) {
    for blocked_cap in blocked {
        allowed.retain(|cap| cap != blocked_cap);
        required.retain(|cap| cap != blocked_cap);
    }
}

fn push_unique<T: PartialEq>(vec: &mut Vec<T>, item: T) {
    if !vec.contains(&item) {
        vec.push(item);
    }
}

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}
