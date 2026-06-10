use super::test_support::classification;
use super::*;
use crate::request_classification::{RequestArtifactType, RequestIntent, RequestRisk};

#[test]
fn test_prompt_review_overlay() {
    let c = classification(
        RequestIntent::Explain,
        vec![RequestRisk::None],
        RequestArtifactType::PlainText,
    );
    let policy = derive_orchestration_policy(
        &c,
        "Review the prompt spec and rewrite it - is this good?",
        false,
    );

    assert_eq!(
        policy.prompt_refinement_policy,
        PromptRefinementPolicy::MultiPassReview
    );
    assert_eq!(policy.edit_policy, EditPolicy::ReadOnly);
    assert_eq!(policy.git_policy, GitPolicy::NoGitChanges);
    assert_eq!(policy.runtime_policy, RuntimePolicy::NoRestart);
    assert!(policy.allowed_tools.contains(&ToolCapability::RepoRead));
    assert!(
        !policy.allowed_tools.contains(&ToolCapability::WebSearch),
        "web_search should not be present without external_web_required"
    );
}

#[test]
fn implementation_text_with_task_and_review_words_keeps_edit_tools() {
    // "task" + "review" are everyday words in implementation requests. They
    // must not trigger the prompt-refinement overlay, which clears the entire
    // allowed-tool surface.
    let c = classification(
        RequestIntent::Implement,
        vec![RequestRisk::None],
        RequestArtifactType::Code,
    );
    let policy = derive_orchestration_policy(
        &c,
        "Implement the task described in docs/spec.md, run cargo test, and review the failing tests.",
        false,
    );

    assert_eq!(
        policy.prompt_refinement_policy,
        PromptRefinementPolicy::None
    );
    assert!(policy.allowed_tools.contains(&ToolCapability::FileEdit));
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
        "Review the prompt spec and rewrite it - is this good?",
        false,
    );

    assert_eq!(
        policy.prompt_refinement_policy,
        PromptRefinementPolicy::MultiPassReview
    );
    assert!(policy.allowed_tools.contains(&ToolCapability::WebSearch));
}
