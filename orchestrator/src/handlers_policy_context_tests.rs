use super::*;

#[test]
fn live_classification_overrides_generic_task_for_context_category() {
    let classification = crate::request_classification::classify_request_text(
        "agentic-os",
        "session-a",
        "Read README.md",
        None,
        "user_message",
    );

    assert_eq!(
        context_task_category("default task", Some(&classification)),
        crate::state::TaskCategory::Narrow
    );
}

#[test]
fn orchestration_policy_narrows_context_memory_levels() {
    let classification = crate::request_classification::classify_request_text(
        "agentic-os",
        "session-a",
        "Read README.md",
        None,
        "user_message",
    );
    let orchestration_policy =
        orchestration_policy::derive_orchestration_policy(&classification, "Read README.md", false);
    let mut context_policy =
        crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Moderate);

    apply_orchestration_context_limits(&mut context_policy, &orchestration_policy);

    assert_eq!(context_policy.l2_limit, 0);
    assert_eq!(context_policy.l3_limit, 0);
    assert_eq!(context_policy.failure_limit, 0);
    assert!(!orchestration_policy
        .context_sources
        .contains(&orchestration_policy::ContextSource::CompiledSummaries));
    assert!(!orchestration_policy
        .context_sources
        .contains(&orchestration_policy::ContextSource::TotalRecall));
}
