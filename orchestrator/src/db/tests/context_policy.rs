use super::*;

#[test]
fn context_policy_small_task_includes_l0_and_l1() {
    let mut evidence = ContextEvidence::default();
    evidence.l0_recent = vec![event_at_level("edit", "fresh implementation detail", 0)];
    evidence.l1_matching = vec![event_at_level("summary", "matching prior task summary", 1)];
    evidence.l2_repo = vec![event_at_level("summary", "repo convention", 2)];

    let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
    let (out, stats) =
        build_layered_context("r", "fix bug", &evidence, &[], &[], &policy, 8000, 4000);

    assert!(out.contains("== Recent Evidence =="));
    assert!(out.contains("fresh implementation detail"));
    assert!(out.contains("== Relevant Session Summaries =="));
    assert!(out.contains("matching prior task summary"));
    assert_eq!(stats.l0_items_injected, 1);
    assert_eq!(stats.l1_items_injected, 1);
}

#[test]
fn context_policy_broad_task_includes_l1_l2_and_l0_tail() {
    let mut evidence = ContextEvidence::default();
    evidence.l0_recent = vec![event_at_level("edit", "latest local change", 0)];
    evidence.l1_matching = vec![event_at_level("summary", "session summary", 1)];
    evidence.l2_repo = vec![event_at_level("summary", "repo-level pattern", 2)];

    let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Broad);
    let (out, _stats) = build_layered_context(
        "r",
        "large refactor",
        &evidence,
        &[],
        &[],
        &policy,
        8000,
        4000,
    );

    assert!(out.contains("latest local change"));
    assert!(out.contains("session summary"));
    assert!(out.contains("repo-level pattern"));
}

#[test]
fn context_policy_architecture_task_prioritizes_l2_l3() {
    let mut evidence = ContextEvidence::default();
    evidence.l2_repo = vec![event_at_level("summary", "repository decision record", 2)];
    evidence.l3_project = vec![event_at_level("summary", "durable architecture truth", 3)];

    let policy =
        crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Architecture);
    let (out, stats) = build_layered_context(
        "r",
        "architecture review",
        &evidence,
        &[],
        &[],
        &policy,
        8000,
        4000,
    );

    let l3_pos = out.find("durable architecture truth").unwrap();
    let l2_pos = out.find("repository decision record").unwrap();
    assert!(l3_pos < l2_pos);
    assert_eq!(stats.l2_items_injected, 1);
    assert_eq!(stats.l3_items_injected, 1);
}

#[test]
fn higher_levels_do_not_eliminate_l0_tail() {
    let mut evidence = ContextEvidence::default();
    evidence.l0_recent = vec![event_at_level("edit", "raw event still included", 0)];
    evidence.l3_project = vec![event_at_level("summary", "durable compressed memory", 3)];

    let policy =
        crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Architecture);
    let (out, _stats) = build_layered_context(
        "r",
        "architecture",
        &evidence,
        &[],
        &[],
        &policy,
        8000,
        4000,
    );

    assert!(out.contains("raw event still included"));
    assert!(out.contains("durable compressed memory"));
}
