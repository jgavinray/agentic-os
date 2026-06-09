use super::*;

#[test]
fn build_context_header_contains_repo_and_task() {
    let out = build_context("my-repo", "my-task", &[], &[], &[], 8000);
    assert!(out.starts_with("Repository: my-repo\nTask: my-task\n"));
}

#[test]
fn build_context_empty_shows_no_prior_memory() {
    let out = build_context("r", "t", &[], &[], &[], 8000);
    assert!(out.contains("No relevant prior memory found"));
}

#[test]
fn build_context_hybrid_hit_appears_in_relevant_section() {
    let hits = vec![hit("edit", "changed auth flow")];
    let out = build_context("r", "t", &[], &hits, &[], 8000);
    assert!(out.contains("== Relevant Prior Memory =="));
    assert!(out.contains("changed auth flow"));
}

#[test]
fn build_context_decision_event_appears_in_decisions_section() {
    let recent = vec![mem(
        "decision",
        "chose retry logic",
        Some("503 intermittent"),
    )];
    let out = build_context("r", "t", &recent, &[], &[], 8000);
    assert!(out.contains("== Recent Decisions =="));
    assert!(out.contains("chose retry logic"));
    assert!(out.contains("Evidence: 503 intermittent"));
}

#[test]
fn build_context_non_decision_recent_not_in_decisions_section() {
    let recent = vec![mem("edit", "changed a file", None)];
    let out = build_context("r", "t", &recent, &[], &[], 8000);
    assert!(!out.contains("== Recent Decisions =="));
}

#[test]
fn build_context_error_appears_in_failed_approaches_section() {
    let errs = vec![err_rec("auth_500", "OAuth callback 500", 3)];
    let out = build_context("r", "t", &[], &[], &errs, 8000);
    assert!(out.contains("== Failed Approaches (do not retry) =="));
    assert!(out.contains("OAuth callback 500"));
    assert!(out.contains("seen 3 times"));
}

#[test]
fn build_context_open_questions_from_checkpoint_metadata() {
    let recent = vec![mem_with_meta(
        "checkpoint",
        "end of sprint",
        serde_json::json!({
            "open_questions": ["Is retry threshold right?", "Do we need caching?"]
        }),
    )];
    let out = build_context("r", "t", &recent, &[], &[], 8000);
    assert!(out.contains("== Open Questions =="));
    assert!(out.contains("Is retry threshold right?"));
}

#[test]
fn build_context_empty_sections_omitted() {
    let hits = vec![hit("edit", "some memory")];
    let out = build_context("r", "t", &[], &hits, &[], 8000);
    assert!(!out.contains("== Recent Decisions =="));
    assert!(!out.contains("== Failed Approaches"));
    assert!(!out.contains("== Open Questions =="));
}

#[test]
fn build_context_deduplicates_same_summary_across_sections() {
    let shared = "shared summary text";
    let hits = vec![hit("edit", shared)];
    let recent = vec![mem("decision", shared, None)];
    let out = build_context("r", "t", &recent, &hits, &[], 8000);
    assert_eq!(out.matches(shared).count(), 1);
}

#[test]
fn build_context_truncates_hybrid_section_at_budget() {
    let hits: Vec<SearchHit> = (0..20)
        .map(|i| {
            hit(
                "edit",
                &format!("event {i} with enough text to consume the hybrid budget allocation here"),
            )
        })
        .collect();
    let out = build_context("r", "t", &[], &hits, &[], 500);
    assert!(out.contains("truncated"));
}
