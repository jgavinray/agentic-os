use super::*;
use crate::state::{ErrorRecord, SearchHit};
use chrono::Utc;

fn mem(event_type: &str, summary: &str, evidence: Option<&str>) -> crate::state::EventMemory {
    crate::state::EventMemory {
        event_type: event_type.to_string(),
        summary: summary.to_string(),
        evidence: evidence.map(str::to_string),
        metadata: serde_json::json!({}),
        created_at: Utc::now(),
        summary_level: 0,
    }
}

fn mem_with_meta(
    event_type: &str,
    summary: &str,
    metadata: serde_json::Value,
) -> crate::state::EventMemory {
    crate::state::EventMemory {
        event_type: event_type.to_string(),
        summary: summary.to_string(),
        evidence: None,
        metadata,
        created_at: Utc::now(),
        summary_level: 0,
    }
}

fn hit(event_type: &str, summary: &str) -> SearchHit {
    SearchHit {
        event_id: uuid::Uuid::new_v4().to_string(),
        event_type: event_type.to_string(),
        summary: summary.to_string(),
        created_at: Some(Utc::now()),
    }
}

fn err_rec(error_type: &str, description: &str, frequency: i64) -> ErrorRecord {
    ErrorRecord {
        id: "eid".to_string(),
        repo: "r".to_string(),
        task: "t".to_string(),
        error_type: error_type.to_string(),
        description: description.to_string(),
        severity: "medium".to_string(),
        frequency,
        last_seen: Utc::now(),
    }
}

fn event(summary: &str) -> AgentEvent {
    AgentEvent {
        id: "tid".to_string(),
        session_id: "tsession".to_string(),
        repo: "trepo".to_string(),
        actor: "tactor".to_string(),
        event_type: "decision".to_string(),
        summary: summary.to_string(),
        evidence: None,
        metadata: serde_json::json!({}),
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: None,
        created_at: Utc::now(),
        summary_level: 0,
    }
}

fn event_at_level(event_type: &str, summary: &str, level: i32) -> AgentEvent {
    AgentEvent {
        id: uuid::Uuid::new_v4().to_string(),
        session_id: "session".to_string(),
        repo: "repo".to_string(),
        actor: "actor".to_string(),
        event_type: event_type.to_string(),
        summary: summary.to_string(),
        evidence: None,
        metadata: serde_json::json!({}),
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: None,
        created_at: Utc::now(),
        summary_level: level,
    }
}

fn failed_event(summary: &str, evidence: &str, outcome: &str) -> AgentEvent {
    AgentEvent {
        evidence: Some(evidence.to_string()),
        metadata: serde_json::json!({
            "source_event_ids": ["a", "b"],
            "outcome": outcome,
        }),
        ..event_at_level("failed_attempt", summary, 0)
    }
}

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

#[cfg(test)]
#[path = "db_context_tests.rs"]
mod context_tests;

#[test]
fn promoted_memory_requires_source_ids() {
    assert!(!crate::summarizer::has_source_ids(&serde_json::json!({})));
    assert!(!crate::summarizer::has_source_ids(
        &serde_json::json!({"summarized_event_ids": []})
    ));
    assert!(crate::summarizer::has_source_ids(
        &serde_json::json!({"summarized_event_ids": ["e1"]})
    ));
    assert!(crate::summarizer::has_source_ids(
        &serde_json::json!({"source_event_ids": ["e1"]})
    ));
}

#[test]
fn vector_text_contains_all_fields() {
    let mut e = event("summary text");
    e.evidence = Some("evidence text".to_string());
    let t = e.vector_text();
    assert!(t.contains("trepo"));
    assert!(t.contains("decision"));
    assert!(t.contains("summary text"));
    assert!(t.contains("evidence text"));
}

#[test]
fn vector_text_with_no_evidence_does_not_panic() {
    let e = event("summary text");
    let t = e.vector_text();
    assert!(t.contains("summary text"));
}

#[test]
fn payload_contains_required_fields() {
    let e = event("payload summary");
    let p = e.payload();
    assert_eq!(p["event_id"], "tid");
    assert_eq!(p["repo"], "trepo");
    assert_eq!(p["event_type"], "decision");
    assert_eq!(p["summary"], "payload summary");
    assert!(p["created_at"].is_string());
}

#[test]
fn to_memory_copies_fields_correctly() {
    let mut e = event("mem summary");
    e.evidence = Some("proof".to_string());
    let m = e.to_memory();
    assert_eq!(m.event_type, "decision");
    assert_eq!(m.summary, "mem summary");
    assert_eq!(m.evidence, Some("proof".to_string()));
}

#[test]
fn error_record_frequency_field_accessible() {
    let rec = crate::state::ErrorRecord {
        id: "id".to_string(),
        repo: "r".to_string(),
        task: "t".to_string(),
        error_type: "e".to_string(),
        description: "d".to_string(),
        severity: "medium".to_string(),
        frequency: 3,
        last_seen: Utc::now(),
    };
    assert_eq!(rec.frequency, 3);
}

#[test]
fn summary_level_filters_correctly() {
    assert_eq!(preferred_summary_levels(0), vec![0]);
    assert_eq!(preferred_summary_levels(19), vec![0]);
    assert_eq!(preferred_summary_levels(20), vec![1]);
    assert_eq!(preferred_summary_levels(199), vec![1]);
    assert_eq!(preferred_summary_levels(200), vec![2]);
    assert_eq!(preferred_summary_levels(1999), vec![2]);
    assert_eq!(preferred_summary_levels(2000), vec![3, 2]);
}
