use super::*;

#[test]
fn promoted_memory_requires_source_ids() {
    assert!(!crate::summarizer_promotion::has_source_ids(
        &serde_json::json!({})
    ));
    assert!(!crate::summarizer_promotion::has_source_ids(
        &serde_json::json!({"summarized_event_ids": []})
    ));
    assert!(crate::summarizer_promotion::has_source_ids(
        &serde_json::json!({"summarized_event_ids": ["e1"]})
    ));
    assert!(crate::summarizer_promotion::has_source_ids(
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
