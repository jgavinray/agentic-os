use super::test_support::{event, feature_string_values_are_bounded};
use super::*;
use std::collections::HashSet;

#[test]
fn feature_extraction_is_deterministic_for_identical_input() {
    let event = event(
        "e-1",
        "Please explain the Docker compose error",
        Some("ERROR failed to connect to http://localhost:8088"),
    );

    let first = classify_request_event(&event);
    let second = classify_request_event(&event);

    assert_eq!(first, second);
    assert_eq!(first.domain, RequestDomain::Docker);
    assert_eq!(first.intent, RequestIntent::Debug);
}

#[test]
fn feature_keys_are_closed_and_stable() {
    let row = classify_request_event(&event(
        "e-keys",
        "Summarize this Kubernetes log",
        Some("[ERROR] pod failed"),
    ));
    let object = row.features.as_object().expect("features must be object");
    let actual: HashSet<&str> = object.keys().map(String::as_str).collect();
    let expected: HashSet<&str> = FEATURE_KEYS.iter().copied().collect();

    assert_eq!(actual, expected);
}

#[test]
fn features_do_not_copy_raw_text() {
    let raw_secret = "sk-raw-secret-value";
    let row = classify_request_event(&event(
        "e-privacy",
        &format!("Please classify this Authorization: Bearer {raw_secret}"),
        Some("Raw evidence body should not be copied into features"),
    ));
    let serialized = row.features.to_string();

    assert!(!serialized.contains(raw_secret));
    assert!(!serialized.contains("Raw evidence body"));
    assert!(feature_string_values_are_bounded(&row.features));
}

#[test]
fn classifiable_request_event_selection_is_bounded() {
    let user_event = event("e-user", "hello", None);
    assert!(is_classifiable_request_event(&user_event));

    let empty_user_event = event("e-empty-user", "", None);
    assert!(!is_classifiable_request_event(&empty_user_event));

    let whitespace_user_event = event("e-whitespace-user", " \n\t ", None);
    assert!(!is_classifiable_request_event(&whitespace_user_event));

    let evidence_only_event = event("e-evidence-only", "", Some("hello from evidence"));
    assert!(is_classifiable_request_event(&evidence_only_event));

    let mut request_role = event("e-role", "hello", None);
    request_role.event_type = "checkpoint".to_string();
    request_role.event_role = Some("request".to_string());
    assert!(is_classifiable_request_event(&request_role));

    let mut maintenance = event("e-maint", "summary maintenance", None);
    maintenance.event_type = "summary".to_string();
    maintenance.event_role = None;
    assert!(!is_classifiable_request_event(&maintenance));
}

#[test]
fn backfill_sql_does_not_treat_separator_newline_as_request_text() {
    let source = include_str!("request_classification_backfill_queries.rs");

    assert!(source.contains("btrim(coalesce(e.summary, '') || coalesce(e.evidence, '')"));
    assert!(!source.contains("btrim(coalesce(e.summary, '') || E'\\n'"));
}
