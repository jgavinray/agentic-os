use crate::request_classification::{
    enum_inventory, RecommendedRoute, RequestArtifactType, RequestClassification,
    RequestComplexity, RequestDomain, RequestIntent, RequestRisk, ResponseContract,
    CLASSIFICATION_SCHEMA_VERSION, CLASSIFIER_SOURCE_DETERMINISTIC_RULES, ROUTING_POLICY_VERSION,
};
use chrono::Utc;
use std::collections::HashSet;

const REQUEST_CLASSIFICATION_MIGRATIONS: &str = concat!(
    include_str!("../migrations/V9__request_classifications.sql"),
    "\n",
    include_str!("../migrations/V17__add_implement_request_intent.sql")
);

#[test]
fn enum_inventory_contains_unknown_for_every_enum() {
    for (name, variants) in enum_inventory() {
        assert!(
            variants.contains(&"unknown"),
            "{name} must include unknown variant"
        );
    }
}

#[test]
fn unknown_labels_map_to_unknown() {
    assert_eq!(
        RequestIntent::from_label("not-a-real-intent"),
        RequestIntent::Unknown
    );
    assert_eq!(
        RequestDomain::from_label("not-a-real-domain"),
        RequestDomain::Unknown
    );
    assert_eq!(
        RequestArtifactType::from_label("not-a-real-artifact"),
        RequestArtifactType::Unknown
    );
    assert_eq!(
        RequestComplexity::from_label("not-a-real-complexity"),
        RequestComplexity::Unknown
    );
    assert_eq!(
        RequestRisk::from_label("not-a-real-risk"),
        RequestRisk::Unknown
    );
    assert_eq!(
        RecommendedRoute::from_label("not-a-real-route"),
        RecommendedRoute::Unknown
    );
    assert_eq!(
        ResponseContract::from_label("not-a-real-contract"),
        ResponseContract::Unknown
    );
}

#[test]
fn migration_has_checks_for_all_closed_labels() {
    for (_name, variants) in enum_inventory() {
        for variant in *variants {
            assert!(
                REQUEST_CLASSIFICATION_MIGRATIONS.contains(&format!("'{variant}'")),
                "migration is missing enum label {variant}"
            );
        }
    }
    assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (intent IN"));
    assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (domain IN"));
    assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (artifact_type IN"));
    assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (complexity IN"));
    assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (recommended_route IN"));
    assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (response_contract IN"));
    assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (risk <@ ARRAY"));
    assert!(REQUEST_CLASSIFICATION_MIGRATIONS.contains("CHECK (secondary_domains <@ ARRAY"));
}

#[test]
fn schema_privacy_allows_only_documented_unbounded_text_columns() {
    let allowed_unbounded_strings: HashSet<&str> = [
        "event_id",
        "repo",
        "session_id",
        "routing_policy_version",
        "classifier_source",
    ]
    .into_iter()
    .collect();
    let text_columns = [
        "event_id",
        "repo",
        "session_id",
        "routing_policy_version",
        "classifier_source",
    ];

    for column in text_columns {
        assert!(
            allowed_unbounded_strings.contains(column),
            "{column} must be explicitly privacy-allowlisted"
        );
    }
    assert!(!REQUEST_CLASSIFICATION_MIGRATIONS.contains("summary TEXT"));
    assert!(!REQUEST_CLASSIFICATION_MIGRATIONS.contains("evidence TEXT"));
    assert!(!REQUEST_CLASSIFICATION_MIGRATIONS.contains("request TEXT"));
    assert!(!REQUEST_CLASSIFICATION_MIGRATIONS.contains("prompt TEXT"));
}

#[test]
fn deterministic_default_row_uses_current_versions() {
    let now = Utc::now();
    let row = RequestClassification::deterministic(
        "event-1".to_string(),
        "repo".to_string(),
        "session".to_string(),
        now,
    );

    assert_eq!(
        row.classification_schema_version,
        CLASSIFICATION_SCHEMA_VERSION
    );
    assert_eq!(row.routing_policy_version, ROUTING_POLICY_VERSION);
    assert_eq!(row.classifier_source, CLASSIFIER_SOURCE_DETERMINISTIC_RULES);
    assert_eq!(row.intent, RequestIntent::Unknown);
    assert_eq!(row.recommended_route, RecommendedRoute::Unknown);
}
