use super::*;

#[test]
fn extract_candidate_features_maps_source_fields_and_defaults_unmapped_fields() {
    let source = SourceRecordRef {
        source_type: SourceType::FailureHistory,
        source_record_id: "source-1".to_string(),
        event_type: Some(EventType::Error),
        estimated_token_cost: 250,
        age_seconds: 42,
        same_repo: true,
        same_session: true,
        same_trajectory: true,
        context_section: ContextSection::FailureHistory,
        estimated_token_cost_bucket: TokenCostBucket::Bucket201_500,
        failure_class: Some(FailureClass::ToolLoop),
        operational_constraint_type: Some(OperationalConstraintType::AvoidToolLoop),
        duplicate_coverage: DuplicateCoverage::Partial,
    };
    let decision_context = DecisionContext {
        decision_id: "decision-1".to_string(),
        repo: "repo-a".to_string(),
        session_id: Some("session-a".to_string()),
        trajectory_id: Some("trajectory-a".to_string()),
    };

    let features = extract_candidate_features(&source, &decision_context);
    let defaults = CandidateFeatureVector::default();

    assert_eq!(features.age_seconds, 42);
    assert!(features.same_repo);
    assert!(features.same_session);
    assert!(features.same_trajectory);
    assert_eq!(features.estimated_token_cost, 250);
    assert_eq!(features.token_cost_bucket, TokenCostBucket::Bucket201_500);
    assert_eq!(features.failure_class, Some(FailureClass::ToolLoop));
    assert_eq!(
        features.operational_constraint_type,
        Some(OperationalConstraintType::AvoidToolLoop)
    );
    assert_eq!(features.context_section, ContextSection::FailureHistory);
    assert_eq!(features.duplicate_coverage, DuplicateCoverage::Partial);
    assert_eq!(
        features.retrieval_score_bucket,
        defaults.retrieval_score_bucket
    );
    assert_eq!(
        features.deterministic_score_bucket,
        defaults.deterministic_score_bucket
    );
    assert_eq!(features.deterministic_rank, defaults.deterministic_rank);
    assert_eq!(features.injected, defaults.injected);
    assert_eq!(features.drop_reason, defaults.drop_reason);
    assert_eq!(
        features.representation_selected,
        defaults.representation_selected
    );
}

#[test]
fn extract_candidate_features_minimal_source_matches_feature_defaults() {
    let source = SourceRecordRef {
        source_type: SourceType::RawEvent,
        source_record_id: "source-1".to_string(),
        event_type: None,
        estimated_token_cost: 0,
        age_seconds: 0,
        same_repo: false,
        same_session: false,
        same_trajectory: false,
        context_section: ContextSection::Unknown,
        estimated_token_cost_bucket: TokenCostBucket::Unknown,
        failure_class: None,
        operational_constraint_type: None,
        duplicate_coverage: DuplicateCoverage::Unknown,
    };
    let decision_context = DecisionContext {
        decision_id: "decision-1".to_string(),
        repo: "repo-a".to_string(),
        session_id: None,
        trajectory_id: None,
    };

    let features = extract_candidate_features(&source, &decision_context);
    let defaults = CandidateFeatureVector::default();

    assert_eq!(features.age_seconds, defaults.age_seconds);
    assert_eq!(features.same_repo, defaults.same_repo);
    assert_eq!(features.same_session, defaults.same_session);
    assert_eq!(features.same_trajectory, defaults.same_trajectory);
    assert_eq!(features.estimated_token_cost, defaults.estimated_token_cost);
    assert_eq!(features.token_cost_bucket, defaults.token_cost_bucket);
    assert_eq!(features.failure_class, defaults.failure_class);
    assert_eq!(
        features.operational_constraint_type,
        defaults.operational_constraint_type
    );
    assert_eq!(features.context_section, defaults.context_section);
    assert_eq!(features.duplicate_coverage, defaults.duplicate_coverage);
    assert_eq!(
        features.retrieval_score_bucket,
        defaults.retrieval_score_bucket
    );
    assert_eq!(
        features.deterministic_score_bucket,
        defaults.deterministic_score_bucket
    );
    assert_eq!(features.deterministic_rank, defaults.deterministic_rank);
    assert_eq!(features.injected, defaults.injected);
    assert_eq!(features.drop_reason, defaults.drop_reason);
    assert_eq!(
        features.representation_selected,
        defaults.representation_selected
    );
}
