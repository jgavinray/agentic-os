use crate::prompt_intervention_detector::{
    detect_prompt_interventions, DETECTOR_VERSION, TAXONOMY_VERSION,
};
use crate::prompt_intervention_extraction::ExtractedPromptSource;
use crate::prompt_intervention_fingerprint::{
    evidence_hash_for_excerpt, PROMPT_FINGERPRINT_VERSION,
};
use crate::prompt_intervention_taxonomy::{
    BurdenType, FailureRelation, InterventionType, SignalFamily, SignalStrength, SourceKind,
};

fn source(text: &str) -> ExtractedPromptSource {
    ExtractedPromptSource {
        source_kind: SourceKind::UserMessage,
        text: text.to_string(),
    }
}

fn types(text: &str) -> Vec<InterventionType> {
    detect_prompt_interventions(&[source(text)])
        .into_iter()
        .map(|finding| finding.intervention_type)
        .collect()
}

#[test]
fn detects_stop_and_redirect_only_when_directional() {
    assert_eq!(
        types("I don't want you to do it, instead review the prompt"),
        vec![InterventionType::StopAndRedirect]
    );
    assert_eq!(
        types("Stop and switch to the other task"),
        vec![InterventionType::StopAndRedirect]
    );
    assert!(types("The stop sign was red").is_empty());
}

#[test]
fn detects_implementation_blocks() {
    assert_eq!(
        types("Do not implement this yet"),
        vec![InterventionType::ImplementationBlock]
    );
    assert_eq!(
        types("Don't deploy this change"),
        vec![InterventionType::ImplementationBlock]
    );
}

#[test]
fn detects_prompt_repair_and_quality_gate() {
    assert_eq!(
        types("Review and refine this prompt"),
        vec![InterventionType::PromptRepair]
    );
    assert_eq!(
        types("This is a blocking issue and not ready to hand off"),
        vec![InterventionType::QualityGate]
    );
}

#[test]
fn detects_scope_narrowing_without_matching_paths_alone() {
    assert_eq!(
        types("Edit only the LiteLLM config"),
        vec![InterventionType::ScopeNarrowing]
    );
    assert!(types("Open /archive/agentic-os/orchestrator/src/lib.rs").is_empty());
}

#[test]
fn detects_validation_requirements_but_not_generated_checklists() {
    assert_eq!(
        types("Run tests before committing"),
        vec![InterventionType::ValidationRequirement]
    );
    assert_eq!(
        types("Acceptance tests required before merge"),
        vec![InterventionType::ValidationRequirement]
    );
    assert!(types("Next steps:\n- run tests\n- deploy").is_empty());
}

#[test]
fn detects_risk_only_when_framed() {
    assert_eq!(
        types("Caution: this production rollout is risky"),
        vec![InterventionType::RiskWarning]
    );
    assert_eq!(
        types("Verify rollback before migration"),
        vec![InterventionType::RiskWarning]
    );
    assert!(types("The production server is running").is_empty());
    assert!(types("Rust has an unsafe keyword").is_empty());
}

#[test]
fn detects_clarification_requests_without_broad_question_matches() {
    assert_eq!(
        types("Which namespace should I use?"),
        vec![InterventionType::ClarificationRequest]
    );
    assert_eq!(
        types("I need the command output before proceeding"),
        vec![InterventionType::ClarificationRequest]
    );
    assert_eq!(
        types("What was decided for the rollout?"),
        vec![InterventionType::ClarificationRequest]
    );
    assert!(types("What is the status?").is_empty());
}

#[test]
fn detects_model_failure_corrections() {
    assert_eq!(
        types("You hallucinated that file"),
        vec![InterventionType::ModelFailureCorrection]
    );
    assert_eq!(
        types("Wrong command, that is not in the repo"),
        vec![InterventionType::ModelFailureCorrection]
    );
}

#[test]
fn emits_multiple_rule_matches_per_source() {
    let detected = detect_prompt_interventions(&[source(
        "Do not implement this. Edit only the config. Run tests.",
    )]);
    let intervention_types = detected
        .iter()
        .map(|finding| finding.intervention_type)
        .collect::<Vec<_>>();
    assert_eq!(
        intervention_types,
        vec![
            InterventionType::ImplementationBlock,
            InterventionType::ScopeNarrowing,
            InterventionType::ValidationRequirement
        ]
    );
}

#[test]
fn populates_hashes_versions_and_taxonomy_fields() {
    let detected = detect_prompt_interventions(&[source("Edit only the config")]);
    let finding = &detected[0];
    assert_eq!(finding.detector_version, DETECTOR_VERSION);
    assert_eq!(finding.taxonomy_version, TAXONOMY_VERSION);
    assert_eq!(
        finding.prompt_fingerprint_version,
        PROMPT_FINGERPRINT_VERSION
    );
    assert_eq!(finding.source_kind, SourceKind::UserMessage);
    assert_eq!(finding.signal_family, SignalFamily::Steering);
    assert_eq!(finding.signal_strength, SignalStrength::Explicit);
    assert_eq!(finding.burden_type, BurdenType::HumanScopeControl);
    assert_eq!(finding.failure_relation, FailureRelation::Prevention);
    assert!(finding.confidence >= 0.9);
    assert!(!finding.evidence_excerpt.is_empty());
    assert_eq!(
        finding.evidence_hash,
        evidence_hash_for_excerpt(&finding.evidence_excerpt)
    );
    assert!(!finding.exact_prompt_hash.is_empty());
    assert!(!finding.normalized_prompt_hash.is_empty());
}

#[test]
fn empty_or_unmatched_sources_return_no_findings() {
    assert!(detect_prompt_interventions(&[]).is_empty());
    assert!(detect_prompt_interventions(&[source("Hello there")]).is_empty());
}
