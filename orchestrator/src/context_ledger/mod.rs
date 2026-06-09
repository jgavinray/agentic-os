//! Shadow Context Decision Ledger — module root.
//!
//! This module records deterministic context-pack assembly decisions
//! for offline learned-ranker training. It does not import assembler
//! internals; the assembler imports this module's narrow public interface.
//!
//! Cross-phase invariants from ARCHITECTURE.md apply in full.

pub mod constants;
pub mod enum_hash;
pub mod enums;
pub mod structs;

pub use constants::*;
pub use enums::*;
pub use structs::*;

/// Typed reference to a source record for feature extraction.
///
/// Keeps the ledger from depending on the assembler's internal record types —
/// only bounded fields relevant to feature vectors are exposed.
#[derive(Debug, Clone)]
pub struct SourceRecordRef {
    pub source_type: SourceType,
    pub source_record_id: String,
    pub event_type: Option<EventType>,
    pub estimated_token_cost: i32,
    pub age_seconds: i32,
    pub same_repo: bool,
    pub same_session: bool,
    pub same_trajectory: bool,
    pub context_section: ContextSection,
    pub estimated_token_cost_bucket: TokenCostBucket,
    pub failure_class: Option<FailureClass>,
    pub operational_constraint_type: Option<OperationalConstraintType>,
    pub duplicate_coverage: DuplicateCoverage,
}

/// Decision context passed to feature extraction for determining
/// environment booleans (same_repo, same_session, same_trajectory).
#[derive(Debug, Clone)]
pub struct DecisionContext {
    pub decision_id: String,
    pub repo: String,
    pub session_id: Option<String>,
    pub trajectory_id: Option<String>,
}

/// Extract bounded feature vector from a candidate.
///
/// This is a pure field mapping from the typed source reference into the
/// canonical candidate feature vector. Fields that are not present on
/// `SourceRecordRef` intentionally come from `CandidateFeatureVector::default()`
/// so enum defaults remain centralized in `context_ledger::enums`.
pub fn extract_candidate_features(
    source_record: &SourceRecordRef,
    #[allow(dead_code)] _decision_context: &DecisionContext,
) -> CandidateFeatureVector {
    CandidateFeatureVector {
        age_seconds: source_record.age_seconds.max(0),
        same_repo: source_record.same_repo,
        same_session: source_record.same_session,
        same_trajectory: source_record.same_trajectory,
        estimated_token_cost: source_record.estimated_token_cost.max(0),
        token_cost_bucket: source_record.estimated_token_cost_bucket,
        failure_class: source_record.failure_class,
        operational_constraint_type: source_record.operational_constraint_type,
        context_section: source_record.context_section,
        duplicate_coverage: source_record.duplicate_coverage,
        ..CandidateFeatureVector::default()
    }
}

#[cfg(test)]
mod tests;
