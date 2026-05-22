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
/// Placeholder — body is `unimplemented!()`. The actual implementation
/// belongs in Phase 3.
///
/// Inputs are typed structs (`SourceRecordRef` + `DecisionContext`) rather
/// than opaque strings so callers cannot pass arbitrary blobs and so the
/// function signature is self-documenting.
pub fn extract_candidate_features(
    _source_record: &SourceRecordRef,
    _decision_context: &DecisionContext,
) -> CandidateFeatureVector {
    unimplemented!("feature extraction body is Phase 3")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::hash::{Hash, Hasher};

    /// Test 1: Schema privacy test.
    ///
    /// Walks the column definitions of all three tables. Fails if any
    /// column is typed as unbounded string except those in the
    /// architecture allowlist. The allowlist is enumerated explicitly.
    #[test]
    fn schema_privacy_test() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum ColumnKind {
            UnboundedString,
            BoundedString,
            ClosedEnum,
            Integer,
            BigInt,
            Boolean,
        }

        // Explicit allowlist per ARCHITECTURE.md privacy rules.
        // Unbounded strings are allowed for these columns:
        // - opaque ids: decision_id, candidate_id, source_record_id
        // - metadata strings: repo, session_id, trajectory_id,
        //   source_decision_id, context_policy_version
        // - bounded free text: drop_reason_other_detail (max 256)
        let allowed_unbounded_strings: HashSet<&str> = [
            // Opaque ids
            "decision_id",
            "candidate_id",
            "source_record_id",
            // Metadata strings
            "repo",
            "session_id",
            "trajectory_id",
            "source_decision_id",
            "context_policy_version",
            // Bounded free text (256 char limit enforced by CHECK)
            "drop_reason_other_detail",
        ]
        .iter()
        .copied()
        .collect();

        // All columns defined in context_decision_envelopes
        // from V7__context_decision_ledger.sql
        let envelope_cols: Vec<(&str, ColumnKind)> = vec![
            ("decision_id", ColumnKind::UnboundedString),
            ("source_decision_id", ColumnKind::UnboundedString),
            ("repo", ColumnKind::UnboundedString),
            ("session_id", ColumnKind::UnboundedString),
            ("trajectory_id", ColumnKind::UnboundedString),
            ("request_route", ColumnKind::ClosedEnum),
            ("task_type", ColumnKind::ClosedEnum),
            ("created_at_micros", ColumnKind::BigInt),
            ("context_policy_version", ColumnKind::UnboundedString),
            ("feature_schema_version", ColumnKind::Integer),
            ("total_token_budget", ColumnKind::Integer),
            ("candidate_tokens_total", ColumnKind::Integer),
            ("fixed_overhead_tokens", ColumnKind::Integer),
            ("final_estimated_context_tokens", ColumnKind::Integer),
            ("cache_status", ColumnKind::ClosedEnum),
            ("candidate_count", ColumnKind::Integer),
            ("injected_candidate_count", ColumnKind::Integer),
            ("assembly_duration_micros", ColumnKind::BigInt),
            ("assembly_status", ColumnKind::ClosedEnum),
        ];

        // All columns in context_decision_candidates
        let candidate_cols: Vec<(&str, ColumnKind)> = vec![
            ("candidate_id", ColumnKind::UnboundedString),
            ("decision_id", ColumnKind::UnboundedString),
            ("source_type", ColumnKind::ClosedEnum),
            ("source_record_id", ColumnKind::UnboundedString),
            ("event_type", ColumnKind::ClosedEnum),
            ("summary_level", ColumnKind::ClosedEnum),
            ("age_seconds", ColumnKind::Integer),
            ("same_repo", ColumnKind::Boolean),
            ("same_session", ColumnKind::Boolean),
            ("same_trajectory", ColumnKind::Boolean),
            ("estimated_token_cost", ColumnKind::Integer),
            ("token_cost_bucket", ColumnKind::ClosedEnum),
            ("retrieval_score_bucket", ColumnKind::ClosedEnum),
            ("failure_class", ColumnKind::ClosedEnum),
            ("operational_constraint_type", ColumnKind::ClosedEnum),
            ("context_section", ColumnKind::ClosedEnum),
            ("duplicate_coverage", ColumnKind::ClosedEnum),
            ("deterministic_score_bucket", ColumnKind::ClosedEnum),
            ("deterministic_rank", ColumnKind::Integer),
            ("budget_before_candidate", ColumnKind::Integer),
            ("budget_after_candidate", ColumnKind::Integer),
            ("injected", ColumnKind::Boolean),
            ("drop_reason", ColumnKind::ClosedEnum),
            ("drop_reason_other_detail", ColumnKind::BoundedString),
            ("representation_selected", ColumnKind::ClosedEnum),
        ];

        // All columns in context_decision_outcomes
        let outcome_cols: Vec<(&str, ColumnKind)> = vec![
            ("decision_id", ColumnKind::UnboundedString),
            ("trajectory_outcome", ColumnKind::ClosedEnum),
            ("user_correction_after_decision", ColumnKind::Boolean),
            ("retry_after_decision", ColumnKind::Boolean),
            ("validation_result", ColumnKind::ClosedEnum),
            ("tool_loop_detected_after_decision", ColumnKind::Boolean),
            ("request_latency_bucket", ColumnKind::ClosedEnum),
            ("input_token_bucket", ColumnKind::ClosedEnum),
            ("output_token_bucket", ColumnKind::ClosedEnum),
            ("outcome_joined_at_micros", ColumnKind::BigInt),
            ("outcome_window_closed", ColumnKind::Boolean),
            ("feature_schema_version", ColumnKind::Integer),
        ];

        fn check_columns(name: &str, cols: &[(&str, ColumnKind)], allowed: &HashSet<&str>) {
            for (col_name, col_kind) in cols {
                let string_kind = matches!(
                    col_kind,
                    ColumnKind::UnboundedString | ColumnKind::BoundedString
                );
                if string_kind && !allowed.contains(*col_name) {
                    panic!(
                        "{}: column '{}' is a string field but is not in the privacy allowlist",
                        name, col_name
                    );
                }
            }
        }

        check_columns("envelopes", &envelope_cols, &allowed_unbounded_strings);
        check_columns("candidates", &candidate_cols, &allowed_unbounded_strings);
        check_columns("outcomes", &outcome_cols, &allowed_unbounded_strings);
    }

    /// Test 2: Enum-version-bump test.
    ///
    /// Hashes the set of enum variants for every enum defined in this
    /// module and compares against the checked-in hash file. If the
    /// hashes differ from the checked-in values, the test fails with a
    /// message instructing the developer to bump
    /// `FEATURE_SCHEMA_VERSION` and update the hash file.
    #[test]
    fn enum_version_bump_test() {
        // Consume the canonical enum inventory from enums.rs. If any
        // enum or variant changes, this test fails until
        // FEATURE_SCHEMA_VERSION and ENUM_VARIANT_HASH are updated.
        let hash_input: String = enums::enum_inventory()
            .iter()
            .map(|(name, variants)| format!("{}:{}", name, variants.join(",")))
            .collect::<Vec<_>>()
            .join("\n");

        // Compare against the checked-in hash file constant.
        let checked_in = enum_hash::ENUM_VARIANT_HASH;

        // Use std::hash for a simple 64-bit fingerprint.
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hash_input.hash(&mut hasher);
        let computed_hash = hasher.finish();

        if checked_in != format!("{computed_hash:x}") {
            panic!(
                "Enum variants changed — bump FEATURE_SCHEMA_VERSION and update context_ledger::enum_hash::ENUM_VARIANT_HASH with the new hash.\nComputed: {computed_hash:x}\nCheckin:  {checked_in}\nEnums:    {hash_input}"
            );
        }
    }
}
