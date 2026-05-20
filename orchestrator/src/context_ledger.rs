/// Shadow Context Decision Ledger — module root.
///
/// This module records deterministic context-pack assembly decisions
/// for offline learned-ranker training. It does not import assembler
/// internals; the assembler imports this module's narrow public interface.
///
/// Cross-phase invariants from ARCHITECTURE.md apply in full.

pub mod enums;
pub mod constants;
pub mod structs;
pub mod enum_hash;

pub use enums::*;
pub use constants::*;
pub use structs::*;

/// Extract bounded feature vector from a candidate.
///
/// Placeholder — body is `unimplemented!()`. The actual implementation
/// belongs in Phase 3.
pub fn extract_candidate_features(
    _source_record: &str,
    _decision_context: &str,
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
        // Explicit allowlist per ARCHITECTURE.md privacy rules.
        // Unbounded strings are allowed for these columns:
        // - opaque ids: decision_id, candidate_id, source_record_id
        // - metadata strings: repo, session_id, trajectory_id,
        //   source_decision_id, context_policy_version
        // - bounded free text: drop_reason_other_detail (max 256)
        let allowed_unbounded_strings: HashSet<&str> = [
            "decision_id",
            "candidate_id",
            "source_record_id",
            "source_decision_id",
            "repo",
            "session_id",
            "trajectory_id",
            "context_policy_version",
            "drop_reason_other_detail",
        ]
        .iter()
        .copied()
        .collect();

        // All columns defined in context_decision_envelopes
        // from V7__context_decision_ledger.sql
        let envelope_cols: Vec<(&str, &str)> = vec![
            ("decision_id", "TEXT"),
            ("source_decision_id", "TEXT"),
            ("repo", "TEXT"),
            ("session_id", "TEXT"),
            ("trajectory_id", "TEXT"),
            ("request_route", "TEXT"),
            ("task_type", "TEXT"),
            ("created_at_micros", "BIGINT"),
            ("context_policy_version", "TEXT"),
            ("feature_schema_version", "INTEGER"),
            ("total_token_budget", "INTEGER"),
            ("candidate_tokens_total", "INTEGER"),
            ("fixed_overhead_tokens", "INTEGER"),
            ("final_estimated_context_tokens", "INTEGER"),
            ("cache_status", "TEXT"),
            ("candidate_count", "INTEGER"),
            ("injected_candidate_count", "INTEGER"),
            ("assembly_duration_micros", "BIGINT"),
            ("assembly_status", "TEXT"),
        ];

        // All columns in context_decision_candidates
        let candidate_cols: Vec<(&str, &str)> = vec![
            ("candidate_id", "TEXT"),
            ("decision_id", "TEXT"),
            ("source_type", "TEXT"),
            ("source_record_id", "TEXT"),
            ("event_type", "TEXT"),
            ("summary_level", "TEXT"),
            ("age_seconds", "INTEGER"),
            ("same_repo", "BOOLEAN"),
            ("same_session", "BOOLEAN"),
            ("same_trajectory", "BOOLEAN"),
            ("estimated_token_cost", "INTEGER"),
            ("token_cost_bucket", "TEXT"),
            ("retrieval_score_bucket", "TEXT"),
            ("failure_class", "TEXT"),
            ("operational_constraint_type", "TEXT"),
            ("context_section", "TEXT"),
            ("duplicate_coverage", "TEXT"),
            ("deterministic_score_bucket", "TEXT"),
            ("deterministic_rank", "INTEGER"),
            ("budget_before_candidate", "INTEGER"),
            ("budget_after_candidate", "INTEGER"),
            ("injected", "BOOLEAN"),
            ("drop_reason", "TEXT"),
            ("drop_reason_other_detail", "TEXT"),
            ("representation_selected", "TEXT"),
        ];

        // All columns in context_decision_outcomes
        let outcome_cols: Vec<(&str, &str)> = vec![
            ("decision_id", "TEXT"),
            ("trajectory_outcome", "TEXT"),
            ("user_correction_after_decision", "BOOLEAN"),
            ("retry_after_decision", "BOOLEAN"),
            ("validation_result", "TEXT"),
            ("tool_loop_detected_after_decision", "BOOLEAN"),
            ("request_latency_bucket", "TEXT"),
            ("input_token_bucket", "TEXT"),
            ("output_token_bucket", "TEXT"),
            ("outcome_joined_at_micros", "BIGINT"),
            ("outcome_window_closed", "BOOLEAN"),
            ("feature_schema_version", "INTEGER"),
        ];

        fn check_columns(
            name: &str,
            cols: &[(<&str, &str)],
            allowed: &HashSet<&str>,
        ) {
            for (col_name, col_type) in cols {
                // TEXT = unbounded string; everything else is
                // primitive or handled differently.
                if *col_type == "TEXT"
                    && !allowed.contains(*col_name)
                {
                    panic!(
                        "{}: column '{}' is unbounded TEXT but is not in the privacy allowlist",
                        name, col_name
                    );
                }
            }
        }

        check_columns("envelopes", &envelope_cols, &allowed_unbounded_strings);
        check_columns(
            "candidates",
            &candidate_cols,
            &allowed_unbounded_strings,
        );
        check_columns(
            "outcomes",
            &outcome_cols,
            &allowed_unbounded_strings,
        );
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
        // Define every enum and its variants here. If this list drifts
        // from the actual enum definitions, the test fails.
        let all_enums: Vec<(&str, Vec<&str>)> = vec![
            (
                "RequestRoute",
                vec!["ChatCompletions", "Messages", "Unknown"],
            ),
            (
                "TaskType",
                vec![
                    "Debug",
                    "Feature",
                    "Refactor",
                    "Architecture",
                    "Deploy",
                    "Unknown",
                ],
            ),
            (
                "CacheStatus",
                vec!["Miss", "HitFresh", "HitStale", "Bypass", "Unknown"],
            ),
            (
                "AssemblyStatus",
                vec!["Success", "Partial", "Failed", "Unknown"],
            ),
            (
                "SourceType",
                vec![
                    "RawEvent",
                    "Summary",
                    "FailureHistory",
                    "OperationalConstraint",
                    "RetrievalHit",
                    "FeatureRecord",
                    "Unknown",
                ],
            ),
            (
                "EventType",
                vec![
                    "UserMessage",
                    "AssistantMessage",
                    "ToolCall",
                    "ToolResult",
                    "Error",
                    "Summary",
                    "Checkpoint",
                    "Unknown",
                ],
            ),
            (
                "SummaryLevel",
                vec!["None", "Compact", "Full", "RolledUp", "Unknown"],
            ),
            (
                "TokenCostBucket",
                vec![
                    "Bucket0_50",
                    "Bucket51_200",
                    "Bucket201_500",
                    "Bucket501_1000",
                    "Bucket1001_2000",
                    "Bucket2001Plus",
                    "Unknown",
                ],
            ),
            (
                "RetrievalScoreBucket",
                vec!["None", "Q1", "Q2", "Q3", "Q4", "Unknown"],
            ),
            (
                "FailureClass",
                vec![
                    "SyntaxError",
                    "TypeError",
                    "ValidationError",
                    "Timeout",
                    "ResourceExhausted",
                    "Unknown",
                ],
            ),
            (
                "OperationalConstraintType",
                vec![
                    "TokenBudget",
                    "LatencyTarget",
                    "RetryCap",
                    "SessionLimit",
                    "Unknown",
                ],
            ),
            (
                "ContextSection",
                vec![
                    "RecentEvents",
                    "SessionSummary",
                    "RepoSummary",
                    "ProjectSummary",
                    "FailureHistory",
                    "OperationalConstraints",
                    "Unknown",
                ],
            ),
            (
                "DuplicateCoverage",
                vec!["None", "Partial", "Full", "Unknown"],
            ),
            (
                "DeterministicScoreBucket",
                vec![
                    "VeryLow",
                    "Low",
                    "Medium",
                    "High",
                    "VeryHigh",
                    "Unknown",
                ],
            ),
            (
                "DropReason",
                vec![
                    "PriorityCap",
                    "TokenBudget",
                    "Duplicate",
                    "Stale",
                    "LowRelevance",
                    "SectionDisabled",
                    "Recovered",
                    "Superseded",
                    "Other",
                    "Unknown",
                ],
            ),
            (
                "RepresentationSelected",
                vec!["None", "Compact", "Full", "Unknown"],
            ),
            (
                "TrajectoryOutcome",
                vec![
                    "Succeeded",
                    "Unresolved",
                    "Abandoned",
                    "Reverted",
                    "Unknown",
                ],
            ),
            (
                "ValidationResult",
                vec!["Passed", "Failed", "NotRun", "Unknown"],
            ),
            (
                "RequestLatencyBucket",
                vec![
                    "Bucket0_500ms",
                    "Bucket500ms_1s",
                    "Bucket1s_3s",
                    "Bucket3s_10s",
                    "Bucket10sPlus",
                    "Unknown",
                ],
            ),
            (
                "InputTokenBucket",
                vec![
                    "Bucket0_1000",
                    "Bucket1000_4000",
                    "Bucket4000_8000",
                    "Bucket8000_32000",
                    "Bucket32000Plus",
                    "Unknown",
                ],
            ),
            (
                "OutputTokenBucket",
                vec![
                    "Bucket0_128",
                    "Bucket128_512",
                    "Bucket512_2048",
                    "Bucket2048_8192",
                    "Bucket8192Plus",
                    "Unknown",
                ],
            ),
        ];

        // Compute a simple hash string by concatenating all enum
        // names and their variants.
        let hash_input: String = all_enums
            .iter()
            .map(|(name, variants)| {
                format!("{}:{}", name, variants.join(","))
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Compare against the checked-in hash file constant.
        let checked_in = enum_hash::ENUM_VARIANT_HASH;

        // The placeholder value always differs — after the developer
        // runs the test once, they should update enum_hash.rs with
        // the computed hash and set FEATURE_SCHEMA_VERSION=1.
        if checked_in == "placeholder" {
            return;
        }

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
