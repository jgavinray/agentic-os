/// In-memory structs mirroring the three ledger tables field-for-field,
/// plus the DecisionBundle envelope + candidates composite and the
/// CandidateFeatureVector feature record.
use super::enums::*;

/// Mirror of `context_decision_envelopes`.
#[derive(Debug, Clone)]
pub struct DecisionEnvelope {
    pub decision_id: String,
    pub source_decision_id: Option<String>,
    pub repo: String,
    pub session_id: Option<String>,
    pub trajectory_id: Option<String>,
    pub request_route: RequestRoute,
    pub task_type: TaskType,
    pub created_at_micros: i64,
    pub context_policy_version: String,
    pub feature_schema_version: i32,
    pub total_token_budget: i32,
    pub candidate_tokens_total: i32,
    pub fixed_overhead_tokens: i32,
    pub final_estimated_context_tokens: i32,
    pub cache_status: CacheStatus,
    pub candidate_count: i32,
    pub injected_candidate_count: i32,
    pub assembly_duration_micros: i64,
    pub assembly_status: AssemblyStatus,
}

/// Mirror of `context_decision_candidates`.
#[derive(Debug, Clone)]
pub struct DecisionCandidate {
    pub candidate_id: String,
    pub decision_id: String,
    pub source_type: SourceType,
    pub source_record_id: String,
    pub event_type: Option<EventType>,
    pub summary_level: SummaryLevel,
    pub age_seconds: i32,
    pub same_repo: bool,
    pub same_session: bool,
    pub same_trajectory: bool,
    pub estimated_token_cost: i32,
    pub token_cost_bucket: TokenCostBucket,
    pub retrieval_score_bucket: RetrievalScoreBucket,
    pub failure_class: Option<FailureClass>,
    pub operational_constraint_type: Option<OperationalConstraintType>,
    pub context_section: ContextSection,
    pub duplicate_coverage: DuplicateCoverage,
    pub deterministic_score_bucket: DeterministicScoreBucket,
    pub deterministic_rank: i32,
    pub budget_before_candidate: i32,
    pub budget_after_candidate: i32,
    pub injected: bool,
    pub drop_reason: Option<DropReason>,
    pub drop_reason_other_detail: Option<String>,
    pub representation_selected: RepresentationSelected,
}

/// Mirror of `context_decision_outcomes`.
#[derive(Debug, Clone)]
pub struct DecisionOutcome {
    pub decision_id: String,
    pub trajectory_outcome: TrajectoryOutcome,
    pub user_correction_after_decision: Option<bool>,
    pub retry_after_decision: Option<bool>,
    pub validation_result: ValidationResult,
    pub tool_loop_detected_after_decision: Option<bool>,
    pub request_latency_bucket: RequestLatencyBucket,
    pub input_token_bucket: InputTokenBucket,
    pub output_token_bucket: OutputTokenBucket,
    pub outcome_joined_at_micros: Option<i64>,
    pub outcome_window_closed: bool,
    pub feature_schema_version: i32,
}

/// The unit the writer persists or drops as a whole.
#[derive(Debug, Clone)]
pub struct DecisionBundle {
    pub envelope: DecisionEnvelope,
    pub candidates: Vec<DecisionCandidate>,
}

/// Bounded feature fields extracted from a candidate for offline
/// learned-ranker training. Only contains buckets, booleans, enums
/// and integers — no raw floats.
#[derive(Debug, Clone, Default)]
pub struct CandidateFeatureVector {
    pub age_seconds: i32,
    pub same_repo: bool,
    pub same_session: bool,
    pub same_trajectory: bool,
    pub estimated_token_cost: i32,
    pub token_cost_bucket: TokenCostBucket,
    pub retrieval_score_bucket: RetrievalScoreBucket,
    pub failure_class: Option<FailureClass>,
    pub operational_constraint_type: Option<OperationalConstraintType>,
    pub context_section: ContextSection,
    pub duplicate_coverage: DuplicateCoverage,
    pub deterministic_score_bucket: DeterministicScoreBucket,
    pub deterministic_rank: i32,
    pub injected: bool,
    pub drop_reason: Option<DropReason>,
    pub representation_selected: RepresentationSelected,
}
