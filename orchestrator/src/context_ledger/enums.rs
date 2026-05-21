/// All closed enums used by the context decision ledger.
///
/// Every enum includes an `unknown` variant per the determinism rules
/// (ARCHITECTURE.md — invariant #1). All enums derive `Default` with the
/// `Unknown` variant so that in-memory structs can derive `Default`.

use serde::{Deserialize, Serialize};

/// Route taken by the request when the context pack was assembled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RequestRoute {
    ChatCompletions,
    Messages,
    #[default]
    Unknown,
}

/// Task label used for context-pack assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TaskType {
    Debug,
    Feature,
    Refactor,
    Architecture,
    Deploy,
    #[default]
    Unknown,
}

/// Cache status for a context-pack assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CacheStatus {
    Miss,
    HitFresh,
    HitStale,
    Bypass,
    #[default]
    Unknown,
}

/// Status of the assembly step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AssemblyStatus {
    Success,
    Partial,
    Failed,
    #[default]
    Unknown,
}

/// Source category for a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SourceType {
    RawEvent,
    Summary,
    FailureHistory,
    OperationalConstraint,
    RetrievalHit,
    FeatureRecord,
    #[default]
    Unknown,
}

/// Event type stored in the candidate record (nullable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EventType {
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
    Error,
    Summary,
    Checkpoint,
    #[default]
    Unknown,
}

/// Summary granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SummaryLevel {
    None,
    Compact,
    Full,
    RolledUp,
    #[default]
    Unknown,
}

/// Token cost bucket variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TokenCostBucket {
    Bucket0_50,
    Bucket51_200,
    Bucket201_500,
    Bucket501_1000,
    Bucket1001_2000,
    Bucket2001Plus,
    #[default]
    Unknown,
}

/// Retrieval score bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RetrievalScoreBucket {
    None,
    Q1,
    Q2,
    Q3,
    Q4,
    #[default]
    Unknown,
}

/// Failure class for a candidate — aligned with real agentic-os detection
/// tags from `feature_extraction`.
///
/// These variants mirror `FEATURE_FAILURE_CLASSES` from `feature_extraction.rs`
/// to prevent vocabulary drift between the ledger and the rest of the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FailureClass {
    ToolLoop,
    UserInterruption,
    MissingAuth,
    WrongEndpoint,
    SummarizationFailure,
    MigrationFailure,
    ContextPackEmpty,
    ContextPackTruncated,
    HighInputTokens,
    SlowUpstreamModel,
    EmptyToolUseMessage,
    AbandonedBeforeModel,
    SingleModelAbandonedNoTools,
    SummarizerSharedUpstream,
    #[default]
    Unknown,
}

/// Operational constraint type — aligned with
/// `OPERATIONAL_CONSTRAINT_TYPES` from `feature_extraction.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OperationalConstraintType {
    UseKnownAuth,
    UseKnownEndpoint,
    UseKnownMigrationFix,
    AvoidToolLoop,
    FixContextRetrieval,
    ReduceContextBloat,
    SeparateSummarizerUpstream,
    HandleUserInterruption,
    HandleSummarizationFailure,
    #[default]
    Unknown,
}

/// Context section a candidate belongs to — aligned with `MemoryLevel` L0-L3
/// sections from `state.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ContextSection {
    RecentEvents,
    SessionSummary,
    RepoSummary,
    ProjectSummary,
    FailureHistory,
    OperationalConstraints,
    #[default]
    Unknown,
}

/// Duplicate coverage for a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DuplicateCoverage {
    None,
    Partial,
    Full,
    #[default]
    Unknown,
}

/// Deterministic score bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DeterministicScoreBucket {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
    #[default]
    Unknown,
}

/// Why a candidate was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DropReason {
    PriorityCap,
    TokenBudget,
    Duplicate,
    Stale,
    LowRelevance,
    SectionDisabled,
    Recovered,
    Superseded,
    Other,
    #[default]
    Unknown,
}

/// Representation chosen for an injected candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RepresentationSelected {
    None,
    Compact,
    Full,
    #[default]
    Unknown,
}

/// Outcome of the trajectory following this decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TrajectoryOutcome {
    Succeeded,
    Unresolved,
    Abandoned,
    Reverted,
    #[default]
    Unknown,
}

/// Validation result after the decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ValidationResult {
    Passed,
    Failed,
    NotRun,
    #[default]
    Unknown,
}

/// Request latency bucket for outcome recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RequestLatencyBucket {
    Bucket0_500ms,
    Bucket500ms1s,
    Bucket1s3s,
    Bucket3s10s,
    Bucket10sPlus,
    #[default]
    Unknown,
}

/// Input token count bucket for outcome recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InputTokenBucket {
    Bucket0_1000,
    Bucket1000_4000,
    Bucket4000_8000,
    Bucket8000_32000,
    Bucket32000Plus,
    #[default]
    Unknown,
}

/// Output token count bucket for outcome recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OutputTokenBucket {
    Bucket0_128,
    Bucket128_512,
    Bucket512_2048,
    Bucket2048_8192,
    Bucket8192Plus,
    #[default]
    Unknown,
}
