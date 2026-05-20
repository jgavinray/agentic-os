/// All closed enums used by the context decision ledger.
///
/// Every enum includes an `unknown` variant per the determinism rules
/// (ARCHITECTURE.md — invariant #1).

use serde::{Deserialize, Serialize};

/// Route taken by the request when the context pack was assembled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestRoute {
    ChatCompletions,
    Messages,
    Unknown,
}

/// Task label used for context-pack assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    Debug,
    Feature,
    Refactor,
    Architecture,
    Deploy,
    Unknown,
}

/// Cache status for a context-pack assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheStatus {
    Miss,
    HitFresh,
    HitStale,
    Bypass,
    Unknown,
}

/// Status of the assembly step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssemblyStatus {
    Success,
    Partial,
    Failed,
    Unknown,
}

/// Source category for a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceType {
    RawEvent,
    Summary,
    FailureHistory,
    OperationalConstraint,
    RetrievalHit,
    FeatureRecord,
    Unknown,
}

/// Event type stored in the candidate record (nullable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
    Error,
    Summary,
    Checkpoint,
    Unknown,
}

/// Summary granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SummaryLevel {
    None,
    Compact,
    Full,
    RolledUp,
    Unknown,
}

/// Token cost bucket variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenCostBucket {
    Bucket0_50,
    Bucket51_200,
    Bucket201_500,
    Bucket501_1000,
    Bucket1001_2000,
    Bucket2001Plus,
    Unknown,
}

/// Retrieval score bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalScoreBucket {
    None,
    Q1,
    Q2,
    Q3,
    Q4,
    Unknown,
}

/// Failure class for a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureClass {
    SyntaxError,
    TypeError,
    ValidationError,
    Timeout,
    ResourceExhausted,
    Unknown,
}

/// Operational constraint type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationalConstraintType {
    TokenBudget,
    LatencyTarget,
    RetryCap,
    SessionLimit,
    Unknown,
}

/// Context section a candidate belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextSection {
    RecentEvents,
    SessionSummary,
    RepoSummary,
    ProjectSummary,
    FailureHistory,
    OperationalConstraints,
    Unknown,
}

/// Duplicate coverage for a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DuplicateCoverage {
    None,
    Partial,
    Full,
    Unknown,
}

/// Deterministic score bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeterministicScoreBucket {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
    Unknown,
}

/// Why a candidate was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    Unknown,
}

/// Representation chosen for an injected candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepresentationSelected {
    None,
    Compact,
    Full,
    Unknown,
}

/// Outcome of the trajectory following this decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrajectoryOutcome {
    Succeeded,
    Unresolved,
    Abandoned,
    Reverted,
    Unknown,
}

/// Validation result after the decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationResult {
    Passed,
    Failed,
    NotRun,
    Unknown,
}

/// Request latency bucket for outcome recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestLatencyBucket {
    Bucket0_500ms,
    Bucket500ms_1s,
    Bucket1s_3s,
    Bucket3s_10s,
    Bucket10sPlus,
    Unknown,
}

/// Input token count bucket for outcome recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputTokenBucket {
    Bucket0_1000,
    Bucket1000_4000,
    Bucket4000_8000,
    Bucket8000_32000,
    Bucket32000Plus,
    Unknown,
}

/// Output token count bucket for outcome recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputTokenBucket {
    Bucket0_128,
    Bucket128_512,
    Bucket512_2048,
    Bucket2048_8192,
    Bucket8192Plus,
    Unknown,
}
