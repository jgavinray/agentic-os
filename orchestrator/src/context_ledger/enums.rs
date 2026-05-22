/// All closed enums used by the context decision ledger.
///
/// Every enum includes an `unknown` variant per the determinism rules
/// (ARCHITECTURE.md — invariant #1). All enums derive `Default` with the
/// `Unknown` variant so that in-memory structs can derive `Default`.
use serde::{Deserialize, Serialize};

macro_rules! ledger_enums {
    (
        $(
            $(#[$enum_meta:meta])*
            pub enum $name:ident {
                $($variant:ident),* $(,)?
            }
        )*
    ) => {
        $(
            $(#[$enum_meta])*
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
            pub enum $name {
                $($variant,)*
                #[default]
                Unknown,
            }
        )*

        /// Canonical inventory of all closed ledger enums and their variants.
        ///
        /// The enum-version-bump test hashes this inventory. The inventory is
        /// generated from the same macro input that defines the enums, so enum
        /// definitions and hash inputs cannot drift independently.
        pub fn enum_inventory() -> &'static [(&'static str, &'static [&'static str])] {
            &[
                $(
                    (
                        stringify!($name),
                        &[$(stringify!($variant),)* "Unknown"],
                    ),
                )*
            ]
        }
    };
}

ledger_enums! {
    /// Route taken by the request when the context pack was assembled.
    pub enum RequestRoute {
        ChatCompletions,
        Messages,
    }

    /// Task label used for context-pack assembly.
    pub enum TaskType {
        Debug,
        Feature,
        Refactor,
        Architecture,
        Deploy,
    }

    /// Cache status for a context-pack assembly.
    pub enum CacheStatus {
        Miss,
        HitFresh,
        HitStale,
        Bypass,
    }

    /// Status of the assembly step.
    pub enum AssemblyStatus {
        Success,
        Partial,
        Failed,
    }

    /// Source category for a candidate.
    pub enum SourceType {
        RawEvent,
        Summary,
        FailureHistory,
        OperationalConstraint,
        RetrievalHit,
        FeatureRecord,
    }

    /// Event type stored in the candidate record (nullable).
    pub enum EventType {
        UserMessage,
        AssistantMessage,
        ToolCall,
        ToolResult,
        Error,
        Summary,
        Checkpoint,
    }

    /// Summary granularity.
    pub enum SummaryLevel {
        None,
        Compact,
        Full,
        RolledUp,
    }

    /// Token cost bucket variant.
    pub enum TokenCostBucket {
        Bucket0_50,
        Bucket51_200,
        Bucket201_500,
        Bucket501_1000,
        Bucket1001_2000,
        Bucket2001Plus,
    }

    /// Retrieval score bucket.
    pub enum RetrievalScoreBucket {
        None,
        Q1,
        Q2,
        Q3,
        Q4,
    }

    /// Failure class for a candidate — aligned with real agentic-os detection
    /// tags from `feature_extraction`.
    ///
    /// These variants mirror `FEATURE_FAILURE_CLASSES` from `feature_extraction.rs`
    /// to prevent vocabulary drift between the ledger and the rest of the system.
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
    }

    /// Operational constraint type — aligned with
    /// `OPERATIONAL_CONSTRAINT_TYPES` from `feature_extraction.rs`.
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
    }

    /// Context section a candidate belongs to — aligned with `MemoryLevel` L0-L3
    /// sections from `state.rs`.
    pub enum ContextSection {
        RecentEvents,
        SessionSummary,
        RepoSummary,
        ProjectSummary,
        FailureHistory,
        OperationalConstraints,
    }

    /// Duplicate coverage for a candidate.
    pub enum DuplicateCoverage {
        None,
        Partial,
        Full,
    }

    /// Deterministic score bucket.
    pub enum DeterministicScoreBucket {
        None,
        VeryLow,
        Low,
        Medium,
        High,
        VeryHigh,
    }

    /// Why a candidate was dropped.
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
    }

    /// Representation chosen for an injected candidate.
    pub enum RepresentationSelected {
        None,
        Compact,
        Full,
    }

    /// Outcome of the trajectory following this decision.
    pub enum TrajectoryOutcome {
        Succeeded,
        Unresolved,
        Abandoned,
        Reverted,
    }

    /// Validation result after the decision.
    pub enum ValidationResult {
        Passed,
        Failed,
        NotRun,
    }

    /// Request latency bucket for outcome recording.
    pub enum RequestLatencyBucket {
        Bucket0_500ms,
        Bucket500ms1s,
        Bucket1s3s,
        Bucket3s10s,
        Bucket10sPlus,
    }

    /// Input token count bucket for outcome recording.
    pub enum InputTokenBucket {
        Bucket0_1000,
        Bucket1000_4000,
        Bucket4000_8000,
        Bucket8000_32000,
        Bucket32000Plus,
    }

    /// Output token count bucket for outcome recording.
    pub enum OutputTokenBucket {
        Bucket0_128,
        Bucket128_512,
        Bucket512_2048,
        Bucket2048_8192,
        Bucket8192Plus,
    }
}
