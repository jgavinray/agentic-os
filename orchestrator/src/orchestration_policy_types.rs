use serde::{Deserialize, Serialize};

/// Current orchestration policy schema version.
pub const POLICY_SCHEMA_VERSION: i32 = 1;

/// Source label for deterministic-rule derived policies.
pub const POLICY_SOURCE_DETERMINISTIC_RULES: &str = "deterministic_rules";

/// Sources of context that may be packed into the request.
///
/// These are eligibility labels, not a retrieval result. A source appearing in
/// this list means the context compiler may draw from it for this request; the
/// compiler still owns scoring, deduplication, and budget decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContextSource {
    /// Episodic memory from the Total Recall service.
    TotalRecall,
    /// Durable structured events stored in Postgres.
    PostgresEvents,
    /// Semantic recall from the Qdrant vector index.
    QdrantSemantic,
    /// Stable summaries or artifacts produced by the context compiler.
    CompiledSummaries,
    /// Prior context-selection decisions and candidate features.
    ContextLedger,
    /// Derived features from raw request/response capture, when enabled.
    RawCaptureFeatures,
    /// Fallback label for corrupt, missing, or future values.
    #[default]
    Unknown,
}

impl ContextSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TotalRecall => "total_recall",
            Self::PostgresEvents => "postgres_events",
            Self::QdrantSemantic => "qdrant_semantic",
            Self::CompiledSummaries => "compiled_summaries",
            Self::ContextLedger => "context_ledger",
            Self::RawCaptureFeatures => "raw_capture_features",
            Self::Unknown => "unknown",
        }
    }
}

/// Tool capabilities that may be offered to the model.
///
/// This is the request-level capability vocabulary. `tool_mediation.rs` keeps a
/// smaller local vocabulary for client tool names and shell command shapes, then
/// maps those local capabilities back into this enum when applying policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolCapability {
    /// Current external information lookup.
    WebSearch,
    /// Repository-scoped reading, searching, and listing.
    RepoRead,
    /// Direct file content read.
    FileRead,
    /// File write or patch application.
    FileEdit,
    /// Non-mutating shell execution, such as validation or safe inspection.
    ShellRead,
    /// Mutating or unclassified shell execution.
    ShellMutation,
    /// Docker inspection, logs, status, or compose reads.
    DockerRead,
    /// Docker mutation, including compose up/down/restart actions.
    DockerMutation,
    /// Metrics and telemetry reads.
    MetricsRead,
    /// Git status/log/diff style reads.
    GitRead,
    /// Git commit, push, branch, or PR mutation.
    GitWrite,
    /// Deployment actions.
    Deploy,
    /// Service restart actions.
    RestartService,
    /// SSH or remote host access.
    RemoteHostAccess,
    /// Fallback label for corrupt, missing, or future values.
    #[default]
    Unknown,
}

impl ToolCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WebSearch => "web_search",
            Self::RepoRead => "repo_read",
            Self::FileRead => "file_read",
            Self::FileEdit => "file_edit",
            Self::ShellRead => "shell_read",
            Self::ShellMutation => "shell_mutation",
            Self::DockerRead => "docker_read",
            Self::DockerMutation => "docker_mutation",
            Self::MetricsRead => "metrics_read",
            Self::GitRead => "git_read",
            Self::GitWrite => "git_write",
            Self::Deploy => "deploy",
            Self::RestartService => "restart_service",
            Self::RemoteHostAccess => "remote_host_access",
            Self::Unknown => "unknown",
        }
    }
}

/// How aggressively the model may edit files.
///
/// These labels describe file-edit posture only. They do not by themselves
/// authorize shell mutation, service restart, git writes, or deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EditPolicy {
    /// The request may inspect context but must not edit files.
    ReadOnly,
    /// The request may edit only files explicitly named by the user/policy.
    ExplicitFileOnly,
    /// The request may edit one target file.
    SingleFileEdit,
    /// The request may edit within the current scoped repo/task boundary.
    ScopedEdit,
    /// The request may edit multiple files in the scoped boundary.
    MultiFileEdit,
    /// Fallback label for corrupt, missing, or future values.
    #[default]
    Unknown,
}

impl EditPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read_only",
            Self::ExplicitFileOnly => "explicit_file_only",
            Self::SingleFileEdit => "single_file_edit",
            Self::ScopedEdit => "scoped_edit",
            Self::MultiFileEdit => "multi_file_edit",
            Self::Unknown => "unknown",
        }
    }
}

/// Validation gates before committing changes.
///
/// `None` means "no validation policy applies." `Unknown` means the policy
/// value itself is unclassified or invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ValidationPolicy {
    /// No validation is required by this policy.
    None,
    /// Formatting only, such as `cargo fmt`.
    FormatOnly,
    /// Build or compile validation.
    Build,
    /// Focused tests relevant to the changed behavior.
    TargetedTests,
    /// Full test suite.
    FullTests,
    /// Docker Compose health/readiness validation.
    DockerComposeHealth,
    /// HTTP or API endpoint probe.
    EndpointProbe,
    /// Fallback label for corrupt, missing, or future values.
    #[default]
    Unknown,
}

impl ValidationPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::FormatOnly => "format_only",
            Self::Build => "build",
            Self::TargetedTests => "targeted_tests",
            Self::FullTests => "full_tests",
            Self::DockerComposeHealth => "docker_compose_health",
            Self::EndpointProbe => "endpoint_probe",
            Self::Unknown => "unknown",
        }
    }
}

/// Git change behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GitPolicy {
    /// The request must not create commits or pushes.
    NoGitChanges,
    /// A commit is allowed but not required.
    CommitAllowed,
    /// A commit is required before the work is complete.
    CommitRequired,
    /// A push is allowed but not required.
    PushAllowed,
    /// A push is required before the work is complete.
    PushRequired,
    /// Fallback label for corrupt, missing, or future values.
    #[default]
    Unknown,
}

impl GitPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoGitChanges => "no_git_changes",
            Self::CommitAllowed => "commit_allowed",
            Self::CommitRequired => "commit_required",
            Self::PushAllowed => "push_allowed",
            Self::PushRequired => "push_required",
            Self::Unknown => "unknown",
        }
    }
}

/// Service restart behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RuntimePolicy {
    /// The request must not restart services.
    NoRestart,
    /// A restart is allowed if validation requires it.
    RestartAllowed,
    /// A restart is required for the requested operation.
    RestartRequired,
    /// The request may inspect active services but should not interrupt them.
    DoNotInterruptActiveService,
    /// Remote host access is allowed by policy.
    RemoteHostAllowed,
    /// Fallback label for corrupt, missing, or future values.
    #[default]
    Unknown,
}

impl RuntimePolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoRestart => "no_restart",
            Self::RestartAllowed => "restart_allowed",
            Self::RestartRequired => "restart_required",
            Self::DoNotInterruptActiveService => "do_not_interrupt_active_service",
            Self::RemoteHostAllowed => "remote_host_allowed",
            Self::Unknown => "unknown",
        }
    }
}

/// Scope of files / repos the model may touch.
///
/// Scope policies are additive flags. `NoScp` is always present in derived
/// policies today, even when other scope flags are added later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ScopePolicy {
    /// Stay inside the current repository.
    CurrentRepoOnly,
    /// Stay inside configured workspace roots.
    WorkspaceRootsOnly,
    /// Touch only files named explicitly by the user or policy.
    ExplicitFilesOnly,
    /// Do not alter unrelated dirty worktree changes.
    IgnoreUnrelatedDirtyChanges,
    /// Do not copy files outside the allowed scope.
    NoExternalCopy,
    /// Do not use SCP/file-copy operations across hosts.
    NoScp,
    /// Fallback label for corrupt, missing, or future values.
    #[default]
    Unknown,
}

impl ScopePolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CurrentRepoOnly => "current_repo_only",
            Self::WorkspaceRootsOnly => "workspace_roots_only",
            Self::ExplicitFilesOnly => "explicit_files_only",
            Self::IgnoreUnrelatedDirtyChanges => "ignore_unrelated_dirty_changes",
            Self::NoExternalCopy => "no_external_copy",
            Self::NoScp => "no_scp",
            Self::Unknown => "unknown",
        }
    }
}

/// Prompt / spec refinement behaviour.
///
/// This policy is a workflow mode layered on top of intent. For example, a
/// request can classify as `Explain` while still being handled as a prompt/spec
/// review workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PromptRefinementPolicy {
    /// No prompt/spec refinement policy applies.
    None,
    /// Review only; do not rewrite.
    CritiqueOnly,
    /// Produce one rewrite.
    RewriteOnce,
    /// Review through multiple passes or model collaborations.
    MultiPassReview,
    /// Execute only after explicit approval.
    ExecuteAfterApproval,
    /// Store the accepted prompt/spec shape for later reuse.
    StoreAcceptedPromptPattern,
    /// Fallback label for corrupt, missing, or future values.
    #[default]
    Unknown,
}

impl PromptRefinementPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::CritiqueOnly => "critique_only",
            Self::RewriteOnce => "rewrite_once",
            Self::MultiPassReview => "multi_pass_review",
            Self::ExecuteAfterApproval => "execute_after_approval",
            Self::StoreAcceptedPromptPattern => "store_accepted_prompt_pattern",
            Self::Unknown => "unknown",
        }
    }
}

/// Risk posture.
///
/// `LowRisk` is present for serialization completeness, but
/// `derive_orchestration_policy` represents low risk as an empty `risk_policy`
/// vector. That keeps positive risk signals explicit and avoids persisting
/// redundant "nothing happened" rows for ordinary requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RiskPolicy {
    /// Serialization-only label; not emitted by the deterministic derivation.
    LowRisk,
    /// Destructive behavior requires confirmation or blocking.
    DestructiveRequiresConfirmation,
    /// Secret-bearing capture is allowed because raw capture is enabled.
    SecretCaptureAllowed,
    /// Raw capture is enabled for this environment/request path.
    RawCaptureEnabled,
    /// Current external information is required.
    ExternalWebRequired,
    /// High-stakes guardrails apply.
    HighStakesGuardrail,
    /// Fallback label for corrupt, missing, or future values.
    #[default]
    Unknown,
}

impl RiskPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LowRisk => "low_risk",
            Self::DestructiveRequiresConfirmation => "destructive_requires_confirmation",
            Self::SecretCaptureAllowed => "secret_capture_allowed",
            Self::RawCaptureEnabled => "raw_capture_enabled",
            Self::ExternalWebRequired => "external_web_required",
            Self::HighStakesGuardrail => "high_stakes_guardrail",
            Self::Unknown => "unknown",
        }
    }
}

/// Bounded orchestration policy derived from a request classification.
///
/// `blocked_tools` is authoritative. A capability that appears in
/// `blocked_tools` is removed from both `allowed_tools` and `required_tools`
/// before the policy is returned. That invariant lets downstream callers check
/// either list without needing to resolve conflicts themselves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrchestrationPolicy {
    /// Context systems eligible to contribute to the request context pack.
    pub context_sources: Vec<ContextSource>,
    /// Capabilities the model/client may use if offered by the client.
    pub allowed_tools: Vec<ToolCapability>,
    /// Capabilities required for a correct answer or operation.
    pub required_tools: Vec<ToolCapability>,
    /// Capabilities that must be hidden or denied.
    pub blocked_tools: Vec<ToolCapability>,
    /// File edit posture.
    pub edit_policy: EditPolicy,
    /// Validation expectation.
    pub validation_policy: ValidationPolicy,
    /// Git mutation expectation.
    pub git_policy: GitPolicy,
    /// Runtime/service operation expectation.
    pub runtime_policy: RuntimePolicy,
    /// Scope invariants and flags.
    pub scope_policy: Vec<ScopePolicy>,
    /// Prompt/spec review workflow mode.
    pub prompt_refinement_policy: PromptRefinementPolicy,
    /// Applied risk overlays.
    pub risk_policy: Vec<RiskPolicy>,
}
