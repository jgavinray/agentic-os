use serde::{Deserialize, Serialize};

pub use crate::orchestration_policy_overlay_types::{PromptRefinementPolicy, RiskPolicy};

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
