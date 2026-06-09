use serde::{Deserialize, Serialize};

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
    /// Named non-shell validation, such as a test or lint tool.
    Validation,
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
            Self::Validation => "validation",
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
