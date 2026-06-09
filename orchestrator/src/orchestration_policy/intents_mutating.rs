use crate::orchestration_policy_base::BasePolicy;
use crate::orchestration_policy_types::{
    ContextSource, EditPolicy, GitPolicy, RuntimePolicy, ScopePolicy, ToolCapability,
    ValidationPolicy,
};
use crate::request_classification_types::RequestArtifactType;

pub(crate) fn implement_policy() -> BasePolicy {
    // Implementation requests may inspect the repo and edit/create files. They
    // deliberately do not expose generic shell, publishing, runtime mutation,
    // deployment, or broad mutation surfaces.
    BasePolicy {
        allowed: vec![
            ToolCapability::RepoRead,
            ToolCapability::FileRead,
            ToolCapability::FileEdit,
            ToolCapability::Validation,
            ToolCapability::GitRead,
        ],
        required: vec![],
        blocked: vec![
            ToolCapability::ShellMutation,
            ToolCapability::ShellRead,
            ToolCapability::DockerMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
            ToolCapability::GitWrite,
            ToolCapability::RemoteHostAccess,
        ],
        context: vec![ContextSource::PostgresEvents, ContextSource::QdrantSemantic],
        edit: EditPolicy::ScopedEdit,
        validation: ValidationPolicy::TargetedTests,
        git: GitPolicy::NoGitChanges,
        runtime: RuntimePolicy::NoRestart,
        scope: vec![
            ScopePolicy::CurrentRepoOnly,
            ScopePolicy::IgnoreUnrelatedDirtyChanges,
            ScopePolicy::NoScp,
        ],
    }
}

pub(crate) fn debug_policy(artifact_type: RequestArtifactType) -> BasePolicy {
    // Debugging starts read-heavy because many debug requests are log analysis
    // or diagnosis. Code artifacts are the explicit signal that a single-file
    // edit may be allowed.
    let mut allowed = vec![
        ToolCapability::RepoRead,
        ToolCapability::FileRead,
        ToolCapability::ShellRead,
        ToolCapability::DockerRead,
        ToolCapability::MetricsRead,
        ToolCapability::GitRead,
    ];
    let mut blocked = vec![ToolCapability::Deploy, ToolCapability::RestartService];

    if artifact_type == RequestArtifactType::Code {
        allowed.push(ToolCapability::FileEdit);
    } else {
        blocked.push(ToolCapability::FileEdit);
    }

    let validation = if artifact_type == RequestArtifactType::Logs
        || artifact_type == RequestArtifactType::Code
    {
        ValidationPolicy::TargetedTests
    } else {
        ValidationPolicy::None
    };

    let edit = if artifact_type == RequestArtifactType::Code {
        EditPolicy::SingleFileEdit
    } else {
        EditPolicy::ReadOnly
    };

    BasePolicy {
        allowed,
        required: vec![],
        blocked,
        context: vec![
            ContextSource::PostgresEvents,
            ContextSource::QdrantSemantic,
            ContextSource::ContextLedger,
        ],
        edit,
        validation,
        git: GitPolicy::NoGitChanges,
        runtime: RuntimePolicy::NoRestart,
        scope: vec![ScopePolicy::NoScp],
    }
}

pub(crate) fn modify_config_policy(artifact_type: RequestArtifactType) -> BasePolicy {
    // Config modification is the first policy that allows file editing by
    // default. Runtime mutations still stay blocked because changing a file and
    // restarting a service are separate operational decisions.
    let allowed = vec![
        ToolCapability::RepoRead,
        ToolCapability::FileRead,
        ToolCapability::FileEdit,
        ToolCapability::GitRead,
        ToolCapability::ShellRead,
    ];
    let blocked = vec![
        ToolCapability::Deploy,
        ToolCapability::RestartService,
        ToolCapability::ShellMutation,
    ];

    let edit = if artifact_type == RequestArtifactType::Yaml
        || artifact_type == RequestArtifactType::Json
    {
        EditPolicy::ExplicitFileOnly
    } else {
        EditPolicy::ScopedEdit
    };

    let validation = if artifact_type == RequestArtifactType::Code {
        ValidationPolicy::Build
    } else if artifact_type == RequestArtifactType::Yaml
        || artifact_type == RequestArtifactType::Json
    {
        ValidationPolicy::TargetedTests
    } else {
        ValidationPolicy::None
    };

    BasePolicy {
        allowed,
        required: vec![],
        blocked,
        context: vec![ContextSource::PostgresEvents],
        edit,
        validation,
        git: GitPolicy::CommitAllowed,
        runtime: RuntimePolicy::NoRestart,
        scope: vec![ScopePolicy::NoScp],
    }
}

pub(crate) fn generate_config_policy(artifact_type: RequestArtifactType) -> BasePolicy {
    // Generating config is draft-oriented. It can share the edit/validation
    // posture of ModifyConfig while avoiding an implied commit requirement.
    let mut policy = modify_config_policy(artifact_type);
    policy.git = GitPolicy::NoGitChanges;
    policy
}

pub(crate) fn operate_tool_policy() -> BasePolicy {
    // OperateTool is for runtime operations, not source editing. The client may
    // execute operational tools, but the orchestrator should not also start
    // editing files as part of the same policy.
    BasePolicy {
        allowed: vec![
            ToolCapability::ShellRead,
            ToolCapability::ShellMutation,
            ToolCapability::DockerRead,
            ToolCapability::DockerMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
            ToolCapability::RemoteHostAccess,
        ],
        required: vec![],
        blocked: vec![ToolCapability::FileEdit, ToolCapability::FileRead],
        context: vec![ContextSource::PostgresEvents],
        edit: EditPolicy::ReadOnly,
        validation: ValidationPolicy::None,
        git: GitPolicy::NoGitChanges,
        runtime: RuntimePolicy::DoNotInterruptActiveService,
        scope: vec![ScopePolicy::NoScp],
    }
}
