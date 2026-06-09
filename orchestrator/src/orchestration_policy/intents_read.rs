use crate::orchestration_policy_base::BasePolicy;
use crate::orchestration_policy_intents::mutation_blocklist;
use crate::orchestration_policy_types::{
    ContextSource, EditPolicy, GitPolicy, RuntimePolicy, ScopePolicy, ToolCapability,
    ValidationPolicy,
};

pub(crate) fn explain_policy() -> BasePolicy {
    // Explanation is non-mutating and should stay lightweight by default.
    // Broader durable memory and compiled summaries are reserved for planning,
    // architecture, and other explicitly broad workflows.
    BasePolicy {
        allowed: vec![
            ToolCapability::WebSearch,
            ToolCapability::RepoRead,
            ToolCapability::GitRead,
            ToolCapability::MetricsRead,
        ],
        required: vec![],
        blocked: mutation_blocklist(),
        context: vec![ContextSource::PostgresEvents, ContextSource::QdrantSemantic],
        edit: EditPolicy::ReadOnly,
        validation: ValidationPolicy::None,
        git: GitPolicy::NoGitChanges,
        runtime: RuntimePolicy::NoRestart,
        scope: vec![ScopePolicy::NoScp],
    }
}

pub(crate) fn plan_policy() -> BasePolicy {
    // Planning may use web/repo/git reads, but it should not mutate the repo or
    // runtime. A user must ask for implementation before edit tools appear.
    BasePolicy {
        allowed: vec![
            ToolCapability::WebSearch,
            ToolCapability::RepoRead,
            ToolCapability::GitRead,
        ],
        required: vec![],
        blocked: mutation_blocklist(),
        context: vec![
            ContextSource::TotalRecall,
            ContextSource::PostgresEvents,
            ContextSource::QdrantSemantic,
            ContextSource::CompiledSummaries,
        ],
        edit: EditPolicy::ReadOnly,
        validation: ValidationPolicy::None,
        git: GitPolicy::NoGitChanges,
        runtime: RuntimePolicy::NoRestart,
        scope: vec![ScopePolicy::NoScp],
    }
}

pub(crate) fn summarize_policy() -> BasePolicy {
    // Summarization is read-only and usually local. External-current-info risk
    // can add web search if the request asks for current outside information.
    BasePolicy {
        allowed: vec![ToolCapability::RepoRead, ToolCapability::MetricsRead],
        required: vec![],
        blocked: mutation_blocklist(),
        context: vec![
            ContextSource::TotalRecall,
            ContextSource::PostgresEvents,
            ContextSource::CompiledSummaries,
        ],
        edit: EditPolicy::ReadOnly,
        validation: ValidationPolicy::None,
        git: GitPolicy::NoGitChanges,
        runtime: RuntimePolicy::NoRestart,
        scope: vec![ScopePolicy::NoScp],
    }
}

pub(crate) fn classify_policy() -> BasePolicy {
    // Classification is a meta-operation. It should need only enough repository
    // context to interpret the request, never mutation capabilities.
    BasePolicy {
        allowed: vec![ToolCapability::RepoRead],
        required: vec![],
        blocked: mutation_blocklist(),
        context: vec![ContextSource::PostgresEvents],
        edit: EditPolicy::ReadOnly,
        validation: ValidationPolicy::None,
        git: GitPolicy::NoGitChanges,
        runtime: RuntimePolicy::NoRestart,
        scope: vec![ScopePolicy::NoScp],
    }
}

pub(crate) fn search_policy() -> BasePolicy {
    // Search can combine web and repository reading, but it remains read-only.
    BasePolicy {
        allowed: vec![ToolCapability::WebSearch, ToolCapability::RepoRead],
        required: vec![],
        blocked: mutation_blocklist(),
        context: vec![ContextSource::PostgresEvents, ContextSource::QdrantSemantic],
        edit: EditPolicy::ReadOnly,
        validation: ValidationPolicy::None,
        git: GitPolicy::NoGitChanges,
        runtime: RuntimePolicy::NoRestart,
        scope: vec![ScopePolicy::NoScp],
    }
}

pub(crate) fn unknown_policy() -> BasePolicy {
    // Unknown intent is the safe fallback: no tools, read-only posture, no git,
    // no restart, and mutation capabilities blocked. This prevents classifier
    // failures from expanding access.
    BasePolicy {
        allowed: vec![],
        required: vec![],
        blocked: mutation_blocklist(),
        context: vec![],
        edit: EditPolicy::ReadOnly,
        validation: ValidationPolicy::None,
        git: GitPolicy::NoGitChanges,
        runtime: RuntimePolicy::NoRestart,
        scope: vec![ScopePolicy::NoScp],
    }
}
