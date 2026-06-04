//! Deterministic orchestration policy derivation.
//!
//! Request classification answers "what kind of request is this?"  This module
//! answers the next question: "given that classification, what operating
//! envelope should the orchestrator expose?"  The envelope is intentionally
//! broader than tool mediation alone: it covers eligible context sources,
//! allowed/required/blocked tool capabilities, edit scope, validation posture,
//! git behavior, runtime behavior, prompt/spec review workflow, and risk
//! overlays.
//!
//! The derivation function is pure, deterministic, and has no I/O. Persistence
//! is a separate append-only step so policy decisions can be audited without
//! mixing database behavior into the policy rules.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Current orchestration policy schema version.
pub const POLICY_SCHEMA_VERSION: i32 = 1;

/// Source label for deterministic-rule derived policies.
pub const POLICY_SOURCE_DETERMINISTIC_RULES: &str = "deterministic_rules";

// ---------------------------------------------------------------------------
// Bounded enums
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// OrchestrationPolicy
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// derive_orchestration_policy
// ---------------------------------------------------------------------------

use crate::request_classification::{RequestClassification, RequestIntent, RequestRisk};

/// Derive a deterministic orchestration policy from a request classification.
///
/// The derivation has four ordered phases:
///
/// 1. Pick a base policy from `RequestIntent` and `RequestArtifactType`.
/// 2. Apply risk overlays from `RequestRisk` and raw-capture state.
/// 3. Apply the prompt/spec review overlay from bounded request text matching.
/// 4. Normalize conflicts so blocked capabilities win over allowed/required.
///
/// The function is pure: no database reads, no network calls, no randomness.
/// That makes policy rows rebuildable from classification labels, request text,
/// and raw-capture configuration.
pub fn derive_orchestration_policy(
    classification: &RequestClassification,
    request_text: &str,
    raw_capture_enabled: bool,
) -> OrchestrationPolicy {
    let intent = classification.intent;
    let risk = &classification.risk;
    let artifact_type = classification.artifact_type;

    // The base policy is intentionally intent-first. Risk overlays can only
    // reduce or require capabilities after the request's ordinary operating
    // shape has been selected.
    let base = base_policy(intent, artifact_type);

    let mut allowed = base.allowed;
    let mut required = base.required;
    let mut blocked = base.blocked;
    let mut scope = base.scope;
    let mut context_sources = base.context;

    // Risk overlays are additive. They record positive risk posture and append
    // blocked/required capabilities without replacing the intent-derived base.
    let mut risk_policy: Vec<RiskPolicy> = Vec::new();

    if risk.contains(&RequestRisk::ExternalCurrentInfoRequired) {
        // A current-information request must surface web search as required so
        // callers can distinguish "web would be nice" from "web is mandatory".
        if !required.contains(&ToolCapability::WebSearch) {
            required.push(ToolCapability::WebSearch);
        }
        // Keep `required_tools` a subset of `allowed_tools` unless a later
        // blocked-tools overlay removes the same capability.
        if !allowed.contains(&ToolCapability::WebSearch) {
            allowed.push(ToolCapability::WebSearch);
        }
        push_unique(&mut risk_policy, RiskPolicy::ExternalWebRequired);
    }

    if risk.contains(&RequestRisk::HighStakes) {
        push_unique(&mut risk_policy, RiskPolicy::HighStakesGuardrail);
        // High-stakes requests keep read/context behavior available but remove
        // mutation surfaces. The existing live request policy may still refuse
        // or demand a guardrail before this policy reaches a model.
        for cap in &[
            ToolCapability::FileEdit,
            ToolCapability::ShellMutation,
            ToolCapability::DockerMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
            ToolCapability::GitWrite,
        ] {
            push_unique(&mut blocked, *cap);
        }
    }

    if risk.contains(&RequestRisk::DestructiveCommand) {
        push_unique(
            &mut risk_policy,
            RiskPolicy::DestructiveRequiresConfirmation,
        );
        for cap in &[
            ToolCapability::ShellMutation,
            ToolCapability::DockerMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
        ] {
            push_unique(&mut blocked, *cap);
        }
    }

    if raw_capture_enabled {
        push_unique(&mut risk_policy, RiskPolicy::RawCaptureEnabled);
        push_unique(&mut context_sources, ContextSource::RawCaptureFeatures);
    }

    if risk.contains(&RequestRisk::SecretPresent) {
        if raw_capture_enabled {
            push_unique(&mut risk_policy, RiskPolicy::SecretCaptureAllowed);
        }
    }

    // `no_scp` is a policy invariant. If future base policies add other scope
    // modes, this invariant still prevents accidental cross-host file copying
    // unless a separate explicit policy is introduced.
    push_unique(&mut scope, ScopePolicy::NoScp);

    // --- prompt/spec refinement overlay ---
    let mut prompt_refinement = PromptRefinementPolicy::None;
    let mut refined_allowed = allowed.clone();
    let mut refined_edit = base.edit;
    let mut refined_git = base.git;
    let mut refined_runtime = base.runtime;

    let lower = request_text.to_ascii_lowercase();
    let has_prompt_word = contains_any(
        &lower,
        &["prompt", "spec", "task", "deliverable", "constraints"],
    );
    let has_feedback_word = contains_any(
        &lower,
        &["feedback", "review", "rewrite", "refine", "is this good"],
    );

    if has_prompt_word && has_feedback_word {
        prompt_refinement = PromptRefinementPolicy::MultiPassReview;
        // Prompt/spec review is a review workflow, not an implementation
        // request. Narrow the model to repository reading and optional web
        // lookup so the review cannot silently become an edit/tool operation.
        refined_allowed.clear();
        refined_allowed.push(ToolCapability::RepoRead);
        if risk.contains(&RequestRisk::ExternalCurrentInfoRequired) {
            refined_allowed.push(ToolCapability::WebSearch);
        }
        refined_edit = EditPolicy::ReadOnly;
        refined_git = GitPolicy::NoGitChanges;
        refined_runtime = RuntimePolicy::NoRestart;
    }

    // Final normalization makes the "blocked wins" invariant explicit. This is
    // deliberately last so any earlier rule may add allowed/required tools
    // without needing to know all possible risk overlays.
    for blocked_cap in &blocked {
        refined_allowed.retain(|cap| cap != blocked_cap);
        required.retain(|cap| cap != blocked_cap);
    }

    OrchestrationPolicy {
        context_sources,
        allowed_tools: refined_allowed,
        required_tools: required,
        blocked_tools: blocked,
        edit_policy: refined_edit,
        validation_policy: base.validation,
        git_policy: refined_git,
        runtime_policy: refined_runtime,
        scope_policy: scope,
        prompt_refinement_policy: prompt_refinement,
        risk_policy,
    }
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

/// Persist an orchestration policy to the append-only ledger.
///
/// This is a pure INSERT with no ON CONFLICT — every call appends a new row.
pub async fn persist_orchestration_policy(
    pool: &deadpool_postgres::Pool,
    classification: &RequestClassification,
    policy: &OrchestrationPolicy,
) -> Result<uuid::Uuid, anyhow::Error> {
    use chrono::Utc;
    let started = std::time::Instant::now();

    let policy_id = uuid::Uuid::new_v4();

    // event_id: NULL when classification.event_id == "live-request"
    let event_id = if classification.event_id == "live-request" {
        None
    } else {
        Some(classification.event_id.clone())
    };

    let context_sources: serde_json::Value = serde_json::json!(policy
        .context_sources
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>());
    let allowed_tools: serde_json::Value = serde_json::json!(policy
        .allowed_tools
        .iter()
        .map(|t| t.as_str())
        .collect::<Vec<_>>());
    let required_tools: serde_json::Value = serde_json::json!(policy
        .required_tools
        .iter()
        .map(|t| t.as_str())
        .collect::<Vec<_>>());
    let blocked_tools: serde_json::Value = serde_json::json!(policy
        .blocked_tools
        .iter()
        .map(|t| t.as_str())
        .collect::<Vec<_>>());
    let scope_policy: serde_json::Value = serde_json::json!(policy
        .scope_policy
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>());
    let risk_policy: serde_json::Value = serde_json::json!(policy
        .risk_policy
        .iter()
        .map(|r| r.as_str())
        .collect::<Vec<_>>());

    let result = async {
        let conn = pool.get().await?;
        conn.execute(
            "INSERT INTO agent_orchestration_policies (
                policy_id,
                event_id,
                session_id,
                repo,
                created_at,
                classification_schema_version,
                routing_policy_version,
                policy_schema_version,
                intent,
                recommended_route,
                context_sources,
                allowed_tools,
                required_tools,
                blocked_tools,
                edit_policy,
                validation_policy,
                git_policy,
                runtime_policy,
                scope_policy,
                prompt_refinement_policy,
                risk_policy,
                source
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22
            )",
            &[
                &policy_id,
                &event_id,
                &classification.session_id,
                &classification.repo,
                &Utc::now(),
                &classification.classification_schema_version,
                &classification.routing_policy_version,
                &POLICY_SCHEMA_VERSION,
                &classification.intent.as_str(),
                &classification.recommended_route.as_str(),
                &context_sources,
                &allowed_tools,
                &required_tools,
                &blocked_tools,
                &policy.edit_policy.as_str(),
                &policy.validation_policy.as_str(),
                &policy.git_policy.as_str(),
                &policy.runtime_policy.as_str(),
                &scope_policy,
                &policy.prompt_refinement_policy.as_str(),
                &risk_policy,
                &POLICY_SOURCE_DETERMINISTIC_RULES,
            ],
        )
        .await?;
        Ok::<uuid::Uuid, anyhow::Error>(policy_id)
    }
    .await;

    crate::telemetry::record_db_query(
        "persist_orchestration_policy",
        started.elapsed(),
        result.is_ok(),
    );

    result
}

// ---------------------------------------------------------------------------
// compact_policy_metadata
// ---------------------------------------------------------------------------

/// Build a compact JSON representation of an orchestration policy.
///
/// Used for request/tool event metadata and telemetry.
/// The append-only policy ledger still persists the normalized columns separately.
pub fn compact_policy_metadata(
    classification: &RequestClassification,
    policy: &OrchestrationPolicy,
) -> serde_json::Value {
    serde_json::json!({
        "intent": classification.intent.as_str(),
        "recommended_route": classification.recommended_route.as_str(),
        "context_sources": policy.context_sources.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        "allowed_tools": policy.allowed_tools.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
        "required_tools": policy.required_tools.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
        "blocked_tools": policy.blocked_tools.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
        "edit_policy": policy.edit_policy.as_str(),
        "validation_policy": policy.validation_policy.as_str(),
        "git_policy": policy.git_policy.as_str(),
        "runtime_policy": policy.runtime_policy.as_str(),
        "scope_policy": policy.scope_policy.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        "prompt_refinement_policy": policy.prompt_refinement_policy.as_str(),
        "risk_policy": policy.risk_policy.iter().map(|r| r.as_str()).collect::<Vec<_>>(),
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "source": POLICY_SOURCE_DETERMINISTIC_RULES,
    })
}

// ---------------------------------------------------------------------------
// base_policy helpers
// ---------------------------------------------------------------------------

/// Internal struct holding the raw policy pieces for a given intent.
///
/// Fields are intentionally private — consumers only see the public
/// `OrchestrationPolicy` produced by `derive_orchestration_policy`.
#[derive(Debug, Clone)]
struct BasePolicy {
    allowed: Vec<ToolCapability>,
    required: Vec<ToolCapability>,
    blocked: Vec<ToolCapability>,
    context: Vec<ContextSource>,
    edit: EditPolicy,
    validation: ValidationPolicy,
    git: GitPolicy,
    runtime: RuntimePolicy,
    scope: Vec<ScopePolicy>,
}

fn base_policy(
    intent: RequestIntent,
    artifact_type: crate::request_classification::RequestArtifactType,
) -> BasePolicy {
    // Keep this match exhaustive. If a new RequestIntent is added upstream, the
    // compiler should force us to define its operational posture here.
    match intent {
        RequestIntent::Explain => explain_policy(),
        RequestIntent::Debug => debug_policy(artifact_type),
        RequestIntent::ModifyConfig => modify_config_policy(artifact_type),
        RequestIntent::GenerateConfig => generate_config_policy(artifact_type),
        RequestIntent::OperateTool => operate_tool_policy(),
        RequestIntent::Plan => plan_policy(),
        RequestIntent::Summarize => summarize_policy(),
        RequestIntent::Classify => classify_policy(),
        RequestIntent::Search => search_policy(),
        RequestIntent::Unknown => unknown_policy(),
    }
}

fn explain_policy() -> BasePolicy {
    // Explanation may read broad context and metrics, but it is intentionally
    // non-mutating. If an "explain" request needs current information, the
    // risk overlay adds required web search later.
    BasePolicy {
        allowed: vec![
            ToolCapability::WebSearch,
            ToolCapability::RepoRead,
            ToolCapability::GitRead,
            ToolCapability::MetricsRead,
        ],
        required: vec![],
        blocked: vec![
            ToolCapability::FileEdit,
            ToolCapability::ShellMutation,
            ToolCapability::DockerMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
            ToolCapability::GitWrite,
        ],
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

fn debug_policy(artifact_type: crate::request_classification::RequestArtifactType) -> BasePolicy {
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
    // `file_edit` is allowed only when the classifier saw code. Non-code debug
    // requests remain diagnostic and explicitly block file editing.
    if artifact_type == crate::request_classification::RequestArtifactType::Code {
        allowed.push(ToolCapability::FileEdit);
    } else {
        blocked.push(ToolCapability::FileEdit);
    }

    let validation = if artifact_type == crate::request_classification::RequestArtifactType::Logs
        || artifact_type == crate::request_classification::RequestArtifactType::Code
    {
        ValidationPolicy::TargetedTests
    } else {
        ValidationPolicy::None
    };

    let edit = if artifact_type == crate::request_classification::RequestArtifactType::Code {
        EditPolicy::SingleFileEdit
    } else {
        EditPolicy::ReadOnly
    };

    BasePolicy {
        allowed,
        required: vec![],
        blocked,
        context: vec![
            ContextSource::TotalRecall,
            ContextSource::PostgresEvents,
            ContextSource::QdrantSemantic,
            ContextSource::CompiledSummaries,
            ContextSource::ContextLedger,
        ],
        edit,
        validation,
        git: GitPolicy::NoGitChanges,
        runtime: RuntimePolicy::NoRestart,
        scope: vec![ScopePolicy::NoScp],
    }
}

fn modify_config_policy(
    artifact_type: crate::request_classification::RequestArtifactType,
) -> BasePolicy {
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

    let edit = if artifact_type == crate::request_classification::RequestArtifactType::Yaml
        || artifact_type == crate::request_classification::RequestArtifactType::Json
    {
        EditPolicy::ExplicitFileOnly
    } else {
        EditPolicy::ScopedEdit
    };

    let validation = if artifact_type == crate::request_classification::RequestArtifactType::Code {
        ValidationPolicy::Build
    } else if artifact_type == crate::request_classification::RequestArtifactType::Yaml
        || artifact_type == crate::request_classification::RequestArtifactType::Json
    {
        ValidationPolicy::TargetedTests
    } else {
        ValidationPolicy::None
    };

    BasePolicy {
        allowed,
        required: vec![],
        blocked,
        context: vec![
            ContextSource::TotalRecall,
            ContextSource::PostgresEvents,
            ContextSource::CompiledSummaries,
        ],
        edit,
        validation,
        git: GitPolicy::CommitAllowed,
        runtime: RuntimePolicy::NoRestart,
        scope: vec![ScopePolicy::NoScp],
    }
}

fn generate_config_policy(
    artifact_type: crate::request_classification::RequestArtifactType,
) -> BasePolicy {
    // Generating config is draft-oriented. It can share the edit/validation
    // posture of ModifyConfig while avoiding an implied commit requirement.
    let mut t = modify_config_policy(artifact_type);
    t.git = GitPolicy::NoGitChanges;
    t
}

fn operate_tool_policy() -> BasePolicy {
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
        context: vec![ContextSource::TotalRecall, ContextSource::PostgresEvents],
        edit: EditPolicy::ReadOnly,
        validation: ValidationPolicy::None,
        git: GitPolicy::NoGitChanges,
        runtime: RuntimePolicy::DoNotInterruptActiveService,
        scope: vec![ScopePolicy::NoScp],
    }
}

fn plan_policy() -> BasePolicy {
    // Planning may use web/repo/git reads, but it should not mutate the repo or
    // runtime. A user must ask for implementation before edit tools appear.
    BasePolicy {
        allowed: vec![
            ToolCapability::WebSearch,
            ToolCapability::RepoRead,
            ToolCapability::GitRead,
        ],
        required: vec![],
        blocked: vec![
            ToolCapability::FileEdit,
            ToolCapability::ShellMutation,
            ToolCapability::DockerMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
            ToolCapability::GitWrite,
        ],
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

fn summarize_policy() -> BasePolicy {
    // Summarization is read-only and usually local. External-current-info risk
    // can add web search if the request asks for current outside information.
    BasePolicy {
        allowed: vec![ToolCapability::RepoRead, ToolCapability::MetricsRead],
        required: vec![],
        blocked: vec![
            ToolCapability::FileEdit,
            ToolCapability::ShellMutation,
            ToolCapability::DockerMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
            ToolCapability::GitWrite,
        ],
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

fn classify_policy() -> BasePolicy {
    // Classification is a meta-operation. It should need only enough repository
    // context to interpret the request, never mutation capabilities.
    BasePolicy {
        allowed: vec![ToolCapability::RepoRead],
        required: vec![],
        blocked: vec![
            ToolCapability::FileEdit,
            ToolCapability::ShellMutation,
            ToolCapability::DockerMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
            ToolCapability::GitWrite,
        ],
        context: vec![ContextSource::TotalRecall, ContextSource::PostgresEvents],
        edit: EditPolicy::ReadOnly,
        validation: ValidationPolicy::None,
        git: GitPolicy::NoGitChanges,
        runtime: RuntimePolicy::NoRestart,
        scope: vec![ScopePolicy::NoScp],
    }
}

fn search_policy() -> BasePolicy {
    // Search can combine web and repository reading, but it remains read-only.
    BasePolicy {
        allowed: vec![ToolCapability::WebSearch, ToolCapability::RepoRead],
        required: vec![],
        blocked: vec![
            ToolCapability::FileEdit,
            ToolCapability::ShellMutation,
            ToolCapability::DockerMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
            ToolCapability::GitWrite,
        ],
        context: vec![
            ContextSource::TotalRecall,
            ContextSource::PostgresEvents,
            ContextSource::QdrantSemantic,
        ],
        edit: EditPolicy::ReadOnly,
        validation: ValidationPolicy::None,
        git: GitPolicy::NoGitChanges,
        runtime: RuntimePolicy::NoRestart,
        scope: vec![ScopePolicy::NoScp],
    }
}

fn unknown_policy() -> BasePolicy {
    // Unknown intent is the safe fallback: no tools, read-only posture, no git,
    // no restart, and mutation capabilities blocked. This prevents classifier
    // failures from expanding access.
    BasePolicy {
        allowed: vec![],
        required: vec![],
        blocked: vec![
            ToolCapability::FileEdit,
            ToolCapability::ShellMutation,
            ToolCapability::DockerMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
            ToolCapability::GitWrite,
        ],
        context: vec![],
        edit: EditPolicy::ReadOnly,
        validation: ValidationPolicy::None,
        git: GitPolicy::NoGitChanges,
        runtime: RuntimePolicy::NoRestart,
        scope: vec![ScopePolicy::NoScp],
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Push an item into a vec only if it is not already present.
///
/// Used by the risk-overlay logic to prevent `risk_policy` and
/// `blocked_tools` from accumulating duplicates as more rules are added.
fn push_unique<T: PartialEq>(vec: &mut Vec<T>, item: T) {
    if !vec.contains(&item) {
        vec.push(item);
    }
}

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_classification::{
        RequestArtifactType, RequestClassification, RequestIntent, RequestRisk,
    };
    use chrono::Utc;

    fn classification(
        intent: RequestIntent,
        risk: Vec<RequestRisk>,
        artifact_type: RequestArtifactType,
    ) -> RequestClassification {
        RequestClassification {
            event_id: "test-event".to_string(),
            repo: "test-repo".to_string(),
            session_id: "test-session".to_string(),
            trajectory_id: None,
            event_created_at: Utc::now(),
            classified_at: Utc::now(),
            classification_schema_version: 1,
            routing_policy_version: "deterministic-v1".to_string(),
            classifier_source: "deterministic_rules".to_string(),
            intent,
            domain: crate::request_classification::RequestDomain::Generic,
            secondary_domains: vec![],
            artifact_type,
            risk,
            complexity: crate::request_classification::RequestComplexity::L1Simple,
            recommended_route: crate::request_classification::RecommendedRoute::SmallLocalModel,
            response_contract: crate::request_classification::ResponseContract::DirectAnswer,
            features: serde_json::json!({}),
        }
    }

    // 1. Explain allows web_search, read_only, no_git_changes
    #[test]
    fn test_explain_allows_web_search_read_only_no_git() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::None],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this", false);

        assert!(policy.allowed_tools.contains(&ToolCapability::WebSearch));
        assert_eq!(policy.edit_policy, EditPolicy::ReadOnly);
        assert_eq!(policy.git_policy, GitPolicy::NoGitChanges);
    }

    // 2. Explain + ExternalCurrentInfoRequired requires web_search
    #[test]
    fn test_explain_external_info_requires_web_search() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::ExternalCurrentInfoRequired],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this", false);

        assert!(policy.required_tools.contains(&ToolCapability::WebSearch));
    }

    // 3. Debug Code => single_file_edit, file_edit allowed, targeted_tests
    #[test]
    fn test_debug_code_single_file_edit() {
        let c = classification(
            RequestIntent::Debug,
            vec![RequestRisk::None],
            RequestArtifactType::Code,
        );
        let policy = derive_orchestration_policy(&c, "fix the bug in main.rs", false);

        assert_eq!(policy.edit_policy, EditPolicy::SingleFileEdit);
        assert!(policy.allowed_tools.contains(&ToolCapability::FileEdit));
        assert_eq!(policy.validation_policy, ValidationPolicy::TargetedTests);
    }

    // 4. ModifyConfig Code => build
    #[test]
    fn test_modify_config_code_build() {
        let c = classification(
            RequestIntent::ModifyConfig,
            vec![RequestRisk::None],
            RequestArtifactType::Code,
        );
        let policy = derive_orchestration_policy(&c, "edit the config", false);

        assert_eq!(policy.validation_policy, ValidationPolicy::Build);
    }

    // 5. ModifyConfig Yaml => targeted_tests
    #[test]
    fn test_modify_config_yaml_targeted_tests() {
        let c = classification(
            RequestIntent::ModifyConfig,
            vec![RequestRisk::None],
            RequestArtifactType::Yaml,
        );
        let policy = derive_orchestration_policy(&c, "edit the yaml config", false);

        assert_eq!(policy.validation_policy, ValidationPolicy::TargetedTests);
    }

    // 5b. ModifyConfig Json => targeted_tests
    #[test]
    fn test_modify_config_json_targeted_tests() {
        let c = classification(
            RequestIntent::ModifyConfig,
            vec![RequestRisk::None],
            RequestArtifactType::Json,
        );
        let policy = derive_orchestration_policy(&c, "edit the json config", false);

        assert_eq!(policy.validation_policy, ValidationPolicy::TargetedTests);
    }

    // 6. DestructiveCommand blocks shell_mutation, deploy, restart_service
    #[test]
    fn test_destructive_command_blocks_mutation() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::DestructiveCommand],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "run rm -rf /tmp", false);

        assert!(policy
            .blocked_tools
            .contains(&ToolCapability::ShellMutation));
        assert!(policy.blocked_tools.contains(&ToolCapability::Deploy));
        assert!(policy
            .blocked_tools
            .contains(&ToolCapability::RestartService));
    }

    // 7. HighStakes adds high_stakes_guardrail and blocks mutations
    #[test]
    fn test_high_stakes_guardrail_and_blocks() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::HighStakes],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this medical issue", false);

        assert!(policy
            .risk_policy
            .contains(&RiskPolicy::HighStakesGuardrail));
        assert!(policy.blocked_tools.contains(&ToolCapability::FileEdit));
        assert!(policy
            .blocked_tools
            .contains(&ToolCapability::ShellMutation));
        assert!(policy
            .blocked_tools
            .contains(&ToolCapability::DockerMutation));
        assert!(policy.blocked_tools.contains(&ToolCapability::Deploy));
        assert!(policy
            .blocked_tools
            .contains(&ToolCapability::RestartService));
        assert!(policy.blocked_tools.contains(&ToolCapability::GitWrite));
    }

    // 8. Unknown minimal read-only empty allowed_tools
    #[test]
    fn test_unknown_minimal() {
        let c = classification(
            RequestIntent::Unknown,
            vec![RequestRisk::None],
            RequestArtifactType::Unknown,
        );
        let policy = derive_orchestration_policy(&c, "???", false);

        assert!(policy.allowed_tools.is_empty());
        assert_eq!(policy.edit_policy, EditPolicy::ReadOnly);
    }

    // 9. NoScp always present
    #[test]
    fn test_no_scp_always_present() {
        for intent in [
            RequestIntent::Explain,
            RequestIntent::Debug,
            RequestIntent::ModifyConfig,
            RequestIntent::GenerateConfig,
            RequestIntent::OperateTool,
            RequestIntent::Plan,
            RequestIntent::Summarize,
            RequestIntent::Classify,
            RequestIntent::Search,
            RequestIntent::Unknown,
        ] {
            let c = classification(
                intent,
                vec![RequestRisk::None],
                RequestArtifactType::PlainText,
            );
            let policy = derive_orchestration_policy(&c, "test", false);
            assert!(
                policy.scope_policy.contains(&ScopePolicy::NoScp),
                "{intent:?} must include no_scp"
            );
        }
    }

    // 10. serde round-trip policy
    #[test]
    fn test_serde_round_trip() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::None],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this", false);

        let serialized = serde_json::to_string(&policy).expect("serialize");
        let deserialized: OrchestrationPolicy =
            serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(policy, deserialized);
    }

    // 11. prompt/spec review => multi_pass_review, read_only, narrowed tools
    #[test]
    fn test_prompt_review_overlay() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::None],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(
            &c,
            "Review the prompt spec and rewrite it — is this good?",
            false,
        );

        assert_eq!(
            policy.prompt_refinement_policy,
            PromptRefinementPolicy::MultiPassReview
        );
        assert_eq!(policy.edit_policy, EditPolicy::ReadOnly);
        assert_eq!(policy.git_policy, GitPolicy::NoGitChanges);
        assert_eq!(policy.runtime_policy, RuntimePolicy::NoRestart);
        // narrowed to repo_read only (no ExternalCurrentInfoRequired)
        assert!(policy.allowed_tools.contains(&ToolCapability::RepoRead));
        assert!(
            !policy.allowed_tools.contains(&ToolCapability::WebSearch),
            "web_search should not be present without external_web_required"
        );
    }

    #[test]
    fn test_prompt_review_overlay_with_external_info() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::ExternalCurrentInfoRequired],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(
            &c,
            "Review the prompt spec and rewrite it — is this good?",
            false,
        );

        assert_eq!(
            policy.prompt_refinement_policy,
            PromptRefinementPolicy::MultiPassReview
        );
        // web_search should be present because ExternalCurrentInfoRequired adds it
        assert!(policy.allowed_tools.contains(&ToolCapability::WebSearch));
    }

    // -----------------------------------------------------------------------
    // Phase 2 — persistence & migration tests
    // -----------------------------------------------------------------------

    // 1. vector label serialization produces snake_case JSON arrays
    #[test]
    fn test_vector_label_serialization_snake_case() {
        let c = classification(
            RequestIntent::Debug,
            vec![RequestRisk::HighStakes, RequestRisk::DestructiveCommand],
            RequestArtifactType::Code,
        );
        let policy = derive_orchestration_policy(&c, "fix the bug", false);

        // context_sources: all snake_case
        let ctx_json: Vec<&str> = policy.context_sources.iter().map(|s| s.as_str()).collect();
        assert!(ctx_json
            .iter()
            .all(|s| s.chars().all(|c| c.is_lowercase() || c == '_')));

        // allowed_tools: snake_case
        let allowed_json: Vec<&str> = policy.allowed_tools.iter().map(|t| t.as_str()).collect();
        assert!(allowed_json
            .iter()
            .all(|s| s.chars().all(|c| c.is_lowercase() || c == '_')));

        // risk_policy: snake_case
        let risk_json: Vec<&str> = policy.risk_policy.iter().map(|r| r.as_str()).collect();
        assert!(risk_json
            .iter()
            .all(|s| s.chars().all(|c| c.is_lowercase() || c == '_')));

        // scope_policy: snake_case
        let scope_json: Vec<&str> = policy.scope_policy.iter().map(|s| s.as_str()).collect();
        assert!(scope_json
            .iter()
            .all(|s| s.chars().all(|c| c.is_lowercase() || c == '_')));

        // Verify actual labels match expected snake_case values
        assert!(allowed_json.contains(&"file_read"));
        assert!(allowed_json.contains(&"repo_read"));
        assert!(risk_json.contains(&"high_stakes_guardrail"));
        assert!(risk_json.contains(&"destructive_requires_confirmation"));
        assert!(scope_json.contains(&"no_scp"));
    }

    // 2. scalar as_str labels remain snake_case
    #[test]
    fn test_scalar_as_str_labels_snake_case() {
        let c = classification(
            RequestIntent::Debug,
            vec![RequestRisk::None],
            RequestArtifactType::Code,
        );
        let policy = derive_orchestration_policy(&c, "fix the bug", false);

        // edit_policy
        let edit = policy.edit_policy.as_str();
        assert_eq!(edit, "single_file_edit");

        // validation_policy
        let val = policy.validation_policy.as_str();
        assert_eq!(val, "targeted_tests");

        // git_policy
        let git = policy.git_policy.as_str();
        assert_eq!(git, "no_git_changes");

        // runtime_policy
        let rt = policy.runtime_policy.as_str();
        assert_eq!(rt, "no_restart");

        // prompt_refinement_policy
        let pr = policy.prompt_refinement_policy.as_str();
        assert_eq!(pr, "none");
    }

    // 3. migration file contains expected columns and structure
    #[test]
    fn test_migration_has_expected_structure() {
        let migration = include_str!("../migrations/V16__orchestration_policies.sql");

        // Table name
        assert!(
            migration.contains("agent_orchestration_policies"),
            "migration must reference agent_orchestration_policies"
        );

        // Key columns
        assert!(
            migration.contains("policy_schema_version"),
            "must have policy_schema_version"
        );
        assert!(
            migration.contains("prompt_refinement_policy"),
            "must have prompt_refinement_policy"
        );
        assert!(
            migration.contains("context_sources"),
            "must have context_sources"
        );
        assert!(
            migration.contains("allowed_tools"),
            "must have allowed_tools"
        );
        assert!(
            migration.contains("required_tools"),
            "must have required_tools"
        );
        assert!(
            migration.contains("blocked_tools"),
            "must have blocked_tools"
        );
        assert!(migration.contains("risk_policy"), "must have risk_policy");
        assert!(migration.contains("scope_policy"), "must have scope_policy");

        // JSONB columns
        assert!(
            migration.contains("JSONB"),
            "must use JSONB for array columns"
        );

        // Indexes
        assert!(
            migration.contains("session_id"),
            "must have session_id index"
        );
        assert!(migration.contains("repo"), "must have repo index");
        assert!(migration.contains("event_id"), "must have event_id index");
        assert!(migration.contains("intent"), "must have intent index");
        assert!(
            migration.contains("prompt_refinement_policy"),
            "must have prompt_refinement index"
        );
    }

    // 4. compact_policy_metadata produces correct JSON shape
    #[test]
    fn test_compact_policy_metadata() {
        let c = classification(
            RequestIntent::Debug,
            vec![RequestRisk::None],
            RequestArtifactType::Code,
        );
        let policy = derive_orchestration_policy(&c, "fix the bug", false);

        let meta = compact_policy_metadata(&c, &policy);

        // scalar fields
        assert!(meta["intent"].is_string());
        assert!(meta["recommended_route"].is_string());
        assert!(meta["edit_policy"].is_string());
        assert!(meta["validation_policy"].is_string());
        assert!(meta["git_policy"].is_string());
        assert!(meta["runtime_policy"].is_string());
        assert!(meta["prompt_refinement_policy"].is_string());

        // array fields
        assert!(meta["context_sources"].is_array());
        assert!(meta["allowed_tools"].is_array());
        assert!(meta["required_tools"].is_array());
        assert!(meta["blocked_tools"].is_array());
        assert!(meta["scope_policy"].is_array());
        assert!(meta["risk_policy"].is_array());

        // schema version & source
        assert_eq!(meta["policy_schema_version"], 1);
        assert_eq!(meta["source"], "deterministic_rules");

        // snake_case labels
        let allowed: Vec<&str> = meta["allowed_tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(allowed.contains(&"repo_read"));

        let scope: Vec<&str> = meta["scope_policy"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(scope.contains(&"no_scp"));
    }

    // -----------------------------------------------------------------------
    // Phase 3 — blocked_tools authority & required⊆allowed invariants
    // -----------------------------------------------------------------------

    // 1. Summarize + ExternalCurrentInfoRequired => web_search is both allowed and required.
    #[test]
    fn test_summarize_external_info_requires_and_allows_web_search() {
        let c = classification(
            RequestIntent::Summarize,
            vec![RequestRisk::ExternalCurrentInfoRequired],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "summarize the latest release", false);

        // web_search must be in allowed_tools
        assert!(
            policy.allowed_tools.contains(&ToolCapability::WebSearch),
            "web_search must be allowed for Summarize+ExternalCurrentInfoRequired"
        );
        // web_search must be in required_tools
        assert!(
            policy.required_tools.contains(&ToolCapability::WebSearch),
            "web_search must be required for Summarize+ExternalCurrentInfoRequired"
        );
    }

    // 2. Debug + Code + HighStakes => file_edit is blocked and not allowed.
    #[test]
    fn test_debug_code_high_stakes_blocks_file_edit() {
        let c = classification(
            RequestIntent::Debug,
            vec![RequestRisk::HighStakes],
            RequestArtifactType::Code,
        );
        let policy = derive_orchestration_policy(&c, "fix the bug in main.rs", false);

        // file_edit must be blocked
        assert!(
            policy.blocked_tools.contains(&ToolCapability::FileEdit),
            "file_edit must be blocked under HighStakes"
        );
        // file_edit must NOT be in allowed_tools
        assert!(
            !policy.allowed_tools.contains(&ToolCapability::FileEdit),
            "file_edit must not be allowed when blocked"
        );
        // file_edit must NOT be in required_tools
        assert!(
            !policy.required_tools.contains(&ToolCapability::FileEdit),
            "file_edit must not be required when blocked"
        );
    }

    // 3. DestructiveCommand on OperateTool => shell_mutation/deploy/restart_service blocked and absent.
    #[test]
    fn test_destructive_command_operate_tool_blocks_mutation_tools() {
        let c = classification(
            RequestIntent::OperateTool,
            vec![RequestRisk::DestructiveCommand],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "run rm -rf /tmp", false);

        let blocked_caps = [
            ToolCapability::ShellMutation,
            ToolCapability::Deploy,
            ToolCapability::RestartService,
        ];

        for cap in &blocked_caps {
            assert!(
                policy.blocked_tools.contains(cap),
                "{cap:?} must be blocked under DestructiveCommand"
            );
            assert!(
                !policy.allowed_tools.contains(cap),
                "{cap:?} must not be allowed when blocked"
            );
            assert!(
                !policy.required_tools.contains(cap),
                "{cap:?} must not be required when blocked"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Phase 4 — raw-capture state in deterministic policy
    // -----------------------------------------------------------------------

    // 1. raw_capture_enabled true with RequestRisk::None emits RawCaptureEnabled
    //    and does not emit LowRisk.
    #[test]
    fn test_raw_capture_enabled_true_emits_raw_capture_enabled_no_low_risk() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::None],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this", true);

        assert!(
            policy.risk_policy.contains(&RiskPolicy::RawCaptureEnabled),
            "raw_capture_enabled=true must emit RawCaptureEnabled"
        );
        assert!(
            !policy.risk_policy.contains(&RiskPolicy::LowRisk),
            "LowRisk must never be emitted by derive_orchestration_policy"
        );
    }

    // 2. SecretPresent with raw_capture_enabled true emits both
    //    RawCaptureEnabled and SecretCaptureAllowed.
    #[test]
    fn test_secret_present_raw_capture_emits_both() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::SecretPresent],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "show me the secrets", true);

        assert!(
            policy.risk_policy.contains(&RiskPolicy::RawCaptureEnabled),
            "SecretPresent+raw_capture must emit RawCaptureEnabled"
        );
        assert!(
            policy
                .risk_policy
                .contains(&RiskPolicy::SecretCaptureAllowed),
            "SecretPresent+raw_capture must emit SecretCaptureAllowed"
        );
        // RawCaptureEnabled should appear exactly once (from the unconditional block above)
        assert_eq!(
            policy
                .risk_policy
                .iter()
                .filter(|&&r| r == RiskPolicy::RawCaptureEnabled)
                .count(),
            1,
            "RawCaptureEnabled must appear exactly once"
        );
    }

    // 3. SecretPresent with raw_capture_enabled false emits neither
    //    RawCaptureEnabled nor SecretCaptureAllowed.
    #[test]
    fn test_secret_present_no_raw_capture_emits_neither() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::SecretPresent],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "show me the secrets", false);

        assert!(
            !policy.risk_policy.contains(&RiskPolicy::RawCaptureEnabled),
            "SecretPresent+no raw_capture must not emit RawCaptureEnabled"
        );
        assert!(
            !policy
                .risk_policy
                .contains(&RiskPolicy::SecretCaptureAllowed),
            "SecretPresent+no raw_capture must not emit SecretCaptureAllowed"
        );
    }

    // 4. raw_capture_enabled true with RequestIntent::Explain includes
    //    ContextSource::RawCaptureFeatures and still includes TotalRecall.
    #[test]
    fn test_raw_capture_explain_includes_raw_capture_features_and_total_recall() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::None],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this", true);

        assert!(
            policy
                .context_sources
                .contains(&ContextSource::RawCaptureFeatures),
            "raw_capture_enabled=true must include RawCaptureFeatures"
        );
        assert!(
            policy.context_sources.contains(&ContextSource::TotalRecall),
            "Explain must still include TotalRecall"
        );
    }

    // 5. raw_capture_enabled false with RequestIntent::Explain does not
    //    include ContextSource::RawCaptureFeatures.
    #[test]
    fn test_no_raw_capture_explain_no_raw_capture_features() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::None],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this", false);

        assert!(
            !policy
                .context_sources
                .contains(&ContextSource::RawCaptureFeatures),
            "raw_capture_enabled=false must not include RawCaptureFeatures"
        );
    }

    // 6. raw_capture_enabled true emits RawCaptureFeatures exactly once.
    #[test]
    fn test_raw_capture_features_no_duplicates() {
        let c = classification(
            RequestIntent::Explain,
            vec![RequestRisk::None],
            RequestArtifactType::PlainText,
        );
        let policy = derive_orchestration_policy(&c, "explain this", true);

        let count = policy
            .context_sources
            .iter()
            .filter(|&&s| s == ContextSource::RawCaptureFeatures)
            .count();
        assert_eq!(count, 1, "RawCaptureFeatures must appear exactly once");
    }
}
