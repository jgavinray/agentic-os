use crate::orchestration_policy_intents::{
    classify_policy, debug_policy, explain_policy, generate_config_policy, implement_policy,
    modify_config_policy, operate_tool_policy, plan_policy, search_policy, summarize_policy,
    unknown_policy,
};
use crate::orchestration_policy_types::{
    ContextSource, EditPolicy, GitPolicy, RuntimePolicy, ScopePolicy, ToolCapability,
    ValidationPolicy,
};
use crate::request_classification_types::{RequestArtifactType, RequestIntent};

/// Raw policy pieces for a request intent before risk overlays are applied.
#[derive(Debug, Clone)]
pub(crate) struct BasePolicy {
    pub(crate) allowed: Vec<ToolCapability>,
    pub(crate) required: Vec<ToolCapability>,
    pub(crate) blocked: Vec<ToolCapability>,
    pub(crate) context: Vec<ContextSource>,
    pub(crate) edit: EditPolicy,
    pub(crate) validation: ValidationPolicy,
    pub(crate) git: GitPolicy,
    pub(crate) runtime: RuntimePolicy,
    pub(crate) scope: Vec<ScopePolicy>,
}

pub(crate) fn base_policy(intent: RequestIntent, artifact_type: RequestArtifactType) -> BasePolicy {
    // Keep this match exhaustive. If a new RequestIntent is added upstream, the
    // compiler should force us to define its operational posture here.
    match intent {
        RequestIntent::Explain => explain_policy(),
        RequestIntent::Debug => debug_policy(artifact_type),
        RequestIntent::Implement => implement_policy(),
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
