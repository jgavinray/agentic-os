use crate::orchestration_policy_types::ToolCapability;

pub(crate) use crate::orchestration_policy_intents_mutating::{
    debug_policy, generate_config_policy, implement_policy, modify_config_policy,
    operate_tool_policy,
};
pub(crate) use crate::orchestration_policy_intents_read::{
    classify_policy, explain_policy, plan_policy, search_policy, summarize_policy, unknown_policy,
};

pub(crate) fn mutation_blocklist() -> Vec<ToolCapability> {
    vec![
        ToolCapability::FileEdit,
        ToolCapability::ShellMutation,
        ToolCapability::DockerMutation,
        ToolCapability::Deploy,
        ToolCapability::RestartService,
        ToolCapability::GitWrite,
    ]
}
