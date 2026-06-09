use std::collections::BTreeSet;

use crate::tool_mediation::{policy_allows_tool_capability, policy_blocks_tool_capability};
use crate::tool_mediation_classification::capability_for_tool_name;
use crate::tool_mediation_types::ToolSummary;

pub(crate) struct ToolPolicyShaping {
    pub(crate) hidden_names: BTreeSet<String>,
    pub(crate) policy_only_hidden_count: usize,
    pub(crate) all_blocked: bool,
}

pub(crate) fn policy_shaping_for_tools(
    offered: &[ToolSummary],
    canonical_hidden_names: &BTreeSet<String>,
    policy: Option<&crate::orchestration_policy::OrchestrationPolicy>,
) -> ToolPolicyShaping {
    let Some(policy) = policy else {
        return ToolPolicyShaping {
            hidden_names: BTreeSet::new(),
            policy_only_hidden_count: 0,
            all_blocked: false,
        };
    };

    let hidden_names = offered
        .iter()
        .filter(|tool| policy_hides_tool(policy, &tool.name))
        .map(|tool| tool.name.clone())
        .collect::<BTreeSet<_>>();

    let policy_only_hidden_count = hidden_names
        .iter()
        .filter(|name| !canonical_hidden_names.contains(*name))
        .count();

    let all_blocked = offered
        .iter()
        .all(|tool| policy_hides_tool(policy, &tool.name));

    ToolPolicyShaping {
        hidden_names,
        policy_only_hidden_count,
        all_blocked,
    }
}

fn policy_hides_tool(
    policy: &crate::orchestration_policy::OrchestrationPolicy,
    tool_name: &str,
) -> bool {
    let capability = capability_for_tool_name(tool_name);
    policy_blocks_tool_capability(policy, capability)
        || !policy_allows_tool_capability(policy, capability)
}
