use std::collections::BTreeSet;

use crate::tool_mediation_types::{ToolCapability, ToolIntent, ToolSummary};

pub(crate) fn hidden_tool_names(intent: ToolIntent, offered: &[ToolSummary]) -> BTreeSet<String> {
    let Some(canonical) = canonical_capability_for_intent(intent) else {
        return BTreeSet::new();
    };
    if !offered
        .iter()
        .any(|tool| tool.capability == canonical.as_str())
    {
        return BTreeSet::new();
    }

    match canonical {
        ToolCapability::FileRead
        | ToolCapability::TextSearch
        | ToolCapability::FileList
        | ToolCapability::ShellMutation => offered
            .iter()
            .filter(|tool| tool.capability == ToolCapability::Shell.as_str())
            .map(|tool| tool.name.clone())
            .collect(),
        _ => BTreeSet::new(),
    }
}

fn canonical_capability_for_intent(intent: ToolIntent) -> Option<ToolCapability> {
    match intent {
        ToolIntent::FileRead => Some(ToolCapability::FileRead),
        ToolIntent::TextSearch => Some(ToolCapability::TextSearch),
        ToolIntent::FileList => Some(ToolCapability::FileList),
        ToolIntent::FileEdit => Some(ToolCapability::FileEdit),
        ToolIntent::Validation => Some(ToolCapability::Validation),
        ToolIntent::Publishing => Some(ToolCapability::Publishing),
        ToolIntent::General | ToolIntent::Unknown => None,
    }
}
