use crate::feature_extraction_types::OperationalConstraint;

pub(crate) fn use_known_auth(header: &str) -> OperationalConstraint {
    OperationalConstraint {
        constraint_type: "use_known_auth".to_string(),
        text: format!("Use `{header}` when calling protected orchestrator endpoints."),
    }
}

pub(crate) fn use_known_endpoint(endpoint: &str) -> OperationalConstraint {
    OperationalConstraint {
        constraint_type: "use_known_endpoint".to_string(),
        text: format!(
            "Do not use `localhost` for host-side orchestrator testing. The correct endpoint for this environment is `{endpoint}`."
        ),
    }
}

pub(crate) fn use_known_migration_fix(fix: &str) -> OperationalConstraint {
    OperationalConstraint {
        constraint_type: "use_known_migration_fix".to_string(),
        text: format!(
            "When retrying the baseline migration, use `{fix}` to make extension creation idempotent."
        ),
    }
}

pub(crate) fn avoid_tool_loop(tools: &str) -> OperationalConstraint {
    OperationalConstraint {
        constraint_type: "avoid_tool_loop".to_string(),
        text: format!(
            "Do not repeat identical {tools} tool calls within this trajectory. Summarize the previous result and choose a different action before reusing the same tool with identical parameters."
        ),
    }
}

pub(crate) fn fix_context_retrieval() -> OperationalConstraint {
    OperationalConstraint {
        constraint_type: "fix_context_retrieval".to_string(),
        text: "If the context pack is empty or near-empty, verify cache warmup and retrieval health before assuming no prior memory exists.".to_string(),
    }
}

pub(crate) fn reduce_context_bloat() -> OperationalConstraint {
    OperationalConstraint {
        constraint_type: "reduce_context_bloat".to_string(),
        text: "Avoid expanding context further; use concise cached memory and inspect why input tokens or context truncation are high before retrying.".to_string(),
    }
}

pub(crate) fn separate_summarizer_upstream() -> OperationalConstraint {
    OperationalConstraint {
        constraint_type: "separate_summarizer_upstream".to_string(),
        text: "Keep background summarization on the dedicated summarizer endpoint instead of sharing foreground LiteLLM capacity.".to_string(),
    }
}

pub(crate) fn handle_user_interruption() -> OperationalConstraint {
    OperationalConstraint {
        constraint_type: "handle_user_interruption".to_string(),
        text: "When the user interrupts due to incorrect operational behavior, explicitly acknowledge the correction and apply it before continuing.".to_string(),
    }
}

pub(crate) fn handle_summarization_failure() -> OperationalConstraint {
    OperationalConstraint {
        constraint_type: "handle_summarization_failure".to_string(),
        text: "If summarization returns an empty response, inspect the provider or LiteLLM response body before retrying.".to_string(),
    }
}
