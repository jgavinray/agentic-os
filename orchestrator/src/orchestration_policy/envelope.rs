//! Render the derived orchestration policy as deterministic operating-envelope
//! guidance for the model.
//!
//! Tool mediation controls what the model *can* do; this module tells the
//! model what it *should* do inside that envelope. Small dense models in
//! particular tend to expand scope (drive-by refactors, opportunistic
//! cleanups), so the envelope states the scope discipline explicitly instead
//! of assuming the model infers it from a narrowed tool menu.

use crate::orchestration_policy::{
    EditPolicy, GitPolicy, OrchestrationPolicy, RuntimePolicy, ScopePolicy, ValidationPolicy,
};

pub const ENVELOPE_GUIDANCE_VERSION: &str = "policy-envelope-v1";

/// Render bounded, deterministic guidance text for the given policy.
///
/// Returns `None` only when the policy carries no actionable posture (all
/// postures unknown), so callers can skip injection entirely.
pub fn envelope_guidance(policy: &OrchestrationPolicy) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();

    if let Some(edit_line) = edit_discipline(policy.edit_policy) {
        lines.push(edit_line);
    }
    if let Some(validation_line) = validation_discipline(policy.validation_policy) {
        lines.push(validation_line);
    }
    if let Some(git_line) = git_discipline(policy.git_policy) {
        lines.push(git_line);
    }
    if let Some(runtime_line) = runtime_discipline(policy.runtime_policy) {
        lines.push(runtime_line);
    }
    if let Some(scope_line) = scope_discipline(&policy.scope_policy) {
        lines.push(scope_line);
    }

    if lines.is_empty() {
        return None;
    }

    let mut guidance = vec![
        "== Operating Envelope ==".to_string(),
        "The orchestrator derived this envelope from the request. Stay inside it.".to_string(),
    ];
    guidance.extend(lines);
    guidance.push(
        "If the request cannot be completed inside this envelope, stop and report what is \
         missing instead of expanding scope."
            .to_string(),
    );
    Some(guidance.join("\n"))
}

fn edit_discipline(edit: EditPolicy) -> Option<String> {
    let line = match edit {
        EditPolicy::ReadOnly => {
            "Edit policy: read_only. Do not create, modify, or delete any files."
        }
        EditPolicy::ExplicitFileOnly => {
            "Edit policy: explicit_file_only. Modify only files explicitly named in the \
             request; touch nothing else."
        }
        EditPolicy::SingleFileEdit => {
            "Edit policy: single_file_edit. Modify at most one file — the one the request \
             is about."
        }
        EditPolicy::ScopedEdit => {
            "Edit policy: scoped_edit. Make only the changes the request requires. Do not \
             refactor, reorganize, reformat, or \"improve\" code beyond the ask. Do not \
             rename files, functions, or variables unless explicitly requested. Leave \
             unrelated code untouched."
        }
        EditPolicy::MultiFileEdit => {
            "Edit policy: multi_file_edit. Edit the files the request requires; avoid \
             opportunistic cleanups in files you pass through."
        }
        EditPolicy::Unknown => return None,
    };
    Some(line.to_string())
}

fn validation_discipline(validation: ValidationPolicy) -> Option<String> {
    let line = match validation {
        ValidationPolicy::None => return None,
        ValidationPolicy::FormatOnly => "Validation: format_only. Run the formatter check only.",
        ValidationPolicy::Build => "Validation: build. Verify the project still builds.",
        ValidationPolicy::TargetedTests => {
            "Validation: targeted_tests. Run the narrowest tests that cover your change; \
             do not run or fix unrelated test suites."
        }
        ValidationPolicy::FullTests => "Validation: full_tests. Run the full test suite.",
        ValidationPolicy::DockerComposeHealth => {
            "Validation: docker_compose_health. Verify service health after the change."
        }
        ValidationPolicy::EndpointProbe => {
            "Validation: endpoint_probe. Verify the affected endpoint responds correctly."
        }
        ValidationPolicy::Unknown => return None,
    };
    Some(line.to_string())
}

fn git_discipline(git: GitPolicy) -> Option<String> {
    let line = match git {
        GitPolicy::NoGitChanges => {
            "Git: no_git_changes — do not commit, push, branch, tag, or otherwise mutate \
             git state."
        }
        GitPolicy::CommitAllowed => "Git: commit_allowed — you may commit; do not push.",
        GitPolicy::CommitRequired => "Git: commit_required — commit the change; do not push.",
        GitPolicy::PushAllowed => "Git: push_allowed.",
        GitPolicy::PushRequired => "Git: push_required.",
        GitPolicy::Unknown => return None,
    };
    Some(line.to_string())
}

fn runtime_discipline(runtime: RuntimePolicy) -> Option<String> {
    let line = match runtime {
        RuntimePolicy::NoRestart => {
            "Runtime: no_restart — do not restart, redeploy, or stop services."
        }
        RuntimePolicy::RestartAllowed => "Runtime: restart_allowed.",
        RuntimePolicy::RestartRequired => "Runtime: restart_required.",
        RuntimePolicy::DoNotInterruptActiveService => {
            "Runtime: do_not_interrupt_active_service — never interrupt a running service."
        }
        RuntimePolicy::RemoteHostAllowed => "Runtime: remote_host_allowed.",
        RuntimePolicy::Unknown => return None,
    };
    Some(line.to_string())
}

fn scope_discipline(scope: &[ScopePolicy]) -> Option<String> {
    if scope.is_empty() {
        return None;
    }
    let labels = scope
        .iter()
        .filter(|policy| !matches!(policy, ScopePolicy::Unknown))
        .map(|policy| policy.as_str())
        .collect::<Vec<_>>();
    if labels.is_empty() {
        return None;
    }
    Some(format!("Scope: {}.", labels.join("; ")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_classification::{
        RequestArtifactType, RequestClassification, RequestIntent,
    };

    fn policy_for(
        intent: RequestIntent,
        artifact_type: RequestArtifactType,
    ) -> OrchestrationPolicy {
        let mut classification = RequestClassification::deterministic(
            format!("event-{}", intent.as_str()),
            "agentic-os".to_string(),
            "session-envelope".to_string(),
            chrono::Utc::now(),
        );
        classification.intent = intent;
        classification.artifact_type = artifact_type;
        crate::orchestration_policy::derive_orchestration_policy(
            &classification,
            "representative request",
            false,
        )
    }

    #[test]
    fn implement_policy_renders_scoped_edit_discipline() {
        let policy = policy_for(RequestIntent::Implement, RequestArtifactType::Code);

        let guidance = envelope_guidance(&policy).expect("guidance");

        assert!(guidance.contains("== Operating Envelope =="));
        assert!(guidance.contains("scoped_edit"));
        assert!(
            guidance.contains("Do not refactor"),
            "scoped edit must forbid drive-by refactoring: {guidance}"
        );
        assert!(guidance.contains("do not commit"));
        assert!(guidance.len() <= 1_600, "guidance must stay bounded");
    }

    #[test]
    fn explain_policy_renders_read_only_discipline() {
        let policy = policy_for(RequestIntent::Explain, RequestArtifactType::PlainText);

        let guidance = envelope_guidance(&policy).expect("guidance");

        assert!(guidance.contains("read_only"));
        assert!(
            guidance.contains("Do not create, modify, or delete"),
            "read-only must forbid all file mutation: {guidance}"
        );
    }

    #[test]
    fn debug_code_policy_renders_single_file_discipline() {
        let policy = policy_for(RequestIntent::Debug, RequestArtifactType::Code);

        let guidance = envelope_guidance(&policy).expect("guidance");

        assert!(guidance.contains("single_file_edit"));
        assert!(guidance.contains("at most one file"));
    }

    #[test]
    fn guidance_includes_validation_and_scope_lines() {
        let policy = policy_for(RequestIntent::Implement, RequestArtifactType::Code);

        let guidance = envelope_guidance(&policy).expect("guidance");

        assert!(guidance.contains("targeted_tests"));
        assert!(guidance.contains("current_repo_only"));
        assert!(
            guidance.contains("stop and report"),
            "envelope must tell the model to stop instead of improvising: {guidance}"
        );
    }
}
