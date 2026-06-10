//! Validation gate: turn the policy's validation expectation into an active
//! mid-trajectory signal.
//!
//! The policy can say `targeted_tests`, but a small model frequently edits
//! files and then declares success without ever validating. This gate scans
//! the trajectory evidence on every request: once edits are observed without
//! any validation command, it injects a deterministic reminder and records a
//! `validation_gate` metadata entry. The metadata is copied onto the model
//! response event, where harness feedback can flag a final answer that closed
//! the trajectory with the gap still open (`completion_without_validation`).

use serde_json::{json, Value};

use crate::orchestration_policy::{OrchestrationPolicy, ValidationPolicy};
use crate::tool_mediation::broadening::TrajectoryToolEvidence;

pub const VALIDATION_GATE_VERSION: &str = "validation-gate-v1";

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationGateOutcome {
    /// Guidance to inject when edits happened but validation has not.
    pub nudge: Option<String>,
    /// `validation_gate` metadata, present whenever the policy requires
    /// validation, so response events always carry the gate state.
    pub metadata: Option<Value>,
}

pub fn evaluate_validation_gate(
    evidence: &TrajectoryToolEvidence,
    policy: &OrchestrationPolicy,
) -> ValidationGateOutcome {
    if !matches!(
        policy.validation_policy,
        ValidationPolicy::Build | ValidationPolicy::TargetedTests | ValidationPolicy::FullTests
    ) {
        return ValidationGateOutcome {
            nudge: None,
            metadata: None,
        };
    }

    let gap = evidence.edits_observed && !evidence.validation_observed;
    let nudge = gap.then(|| {
        format!(
            "== Validation Gate ==\nYou have modified files in this session but have not run \
             validation yet. Required validation: {}. Run the narrowest build/test command that \
             covers your changes and show its output before declaring the task complete.",
            policy.validation_policy.as_str()
        )
    });

    let metadata = Some(json!({
        "validation_gate": {
            "version": VALIDATION_GATE_VERSION,
            "required": policy.validation_policy.as_str(),
            "edits_observed": evidence.edits_observed,
            "validation_observed": evidence.validation_observed,
            "nudge_injected": nudge.is_some(),
        }
    }));

    ValidationGateOutcome { nudge, metadata }
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
            format!("event-gate-{}", intent.as_str()),
            "agentic-os".to_string(),
            "session-gate".to_string(),
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
    fn edits_without_validation_produce_nudge_and_gap_metadata() {
        let evidence = TrajectoryToolEvidence {
            edits_observed: true,
            validation_observed: false,
        };
        let policy = policy_for(RequestIntent::Implement, RequestArtifactType::Code);

        let outcome = evaluate_validation_gate(&evidence, &policy);

        let nudge = outcome.nudge.expect("nudge");
        assert!(nudge.contains("have not run validation"));
        let metadata = outcome.metadata.expect("metadata");
        assert_eq!(metadata["validation_gate"]["edits_observed"], true);
        assert_eq!(metadata["validation_gate"]["validation_observed"], false);
        assert_eq!(metadata["validation_gate"]["nudge_injected"], true);
    }

    #[test]
    fn validated_trajectory_has_metadata_but_no_nudge() {
        let evidence = TrajectoryToolEvidence {
            edits_observed: true,
            validation_observed: true,
        };
        let policy = policy_for(RequestIntent::Implement, RequestArtifactType::Code);

        let outcome = evaluate_validation_gate(&evidence, &policy);

        assert!(outcome.nudge.is_none());
        let metadata = outcome.metadata.expect("metadata");
        assert_eq!(metadata["validation_gate"]["validation_observed"], true);
    }

    #[test]
    fn read_only_policies_skip_the_gate_entirely() {
        let evidence = TrajectoryToolEvidence {
            edits_observed: true,
            validation_observed: false,
        };
        let policy = policy_for(RequestIntent::Explain, RequestArtifactType::PlainText);

        let outcome = evaluate_validation_gate(&evidence, &policy);

        assert!(outcome.nudge.is_none());
        assert!(outcome.metadata.is_none());
    }

    #[test]
    fn no_edits_yet_means_no_nudge() {
        let evidence = TrajectoryToolEvidence::default();
        let policy = policy_for(RequestIntent::Implement, RequestArtifactType::Code);

        let outcome = evaluate_validation_gate(&evidence, &policy);

        assert!(outcome.nudge.is_none());
        assert_eq!(
            outcome.metadata.expect("metadata")["validation_gate"]["nudge_injected"],
            false
        );
    }
}
