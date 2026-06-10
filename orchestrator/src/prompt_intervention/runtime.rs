use serde_json::{json, Value};

use crate::prompt_intervention_detector::{
    detect_prompt_interventions, DetectedPromptIntervention, DETECTOR_VERSION,
};
use crate::prompt_intervention_extraction::ExtractedPromptSource;
use crate::prompt_intervention_taxonomy::SourceKind;

pub const RUNTIME_INTERVENTION_VERSION: &str = "prompt-intervention-runtime-v1";
const MAX_GUIDANCE_CHARS: usize = 1_200;
const MAX_SIGNAL_LINES: usize = 5;

#[derive(Clone, Debug, PartialEq)]
pub struct RuntimePromptIntervention {
    pub findings: Vec<DetectedPromptIntervention>,
    pub guidance: Option<String>,
    pub metadata: Value,
}

pub fn build_runtime_prompt_intervention(user_content: &str) -> RuntimePromptIntervention {
    let sources = if user_content.trim().is_empty() {
        Vec::new()
    } else {
        vec![ExtractedPromptSource {
            source_kind: SourceKind::UserMessage,
            text: user_content.to_string(),
        }]
    };
    let findings = detect_prompt_interventions(&sources);
    let guidance = guidance_for_findings(&findings);
    let metadata = metadata_for_findings(&findings, guidance.is_some());

    RuntimePromptIntervention {
        findings,
        guidance,
        metadata,
    }
}

fn guidance_for_findings(findings: &[DetectedPromptIntervention]) -> Option<String> {
    if findings.is_empty() {
        return None;
    }

    let mut lines = vec![
        "== Runtime Prompt Intervention ==".to_string(),
        "Deterministic steering signals were found in the current user request. Preserve these constraints while completing the task.".to_string(),
    ];

    for finding in findings.iter().take(MAX_SIGNAL_LINES) {
        let mut line = format!(
            "- {} ({})",
            finding.signal_type,
            finding.intervention_type.as_str()
        );
        if let Some(target) = finding.target_behavior.as_deref() {
            line.push_str(&format!(" target={target}"));
        }
        if let Some(blocked) = finding.blocked_behavior.as_deref() {
            line.push_str(&format!(" avoid={blocked}"));
        }
        if let Some(replacement) = finding.replacement_behavior.as_deref() {
            line.push_str(&format!(" instead={replacement}"));
        }
        lines.push(line);
    }

    if findings.len() > MAX_SIGNAL_LINES {
        lines.push(format!(
            "- {} additional signal(s) omitted",
            findings.len() - MAX_SIGNAL_LINES
        ));
    }

    Some(bound_guidance(lines.join("\n")))
}

fn metadata_for_findings(
    findings: &[DetectedPromptIntervention],
    guidance_injected: bool,
) -> Value {
    let signals = findings
        .iter()
        .take(MAX_SIGNAL_LINES)
        .map(|finding| {
            json!({
                "signal_type": finding.signal_type,
                "intervention_type": finding.intervention_type.as_str(),
                "signal_family": finding.signal_family.as_str(),
                "source_kind": finding.source_kind.as_str(),
                "confidence": finding.confidence,
            })
        })
        .collect::<Vec<_>>();

    json!({
        "prompt_intervention_runtime": {
            "runtime_version": RUNTIME_INTERVENTION_VERSION,
            "detector_version": DETECTOR_VERSION,
            "finding_count": findings.len(),
            "guidance_injected": guidance_injected,
            "signals": signals,
        }
    })
}

fn bound_guidance(text: String) -> String {
    if text.len() <= MAX_GUIDANCE_CHARS {
        return text;
    }
    let mut bounded = text;
    bounded.truncate(MAX_GUIDANCE_CHARS);
    bounded.push_str("...");
    bounded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_findings_returns_no_guidance() {
        let outcome = build_runtime_prompt_intervention("Please explain this module.");

        assert!(outcome.findings.is_empty());
        assert!(outcome.guidance.is_none());
        assert_eq!(
            outcome.metadata["prompt_intervention_runtime"]["finding_count"],
            0
        );
    }

    #[test]
    fn scope_narrowing_returns_guidance() {
        let outcome = build_runtime_prompt_intervention("Edit only orchestrator/src/lib.rs.");

        assert_eq!(outcome.findings.len(), 1);
        let guidance = outcome.guidance.expect("guidance");
        assert!(guidance.contains("scope_narrowing"));
        assert!(guidance.contains("single-file edit"));
        assert_eq!(
            outcome.metadata["prompt_intervention_runtime"]["guidance_injected"],
            true
        );
    }

    #[test]
    fn validation_requirement_returns_stable_signal_label() {
        let outcome = build_runtime_prompt_intervention("Run tests before finishing.");

        let guidance = outcome.guidance.expect("guidance");
        assert!(guidance.contains("validation_requirement"));
        assert!(guidance.len() <= MAX_GUIDANCE_CHARS + 3);
        assert_eq!(
            outcome.metadata["prompt_intervention_runtime"]["signals"][0]["signal_type"],
            "validation_requirement"
        );
    }
}
