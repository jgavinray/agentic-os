use serde_json::{json, Value};

use crate::prompt_intervention_detector::{
    detect_prompt_interventions, DetectedPromptIntervention, DETECTOR_VERSION,
};
use crate::prompt_intervention_extraction::ExtractedPromptSource;
use crate::prompt_intervention_taxonomy::SourceKind;

pub const RUNTIME_INTERVENTION_VERSION: &str = "prompt-intervention-runtime-v2";
const MAX_GUIDANCE_CHARS: usize = 1_200;
const MAX_SIGNAL_LINES: usize = 5;
/// How many recent typed user messages to scan for operator constraints.
const MAX_SCANNED_USER_MESSAGES: usize = 8;
/// Longest evidence quote echoed back into guidance.
const MAX_EVIDENCE_QUOTE_CHARS: usize = 80;

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
    build_from_sources(sources)
}

/// Build runtime interventions by scanning the request's recent typed user
/// messages (newest first), not just the latest one.
///
/// Agent loops append tool-result user messages and short "continue" turns
/// after the operator's steering message; scanning the bounded message window
/// keeps stated constraints alive for the rest of the trajectory without any
/// server-side session state. Findings are deduplicated by signal type with
/// the newest occurrence winning.
pub fn build_runtime_prompt_intervention_from_request(req: &Value) -> RuntimePromptIntervention {
    let sources = user_sources_from_request(req);
    build_from_sources(sources)
}

fn build_from_sources(sources: Vec<ExtractedPromptSource>) -> RuntimePromptIntervention {
    let mut findings = detect_prompt_interventions(&sources);
    dedupe_by_signal_type(&mut findings);
    let guidance = guidance_for_findings(&findings);
    let metadata = metadata_for_findings(&findings, guidance.is_some());

    RuntimePromptIntervention {
        findings,
        guidance,
        metadata,
    }
}

fn user_sources_from_request(req: &Value) -> Vec<ExtractedPromptSource> {
    let Some(messages) = req.get("messages").and_then(Value::as_array) else {
        return Vec::new();
    };
    messages
        .iter()
        .rev()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        .map(typed_user_text)
        .filter(|text| !text.trim().is_empty())
        .take(MAX_SCANNED_USER_MESSAGES)
        .map(|text| ExtractedPromptSource {
            source_kind: SourceKind::UserMessage,
            text,
        })
        .collect()
}

/// Extract operator-typed text from a user message, excluding tool-result
/// blocks and harness `<system-reminder>` content.
fn typed_user_text(message: &Value) -> String {
    let text = match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter(|block| {
                block.get("type").and_then(Value::as_str) == Some("text")
                    || (block.get("type").is_none() && block.get("text").is_some())
            })
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    };
    crate::anthropic_user_content::strip_system_reminders(&text)
        .trim()
        .to_string()
}

fn dedupe_by_signal_type(findings: &mut Vec<DetectedPromptIntervention>) {
    let mut seen = std::collections::BTreeSet::new();
    findings.retain(|finding| seen.insert(finding.signal_type.clone()));
}

fn guidance_for_findings(findings: &[DetectedPromptIntervention]) -> Option<String> {
    if findings.is_empty() {
        return None;
    }

    let mut lines = vec![
        "== Operator Constraints ==".to_string(),
        "The request states explicit constraints. Follow them exactly:".to_string(),
    ];

    for finding in findings.iter().take(MAX_SIGNAL_LINES) {
        lines.push(format!("- {}", imperative_for(finding)));
    }

    if findings.len() > MAX_SIGNAL_LINES {
        lines.push(format!(
            "- {} additional constraint(s) omitted",
            findings.len() - MAX_SIGNAL_LINES
        ));
    }

    Some(bound_guidance(lines.join("\n")))
}

/// Render one finding as a plain imperative a small model can follow, grounded
/// with a short quote from the request.
fn imperative_for(finding: &DetectedPromptIntervention) -> String {
    let base = match finding.signal_type.as_str() {
        "scope_narrowing" => match finding.target_behavior.as_deref() {
            Some("single-file edit") => {
                "Edit only the file the request names. Do not open, create, or modify other files."
            }
            _ => "Keep every change inside the scope the request defines. Do not expand scope.",
        },
        "refactor_block" => {
            "Do not refactor, clean up, or reorganize anything beyond the explicit request."
        }
        "validation_requirement" => {
            "Run the required validation and show its output before declaring the task complete."
        }
        "implementation_block" => {
            "Do not implement, edit, or execute anything; this request forbids those actions."
        }
        "stop_and_redirect" => {
            "Stop the current approach and do what the request redirects you to instead."
        }
        "risk_warning" => {
            "This is risk-sensitive: verify before acting and do not perform unsafe or destructive operations."
        }
        "clarification_request" => {
            "Information is missing; address the clarification before proceeding with the task."
        }
        "prompt_repair" => {
            "The task is to repair or refine the prompt/spec itself, not to implement it."
        }
        "quality_gate" => "A blocking quality issue is named; resolve it before handing off.",
        "model_failure_correction" => {
            "A previous output was wrong; re-verify against the repository instead of repeating it."
        }
        _ => "Honor the stated constraint.",
    };
    format!(
        "{base} (request: \"{}\")",
        evidence_quote(&finding.evidence_excerpt)
    )
}

fn evidence_quote(excerpt: &str) -> String {
    let cleaned = excerpt.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.chars().count() <= MAX_EVIDENCE_QUOTE_CHARS {
        return cleaned;
    }
    let mut quote: String = cleaned.chars().take(MAX_EVIDENCE_QUOTE_CHARS).collect();
    quote.push('…');
    quote
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
    fn scope_narrowing_renders_imperative_guidance() {
        let outcome = build_runtime_prompt_intervention("Edit only orchestrator/src/lib.rs.");

        assert_eq!(outcome.findings.len(), 1);
        let guidance = outcome.guidance.expect("guidance");
        assert!(
            guidance.contains("Edit only the file the request names"),
            "guidance must be a plain imperative, got: {guidance}"
        );
        assert!(
            !guidance.contains("corrective_steering"),
            "taxonomy labels must not leak into model guidance: {guidance}"
        );
        assert_eq!(
            outcome.metadata["prompt_intervention_runtime"]["guidance_injected"],
            true
        );
    }

    #[test]
    fn refactor_block_renders_imperative_guidance() {
        let outcome = build_runtime_prompt_intervention(
            "Add the helper to rule_utils.rs. Do not refactor anything else.",
        );

        let guidance = outcome.guidance.expect("guidance");
        assert!(
            guidance.contains("Do not refactor, clean up, or reorganize"),
            "got: {guidance}"
        );
    }

    #[test]
    fn validation_requirement_keeps_stable_signal_metadata() {
        let outcome = build_runtime_prompt_intervention("Run tests before finishing.");

        let guidance = outcome.guidance.expect("guidance");
        assert!(
            guidance.contains("Run the required validation"),
            "got: {guidance}"
        );
        assert!(guidance.len() <= MAX_GUIDANCE_CHARS + 3);
        assert_eq!(
            outcome.metadata["prompt_intervention_runtime"]["signals"][0]["signal_type"],
            "validation_requirement"
        );
    }

    #[test]
    fn request_scan_picks_up_constraints_from_earlier_user_messages() {
        // Agent loops append tool-result user messages after the operator's
        // steering message; the constraint must survive those turns.
        let req = serde_json::json!({
            "messages": [
                {"role": "user", "content": "Edit only orchestrator/src/lib.rs."},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Working on it."},
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "file body"}
                ]},
                {"role": "user", "content": "continue"}
            ]
        });

        let outcome = build_runtime_prompt_intervention_from_request(&req);

        let guidance = outcome.guidance.expect("guidance");
        assert!(
            guidance.contains("Edit only the file the request names"),
            "constraint from earlier message must persist: {guidance}"
        );
    }

    #[test]
    fn request_scan_dedupes_repeated_signal_types() {
        let req = serde_json::json!({
            "messages": [
                {"role": "user", "content": "Run tests before finishing."},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "Remember to run tests before finishing."}
            ]
        });

        let outcome = build_runtime_prompt_intervention_from_request(&req);

        assert_eq!(
            outcome
                .findings
                .iter()
                .filter(|finding| finding.signal_type == "validation_requirement")
                .count(),
            1,
            "repeated signals across messages must dedupe"
        );
    }

    #[test]
    fn request_scan_ignores_system_reminder_text() {
        let req = serde_json::json!({
            "messages": [
                {"role": "user", "content": "Explain the module.\n<system-reminder>Run tests before finishing.</system-reminder>"}
            ]
        });

        let outcome = build_runtime_prompt_intervention_from_request(&req);

        assert!(
            outcome.findings.is_empty(),
            "harness reminders must not be treated as operator constraints"
        );
    }
}
