/// Deterministic rule detector for prompt intervention signals.
///
/// Detector findings are derived allegations. Runtime persistence and backfill
/// orchestration live outside this module.
use crate::prompt_intervention_extraction::ExtractedPromptSource;
use crate::prompt_intervention_fingerprint::{
    evidence_excerpt, evidence_hash_for_excerpt, fingerprint_prompt, PROMPT_FINGERPRINT_VERSION,
};
use crate::prompt_intervention_taxonomy::{
    BurdenType, FailureRelation, InterventionType, SignalFamily, SignalStrength, SourceKind,
};

/// MVP deterministic detector version.
pub const DETECTOR_VERSION: &str = "prompt-intervention-detector-v1";

/// MVP taxonomy version used by detector outputs.
pub const TAXONOMY_VERSION: &str = "prompt-interventions-v1";

/// A detected intervention with evidence and hashes ready for record assembly.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectedPromptIntervention {
    pub source_kind: SourceKind,
    pub intervention_type: InterventionType,
    pub signal_family: SignalFamily,
    pub signal_type: String,
    pub signal_strength: SignalStrength,
    pub burden_type: BurdenType,
    pub failure_relation: FailureRelation,
    pub confidence: f64,
    pub target_behavior: Option<String>,
    pub blocked_behavior: Option<String>,
    pub replacement_behavior: Option<String>,
    pub evidence_excerpt: String,
    pub evidence_hash: String,
    pub exact_prompt_hash: String,
    pub normalized_prompt_hash: String,
    pub prompt_fingerprint_version: u32,
    pub detector_version: &'static str,
    pub taxonomy_version: &'static str,
}

#[derive(Clone, Debug)]
struct RuleMatch {
    signal_type: &'static str,
    matched_phrase: &'static str,
    intervention_type: InterventionType,
    signal_family: SignalFamily,
    burden_type: BurdenType,
    failure_relation: FailureRelation,
    confidence: f64,
    target_behavior: Option<&'static str>,
    blocked_behavior: Option<&'static str>,
    replacement_behavior: Option<&'static str>,
}

/// Detect prompt intervention signals in extracted source candidates.
pub fn detect_prompt_interventions(
    sources: &[ExtractedPromptSource],
) -> Vec<DetectedPromptIntervention> {
    let mut detected = Vec::new();
    for source in sources {
        let fingerprint = fingerprint_prompt(&source.text);
        for rule_match in rule_matches(&source.text) {
            let excerpt = evidence_excerpt(&source.text, rule_match.matched_phrase);
            let evidence_hash = evidence_hash_for_excerpt(&excerpt);
            detected.push(DetectedPromptIntervention {
                source_kind: source.source_kind,
                intervention_type: rule_match.intervention_type,
                signal_family: rule_match.signal_family,
                signal_type: rule_match.signal_type.to_string(),
                signal_strength: SignalStrength::Explicit,
                burden_type: rule_match.burden_type,
                failure_relation: rule_match.failure_relation,
                confidence: rule_match.confidence,
                target_behavior: rule_match.target_behavior.map(str::to_string),
                blocked_behavior: rule_match.blocked_behavior.map(str::to_string),
                replacement_behavior: rule_match.replacement_behavior.map(str::to_string),
                evidence_excerpt: excerpt,
                evidence_hash,
                exact_prompt_hash: fingerprint.exact_prompt_hash.clone(),
                normalized_prompt_hash: fingerprint.normalized_prompt_hash.clone(),
                prompt_fingerprint_version: PROMPT_FINGERPRINT_VERSION,
                detector_version: DETECTOR_VERSION,
                taxonomy_version: TAXONOMY_VERSION,
            });
        }
    }
    detected
}

fn rule_matches(text: &str) -> Vec<RuleMatch> {
    let lower = text.to_lowercase();
    let mut matches = Vec::new();

    if let Some(phrase) = stop_and_redirect_phrase(&lower) {
        let replacement_behavior =
            if lower.contains("develop") && lower.contains("prompt") && lower.contains("report it")
            {
                "develop prompt and report it"
            } else {
                "alternate requested behavior"
            };

        matches.push(with_behaviors(
            rule(
                "stop_and_redirect",
                phrase,
                InterventionType::StopAndRedirect,
                SignalFamily::Steering,
                BurdenType::HumanStopControl,
                FailureRelation::Prevention,
                0.9,
            ),
            None,
            Some("implementation"),
            Some(replacement_behavior),
        ));
    }
    if let Some(phrase) = first_phrase(
        &lower,
        &[
            "do not implement",
            "do not edit",
            "do not write",
            "do not run",
            "do not deploy",
            "do not mutate",
            "don't implement",
            "don't edit",
            "don't write",
            "don't run",
            "don't deploy",
            "don't mutate",
        ],
    ) {
        matches.push(rule(
            "implementation_block",
            phrase,
            InterventionType::ImplementationBlock,
            SignalFamily::Steering,
            BurdenType::HumanStopControl,
            FailureRelation::Prevention,
            0.9,
        ));
    }
    if let Some(phrase) = first_phrase(
        &lower,
        &[
            "review and refine",
            "rewrite this prompt",
            "make the prompt executable",
            "repair this prompt",
        ],
    ) {
        matches.push(rule(
            "prompt_repair",
            phrase,
            InterventionType::PromptRepair,
            SignalFamily::Steering,
            BurdenType::HumanPromptRepair,
            FailureRelation::QualityControl,
            0.85,
        ));
    }
    if let Some(phrase) = first_phrase(
        &lower,
        &[
            "implementation-breaking",
            "not ready to hand off",
            "major issue",
            "blocking issue",
        ],
    ) {
        matches.push(rule(
            "quality_gate",
            phrase,
            InterventionType::QualityGate,
            SignalFamily::Steering,
            BurdenType::HumanPromptRepair,
            FailureRelation::QualityControl,
            0.85,
        ));
    }
    if let Some(phrase) = first_phrase(
        &lower,
        &[
            "edit only",
            "only edit",
            "do not read other files",
            "do not create files",
            "limit scope",
            "narrow the scope",
        ],
    ) {
        let target_behavior = if phrase == "edit only" || phrase == "only edit" {
            "single-file edit"
        } else {
            "scope-limited work"
        };

        matches.push(with_behaviors(
            rule(
                "scope_narrowing",
                phrase,
                InterventionType::ScopeNarrowing,
                SignalFamily::Steering,
                BurdenType::HumanScopeControl,
                FailureRelation::Prevention,
                0.9,
            ),
            Some(target_behavior),
            Some("broad repo edit or additional file access"),
            None,
        ));
    }
    if let Some(phrase) = validation_phrase(&lower) {
        matches.push(with_behaviors(
            rule(
                "validation_requirement",
                phrase,
                InterventionType::ValidationRequirement,
                SignalFamily::ValidationPressure,
                BurdenType::HumanValidationControl,
                FailureRelation::QualityControl,
                0.85,
            ),
            Some("testable implementation"),
            None,
            None,
        ));
    }
    if let Some(phrase) = risk_phrase(&lower) {
        matches.push(rule(
            "risk_warning",
            phrase,
            InterventionType::RiskWarning,
            SignalFamily::RiskControl,
            BurdenType::HumanRiskControl,
            FailureRelation::RiskControl,
            0.8,
        ));
    }
    if let Some(phrase) = clarification_phrase(&lower) {
        matches.push(rule(
            "clarification_request",
            phrase,
            InterventionType::ClarificationRequest,
            SignalFamily::ContextPressure,
            BurdenType::ContextRecovery,
            FailureRelation::Recovery,
            0.8,
        ));
    }
    if let Some(phrase) = first_phrase(
        &lower,
        &[
            "hallucinated",
            "wrong file",
            "wrong command",
            "fake",
            "malformed",
            "scope drift",
            "that is not in the repo",
            "not in the repo",
        ],
    ) {
        matches.push(rule(
            "model_failure_correction",
            phrase,
            InterventionType::ModelFailureCorrection,
            SignalFamily::FailureCorrection,
            BurdenType::ModelErrorRecovery,
            FailureRelation::Correction,
            0.8,
        ));
    }

    matches
}

fn rule(
    signal_type: &'static str,
    matched_phrase: &'static str,
    intervention_type: InterventionType,
    signal_family: SignalFamily,
    burden_type: BurdenType,
    failure_relation: FailureRelation,
    confidence: f64,
) -> RuleMatch {
    RuleMatch {
        signal_type,
        matched_phrase,
        intervention_type,
        signal_family,
        burden_type,
        failure_relation,
        confidence,
        target_behavior: None,
        blocked_behavior: None,
        replacement_behavior: None,
    }
}

fn with_behaviors(
    mut rule_match: RuleMatch,
    target_behavior: Option<&'static str>,
    blocked_behavior: Option<&'static str>,
    replacement_behavior: Option<&'static str>,
) -> RuleMatch {
    rule_match.target_behavior = target_behavior;
    rule_match.blocked_behavior = blocked_behavior;
    rule_match.replacement_behavior = replacement_behavior;
    rule_match
}

fn stop_and_redirect_phrase(lower: &str) -> Option<&'static str> {
    for phrase in ["i don't want you to do it", "do not do it"] {
        if lower.contains(phrase) && has_alternate_behavior(lower) {
            return Some(phrase);
        }
    }
    first_phrase(
        lower,
        &[
            "stop and",
            "stop now",
            "stop working",
            "stop doing",
            "please stop",
            "just stop",
            "stop here",
            "pause",
            "do not continue",
            "don't continue",
            "change task",
            "change direction",
            "switch to",
        ],
    )
}

fn has_alternate_behavior(lower: &str) -> bool {
    lower.contains("instead")
        || lower.contains("switch to")
        || lower.contains("do this")
        || lower.contains("review")
        || lower.contains("explain")
        || lower.contains("fix")
        || lower.contains("develop")
        || lower.contains("report it")
        || lower.contains("prompt")
}

fn validation_phrase(lower: &str) -> Option<&'static str> {
    first_phrase(
        lower,
        &[
            "acceptance tests required",
            "run tests",
            "validate",
            "dry run",
            "check before proceeding",
            "verification required",
        ],
    )
    .filter(|phrase| !is_generated_checklist(lower, phrase))
}

fn is_generated_checklist(lower: &str, phrase: &str) -> bool {
    let Some(pos) = lower.find(phrase) else {
        return false;
    };
    let line_start = lower[..pos].rfind('\n').map(|idx| idx + 1).unwrap_or(0);
    let line_prefix = lower[line_start..pos].trim_start();
    let in_list_item = line_prefix.starts_with("- ")
        || line_prefix.starts_with("* ")
        || line_prefix.starts_with("1. ")
        || line_prefix.starts_with("2. ")
        || line_prefix.starts_with("3. ");
    in_list_item
        && (lower.contains("next steps")
            || lower.contains("todo")
            || lower.contains("checklist")
            || lower.contains("action items"))
}

fn risk_phrase(lower: &str) -> Option<&'static str> {
    let phrase = first_phrase(
        lower,
        &[
            "unsafe",
            "production",
            "destructive",
            "secret",
            "rollback",
            "migration",
            "data loss",
            "iam",
        ],
    )?;
    let framed = first_phrase(
        lower,
        &[
            "caution",
            "blocker",
            "safety",
            "validation",
            "risk",
            "warning",
            "danger",
            "careful",
            "verify",
            "confirm",
            "do not proceed",
            "not safe",
            "unsafe to",
            "could break",
            "might break",
        ],
    )
    .is_some();
    framed.then_some(phrase)
}

fn clarification_phrase(lower: &str) -> Option<&'static str> {
    let targets = [
        "namespace",
        "deployment",
        "file",
        "repo",
        "command output",
        "command result",
        "prior decision",
        "previous decision",
    ];
    for target in targets {
        for prefix in [
            "what ",
            "which ",
            "missing ",
            "need ",
            "need the ",
            "need more ",
            "please provide ",
            "can you provide ",
        ] {
            let phrase = format!("{prefix}{target}");
            if lower.contains(&phrase) {
                return Some(match target {
                    "namespace" => "namespace",
                    "deployment" => "deployment",
                    "file" => "file",
                    "repo" => "repo",
                    "command output" => "command output",
                    "command result" => "command result",
                    "prior decision" => "prior decision",
                    "previous decision" => "previous decision",
                    _ => unreachable!(),
                });
            }
        }
    }
    if lower.contains("what was decided") {
        return Some("what was decided");
    }
    None
}

fn first_phrase<'a>(lower: &str, phrases: &'a [&'static str]) -> Option<&'a str> {
    phrases
        .iter()
        .copied()
        .find(|phrase| lower.contains(phrase))
}
