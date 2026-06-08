/// Closed taxonomy enums for prompt intervention signals.
///
/// These enums are exhaustive — new values require an explicit taxonomy change.
/// They map directly to the canonical definitions in
/// docs/PromptInterventions/03-signal-taxonomy.md.
use serde::{Deserialize, Serialize};

// ── Source Kinds ───────────────────────────────────────────────

/// Where the evidence for an intervention came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// Evidence came from the whole extracted raw prompt body.
    RawPrompt,
    /// Evidence came from a user-role message.
    UserMessage,
    /// Evidence came from an assistant-role message present in the request history.
    AssistantMessage,
    /// Evidence came from a tool-result message or equivalent tool output message.
    ToolResult,
    /// Evidence was added by a later operator or review pass.
    PosthocReview,
}

impl SourceKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            SourceKind::RawPrompt => "raw_prompt",
            SourceKind::UserMessage => "user_message",
            SourceKind::AssistantMessage => "assistant_message",
            SourceKind::ToolResult => "tool_result",
            SourceKind::PosthocReview => "posthoc_review",
        }
    }
}

// ── Intervention Types ─────────────────────────────────────────

/// The kind of intervention observed in the prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionType {
    /// Blocks the current action and replaces it with another task.
    StopAndRedirect,
    /// Reduces files, tools, task breadth, context breadth, or allowed behavior.
    ScopeNarrowing,
    /// Rewrites or repairs a prompt/spec before handoff.
    PromptRepair,
    /// Blocks handoff until a concrete issue is fixed.
    QualityGate,
    /// Identifies safety, production, data, secret, destructive, or rollback risk.
    RiskWarning,
    /// Asks for missing context required to proceed.
    ClarificationRequest,
    /// Explicitly says not to implement, edit, write, run, deploy, or mutate state.
    ImplementationBlock,
    /// Adds tests, commands, dry-runs, checks, or acceptance gates required before completion.
    ValidationRequirement,
    /// Corrects hallucination, wrong files, wrong commands, malformed output, or scope drift.
    ModelFailureCorrection,
    /// Bounded fallback for early backfill; must be rare.
    Other,
}

impl InterventionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            InterventionType::StopAndRedirect => "stop_and_redirect",
            InterventionType::ScopeNarrowing => "scope_narrowing",
            InterventionType::PromptRepair => "prompt_repair",
            InterventionType::QualityGate => "quality_gate",
            InterventionType::RiskWarning => "risk_warning",
            InterventionType::ClarificationRequest => "clarification_request",
            InterventionType::ImplementationBlock => "implementation_block",
            InterventionType::ValidationRequirement => "validation_requirement",
            InterventionType::ModelFailureCorrection => "model_failure_correction",
            InterventionType::Other => "other",
        }
    }

    /// Returns the primary signal family for this intervention type.
    pub fn primary_signal_family(&self) -> SignalFamily {
        match self {
            InterventionType::StopAndRedirect
            | InterventionType::ScopeNarrowing
            | InterventionType::PromptRepair
            | InterventionType::QualityGate
            | InterventionType::ImplementationBlock => SignalFamily::Steering,
            InterventionType::ModelFailureCorrection => SignalFamily::FailureCorrection,
            InterventionType::RiskWarning => SignalFamily::RiskControl,
            InterventionType::ValidationRequirement => SignalFamily::ValidationPressure,
            InterventionType::ClarificationRequest => SignalFamily::ContextPressure,
            InterventionType::Other => SignalFamily::NoSignal,
        }
    }
}

// ── Signal Families ────────────────────────────────────────────

/// The broad category of steering or control represented by the evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalFamily {
    /// The prompt redirects, narrows, blocks, or repairs model behavior.
    Steering,
    /// The prompt corrects a model or harness mistake.
    FailureCorrection,
    /// The prompt introduces a safety, production, data, secret, or destructive-action constraint.
    RiskControl,
    /// The prompt requires tests, dry-runs, checks, or other verification before acceptance.
    ValidationPressure,
    /// The prompt asks for missing context, prior decisions, identifiers, files, logs, etc.
    ContextPressure,
    /// No intervention signal was detected.
    NoSignal,
}

impl SignalFamily {
    pub fn as_str(&self) -> &'static str {
        match self {
            SignalFamily::Steering => "steering",
            SignalFamily::FailureCorrection => "failure_correction",
            SignalFamily::RiskControl => "risk_control",
            SignalFamily::ValidationPressure => "validation_pressure",
            SignalFamily::ContextPressure => "context_pressure",
            SignalFamily::NoSignal => "no_signal",
        }
    }
}

// ── Signal Strength ────────────────────────────────────────────

/// How directly the prompt states the steering or failure correction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalStrength {
    /// The prompt directly states the steering or failure correction.
    Explicit,
    /// The prompt implies steering from constrained instructions.
    Implicit,
    /// There is a weak match; should not count in headline metrics unless confidence is high.
    Ambiguous,
}

impl SignalStrength {
    pub fn as_str(&self) -> &'static str {
        match self {
            SignalStrength::Explicit => "explicit",
            SignalStrength::Implicit => "implicit",
            SignalStrength::Ambiguous => "ambiguous",
        }
    }
}

// ── Burden Types ───────────────────────────────────────────────

/// What kind of burden or failure the intervention represents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BurdenType {
    /// A user or prior assistant rewrote the prompt so the next model attempt could succeed.
    HumanPromptRepair,
    /// A user or prior assistant narrowed files, tools, behavior, or task breadth.
    HumanScopeControl,
    /// A user or prior assistant stopped or redirected an undesired path.
    HumanStopControl,
    /// A user or prior assistant added required verification gates.
    HumanValidationControl,
    /// A user or prior assistant added safety or operational constraints.
    HumanRiskControl,
    /// The prompt requests missing context needed before safe progress.
    ContextRecovery,
    /// The prompt corrects hallucination, wrong file, malformed output, or scope drift.
    ModelErrorRecovery,
    /// Used only when a valid signal exists but the burden cannot be classified more specifically.
    UnknownBurden,
}

impl BurdenType {
    pub fn as_str(&self) -> &'static str {
        match self {
            BurdenType::HumanPromptRepair => "human_prompt_repair",
            BurdenType::HumanScopeControl => "human_scope_control",
            BurdenType::HumanStopControl => "human_stop_control",
            BurdenType::HumanValidationControl => "human_validation_control",
            BurdenType::HumanRiskControl => "human_risk_control",
            BurdenType::ContextRecovery => "context_recovery",
            BurdenType::ModelErrorRecovery => "model_error_recovery",
            BurdenType::UnknownBurden => "unknown_burden",
        }
    }
}

// ── Failure Relations ──────────────────────────────────────────

/// How the steering relates to failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureRelation {
    /// Steering was applied before a bad action occurred.
    Prevention,
    /// Steering corrected an observed bad or inadequate action.
    Correction,
    /// Steering helped resume after failure or missing context.
    Recovery,
    /// Steering raised the quality bar before handoff.
    QualityControl,
    /// Steering reduced safety or operational risk.
    RiskControl,
    /// Used only when the relationship is unclear.
    UnknownRelation,
}

impl FailureRelation {
    pub fn as_str(&self) -> &'static str {
        match self {
            FailureRelation::Prevention => "prevention",
            FailureRelation::Correction => "correction",
            FailureRelation::Recovery => "recovery",
            FailureRelation::QualityControl => "quality_control",
            FailureRelation::RiskControl => "risk_control",
            FailureRelation::UnknownRelation => "unknown_relation",
        }
    }
}

// ── Labeler Types ──────────────────────────────────────────────

/// What produced the intervention label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelerType {
    /// Deterministic detector produced the label.
    Rule,
    /// Operator or backfill review produced the label.
    Posthoc,
    /// Human-facing correction workflow produced the label.
    Human,
    /// Reserved for later offline experiments.
    LocalModel,
    /// Reserved for later offline experiments.
    FrontierModel,
}

impl LabelerType {
    pub fn as_str(&self) -> &'static str {
        match self {
            LabelerType::Rule => "rule",
            LabelerType::Posthoc => "posthoc",
            LabelerType::Human => "human",
            LabelerType::LocalModel => "local_model",
            LabelerType::FrontierModel => "frontier_model",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_kind_as_str() {
        assert_eq!(SourceKind::RawPrompt.as_str(), "raw_prompt");
        assert_eq!(SourceKind::UserMessage.as_str(), "user_message");
        assert_eq!(SourceKind::AssistantMessage.as_str(), "assistant_message");
        assert_eq!(SourceKind::ToolResult.as_str(), "tool_result");
        assert_eq!(SourceKind::PosthocReview.as_str(), "posthoc_review");
    }

    #[test]
    fn test_intervention_type_as_str() {
        assert_eq!(
            InterventionType::StopAndRedirect.as_str(),
            "stop_and_redirect"
        );
        assert_eq!(InterventionType::ScopeNarrowing.as_str(), "scope_narrowing");
        assert_eq!(InterventionType::PromptRepair.as_str(), "prompt_repair");
        assert_eq!(InterventionType::QualityGate.as_str(), "quality_gate");
        assert_eq!(InterventionType::RiskWarning.as_str(), "risk_warning");
        assert_eq!(
            InterventionType::ClarificationRequest.as_str(),
            "clarification_request"
        );
        assert_eq!(
            InterventionType::ImplementationBlock.as_str(),
            "implementation_block"
        );
        assert_eq!(
            InterventionType::ValidationRequirement.as_str(),
            "validation_requirement"
        );
        assert_eq!(
            InterventionType::ModelFailureCorrection.as_str(),
            "model_failure_correction"
        );
        assert_eq!(InterventionType::Other.as_str(), "other");
    }

    #[test]
    fn test_signal_family_as_str() {
        assert_eq!(SignalFamily::Steering.as_str(), "steering");
        assert_eq!(
            SignalFamily::FailureCorrection.as_str(),
            "failure_correction"
        );
        assert_eq!(SignalFamily::RiskControl.as_str(), "risk_control");
        assert_eq!(
            SignalFamily::ValidationPressure.as_str(),
            "validation_pressure"
        );
        assert_eq!(SignalFamily::ContextPressure.as_str(), "context_pressure");
        assert_eq!(SignalFamily::NoSignal.as_str(), "no_signal");
    }

    #[test]
    fn test_signal_strength_as_str() {
        assert_eq!(SignalStrength::Explicit.as_str(), "explicit");
        assert_eq!(SignalStrength::Implicit.as_str(), "implicit");
        assert_eq!(SignalStrength::Ambiguous.as_str(), "ambiguous");
    }

    #[test]
    fn test_burden_type_as_str() {
        assert_eq!(
            BurdenType::HumanPromptRepair.as_str(),
            "human_prompt_repair"
        );
        assert_eq!(
            BurdenType::HumanScopeControl.as_str(),
            "human_scope_control"
        );
        assert_eq!(BurdenType::HumanStopControl.as_str(), "human_stop_control");
        assert_eq!(
            BurdenType::HumanValidationControl.as_str(),
            "human_validation_control"
        );
        assert_eq!(BurdenType::HumanRiskControl.as_str(), "human_risk_control");
        assert_eq!(BurdenType::ContextRecovery.as_str(), "context_recovery");
        assert_eq!(
            BurdenType::ModelErrorRecovery.as_str(),
            "model_error_recovery"
        );
        assert_eq!(BurdenType::UnknownBurden.as_str(), "unknown_burden");
    }

    #[test]
    fn test_failure_relation_as_str() {
        assert_eq!(FailureRelation::Prevention.as_str(), "prevention");
        assert_eq!(FailureRelation::Correction.as_str(), "correction");
        assert_eq!(FailureRelation::Recovery.as_str(), "recovery");
        assert_eq!(FailureRelation::QualityControl.as_str(), "quality_control");
        assert_eq!(FailureRelation::RiskControl.as_str(), "risk_control");
        assert_eq!(
            FailureRelation::UnknownRelation.as_str(),
            "unknown_relation"
        );
    }

    #[test]
    fn test_labeler_type_as_str() {
        assert_eq!(LabelerType::Rule.as_str(), "rule");
        assert_eq!(LabelerType::Posthoc.as_str(), "posthoc");
        assert_eq!(LabelerType::Human.as_str(), "human");
        assert_eq!(LabelerType::LocalModel.as_str(), "local_model");
        assert_eq!(LabelerType::FrontierModel.as_str(), "frontier_model");
    }

    #[test]
    fn test_intervention_type_primary_signal_family() {
        assert_eq!(
            InterventionType::StopAndRedirect.primary_signal_family(),
            SignalFamily::Steering
        );
        assert_eq!(
            InterventionType::ScopeNarrowing.primary_signal_family(),
            SignalFamily::Steering
        );
        assert_eq!(
            InterventionType::PromptRepair.primary_signal_family(),
            SignalFamily::Steering
        );
        assert_eq!(
            InterventionType::QualityGate.primary_signal_family(),
            SignalFamily::Steering
        );
        assert_eq!(
            InterventionType::ImplementationBlock.primary_signal_family(),
            SignalFamily::Steering
        );
        assert_eq!(
            InterventionType::ModelFailureCorrection.primary_signal_family(),
            SignalFamily::FailureCorrection
        );
        assert_eq!(
            InterventionType::RiskWarning.primary_signal_family(),
            SignalFamily::RiskControl
        );
        assert_eq!(
            InterventionType::ValidationRequirement.primary_signal_family(),
            SignalFamily::ValidationPressure
        );
        assert_eq!(
            InterventionType::ClarificationRequest.primary_signal_family(),
            SignalFamily::ContextPressure
        );
        assert_eq!(
            InterventionType::Other.primary_signal_family(),
            SignalFamily::NoSignal
        );
    }

    #[test]
    fn test_serde_source_kind_snake_case() {
        let json = serde_json::to_string(&SourceKind::RawPrompt).unwrap();
        assert_eq!(json, "\"raw_prompt\"");
        let json = serde_json::to_string(&SourceKind::PosthocReview).unwrap();
        assert_eq!(json, "\"posthoc_review\"");
    }

    #[test]
    fn test_serde_intervention_type_snake_case() {
        let json = serde_json::to_string(&InterventionType::StopAndRedirect).unwrap();
        assert_eq!(json, "\"stop_and_redirect\"");
        let json = serde_json::to_string(&InterventionType::ScopeNarrowing).unwrap();
        assert_eq!(json, "\"scope_narrowing\"");
        let json = serde_json::to_string(&InterventionType::Other).unwrap();
        assert_eq!(json, "\"other\"");
    }

    #[test]
    fn test_serde_signal_family_snake_case() {
        let json = serde_json::to_string(&SignalFamily::Steering).unwrap();
        assert_eq!(json, "\"steering\"");
        let json = serde_json::to_string(&SignalFamily::NoSignal).unwrap();
        assert_eq!(json, "\"no_signal\"");
    }

    #[test]
    fn test_serde_signal_strength_snake_case() {
        let json = serde_json::to_string(&SignalStrength::Explicit).unwrap();
        assert_eq!(json, "\"explicit\"");
        let json = serde_json::to_string(&SignalStrength::Ambiguous).unwrap();
        assert_eq!(json, "\"ambiguous\"");
    }

    #[test]
    fn test_serde_burden_type_snake_case() {
        let json = serde_json::to_string(&BurdenType::HumanPromptRepair).unwrap();
        assert_eq!(json, "\"human_prompt_repair\"");
        let json = serde_json::to_string(&BurdenType::UnknownBurden).unwrap();
        assert_eq!(json, "\"unknown_burden\"");
    }

    #[test]
    fn test_serde_failure_relation_snake_case() {
        let json = serde_json::to_string(&FailureRelation::Prevention).unwrap();
        assert_eq!(json, "\"prevention\"");
        let json = serde_json::to_string(&FailureRelation::UnknownRelation).unwrap();
        assert_eq!(json, "\"unknown_relation\"");
    }

    #[test]
    fn test_serde_labeler_type_snake_case() {
        let json = serde_json::to_string(&LabelerType::Rule).unwrap();
        assert_eq!(json, "\"rule\"");
        let json = serde_json::to_string(&LabelerType::FrontierModel).unwrap();
        assert_eq!(json, "\"frontier_model\"");
    }

    #[test]
    fn test_serde_roundtrip_all_enums() {
        let src: SourceKind = serde_json::from_str("\"raw_prompt\"").unwrap();
        assert_eq!(src, SourceKind::RawPrompt);

        let it: InterventionType = serde_json::from_str("\"model_failure_correction\"").unwrap();
        assert_eq!(it, InterventionType::ModelFailureCorrection);

        let sf: SignalFamily = serde_json::from_str("\"context_pressure\"").unwrap();
        assert_eq!(sf, SignalFamily::ContextPressure);

        let ss: SignalStrength = serde_json::from_str("\"implicit\"").unwrap();
        assert_eq!(ss, SignalStrength::Implicit);

        let bt: BurdenType = serde_json::from_str("\"context_recovery\"").unwrap();
        assert_eq!(bt, BurdenType::ContextRecovery);

        let fr: FailureRelation = serde_json::from_str("\"recovery\"").unwrap();
        assert_eq!(fr, FailureRelation::Recovery);

        let lt: LabelerType = serde_json::from_str("\"posthoc\"").unwrap();
        assert_eq!(lt, LabelerType::Posthoc);
    }
}
