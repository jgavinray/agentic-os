use serde::{Deserialize, Serialize};

/// Prompt / spec refinement behaviour.
///
/// This policy is a workflow mode layered on top of intent. For example, a
/// request can classify as `Explain` while still being handled as a prompt/spec
/// review workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PromptRefinementPolicy {
    /// No prompt/spec refinement policy applies.
    None,
    /// Review only; do not rewrite.
    CritiqueOnly,
    /// Produce one rewrite.
    RewriteOnce,
    /// Review through multiple passes or model collaborations.
    MultiPassReview,
    /// Execute only after explicit approval.
    ExecuteAfterApproval,
    /// Store the accepted prompt/spec shape for later reuse.
    StoreAcceptedPromptPattern,
    /// Fallback label for corrupt, missing, or future values.
    #[default]
    Unknown,
}

impl PromptRefinementPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::CritiqueOnly => "critique_only",
            Self::RewriteOnce => "rewrite_once",
            Self::MultiPassReview => "multi_pass_review",
            Self::ExecuteAfterApproval => "execute_after_approval",
            Self::StoreAcceptedPromptPattern => "store_accepted_prompt_pattern",
            Self::Unknown => "unknown",
        }
    }
}

/// Risk posture.
///
/// `LowRisk` is present for serialization completeness, but
/// `derive_orchestration_policy` represents low risk as an empty `risk_policy`
/// vector. That keeps positive risk signals explicit and avoids persisting
/// redundant "nothing happened" rows for ordinary requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RiskPolicy {
    /// Serialization-only label; not emitted by the deterministic derivation.
    LowRisk,
    /// Destructive behavior requires confirmation or blocking.
    DestructiveRequiresConfirmation,
    /// Secret-bearing capture is allowed because raw capture is enabled.
    SecretCaptureAllowed,
    /// Raw capture is enabled for this environment/request path.
    RawCaptureEnabled,
    /// Current external information is required.
    ExternalWebRequired,
    /// High-stakes guardrails apply.
    HighStakesGuardrail,
    /// Fallback label for corrupt, missing, or future values.
    #[default]
    Unknown,
}

impl RiskPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LowRisk => "low_risk",
            Self::DestructiveRequiresConfirmation => "destructive_requires_confirmation",
            Self::SecretCaptureAllowed => "secret_capture_allowed",
            Self::RawCaptureEnabled => "raw_capture_enabled",
            Self::ExternalWebRequired => "external_web_required",
            Self::HighStakesGuardrail => "high_stakes_guardrail",
            Self::Unknown => "unknown",
        }
    }
}
