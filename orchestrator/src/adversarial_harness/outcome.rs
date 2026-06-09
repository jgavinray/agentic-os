use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const BASELINE_ARM_HEADER: &str = "x-agent-baseline-arm";
pub const DEFAULT_BASELINE_ARM: BaselineArm = BaselineArm::OrchestratorPolicyEnabled;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BaselineArm {
    RawVllmViaLitellm,
    OrchestratorPolicyDisabled,
    OrchestratorPolicyEnabled,
    OrchestratorPolicyPlusObservationalReview,
    FrontierEscalation,
    Unknown,
}

impl BaselineArm {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RawVllmViaLitellm => "raw_vllm_via_litellm",
            Self::OrchestratorPolicyDisabled => "orchestrator_policy_disabled",
            Self::OrchestratorPolicyEnabled => "orchestrator_policy_enabled",
            Self::OrchestratorPolicyPlusObservationalReview => {
                "orchestrator_policy_plus_observational_review"
            }
            Self::FrontierEscalation => "frontier_escalation",
            Self::Unknown => "unknown",
        }
    }

    pub fn from_label(value: &str) -> Option<Self> {
        Some(match value.trim() {
            "raw_vllm_via_litellm" => Self::RawVllmViaLitellm,
            "orchestrator_policy_disabled" => Self::OrchestratorPolicyDisabled,
            "orchestrator_policy_enabled" => Self::OrchestratorPolicyEnabled,
            "orchestrator_policy_plus_observational_review" => {
                Self::OrchestratorPolicyPlusObservationalReview
            }
            "frontier_escalation" => Self::FrontierEscalation,
            "unknown" => Self::Unknown,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanAttentionBucket {
    None,
    Low,
    Medium,
    High,
    Unknown,
}

impl HumanAttentionBucket {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeLabelerType {
    EvalOperator,
    ClientHook,
    Human,
    Posthoc,
}

impl OutcomeLabelerType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EvalOperator => "eval_operator",
            Self::ClientHook => "client_hook",
            Self::Human => "human",
            Self::Posthoc => "posthoc",
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct HarnessOutcomeRequest {
    pub trajectory_id: Uuid,
    #[serde(default)]
    pub source_event_ids: Vec<Uuid>,
    pub baseline_arm: BaselineArm,
    pub accepted: bool,
    #[serde(default)]
    pub invalid_output: bool,
    #[serde(default)]
    pub manual_prompt_edits_required: i32,
    pub human_attention_bucket: HumanAttentionBucket,
    pub labeler_type: OutcomeLabelerType,
    pub notes: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct HarnessOutcomeResponse {
    pub captured: bool,
    pub outcome_event_id: Uuid,
}

pub fn baseline_arm_from_header(value: Option<&str>) -> Result<BaselineArm, String> {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        Some(value) => {
            BaselineArm::from_label(value).ok_or_else(|| format!("invalid baseline arm `{value}`"))
        }
        None => Ok(DEFAULT_BASELINE_ARM),
    }
}

pub fn validate_outcome_request(req: &HarnessOutcomeRequest) -> Result<(), String> {
    if req.manual_prompt_edits_required < 0 {
        return Err("manual_prompt_edits_required must be non-negative".to_string());
    }
    if req.invalid_output && req.accepted {
        return Err("accepted outcomes cannot also be invalid_output".to_string());
    }
    Ok(())
}
