use deadpool_postgres::Pool;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use crate::adversarial_harness_callbacks::{
    record_litellm_callback_payload, LiteLlmCallbackPayloadRequest, LiteLlmCallbackPayloadResponse,
    NormalizedLiteLlmCallbackFields,
};

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

pub async fn record_harness_outcome(
    pool: &Pool,
    req: &HarnessOutcomeRequest,
) -> Result<Uuid, anyhow::Error> {
    validate_outcome_request(req).map_err(anyhow::Error::msg)?;
    let outcome_event_id = Uuid::new_v4();
    let conn = pool.get().await?;
    conn.execute(
        "INSERT INTO harness_outcome_events
         (outcome_event_id, trajectory_id, source_event_ids, baseline_arm,
          accepted, invalid_output, manual_prompt_edits_required,
          human_attention_bucket, labeler_type, notes)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
        &[
            &outcome_event_id,
            &req.trajectory_id,
            &req.source_event_ids,
            &req.baseline_arm.as_str(),
            &req.accepted,
            &req.invalid_output,
            &req.manual_prompt_edits_required,
            &req.human_attention_bucket.as_str(),
            &req.labeler_type.as_str(),
            &req.notes,
        ],
    )
    .await?;
    Ok(outcome_event_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_arm_header_defaults_to_policy_enabled() {
        assert_eq!(
            baseline_arm_from_header(None).unwrap(),
            BaselineArm::OrchestratorPolicyEnabled
        );
        assert_eq!(
            baseline_arm_from_header(Some("raw_vllm_via_litellm")).unwrap(),
            BaselineArm::RawVllmViaLitellm
        );
        assert!(baseline_arm_from_header(Some("freeform-arm")).is_err());
    }

    #[test]
    fn outcome_rejects_negative_manual_edit_count() {
        let req = HarnessOutcomeRequest {
            trajectory_id: Uuid::new_v4(),
            source_event_ids: vec![],
            baseline_arm: BaselineArm::OrchestratorPolicyEnabled,
            accepted: false,
            invalid_output: false,
            manual_prompt_edits_required: -1,
            human_attention_bucket: HumanAttentionBucket::Low,
            labeler_type: OutcomeLabelerType::EvalOperator,
            notes: None,
        };
        assert!(validate_outcome_request(&req).is_err());
    }

    #[test]
    fn outcome_rejects_accepted_invalid_output() {
        let req = HarnessOutcomeRequest {
            trajectory_id: Uuid::new_v4(),
            source_event_ids: vec![],
            baseline_arm: BaselineArm::OrchestratorPolicyEnabled,
            accepted: true,
            invalid_output: true,
            manual_prompt_edits_required: 0,
            human_attention_bucket: HumanAttentionBucket::None,
            labeler_type: OutcomeLabelerType::EvalOperator,
            notes: None,
        };
        assert!(validate_outcome_request(&req).is_err());
    }
}
