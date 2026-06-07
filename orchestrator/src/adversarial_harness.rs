use deadpool_postgres::Pool;
use serde::{Deserialize, Serialize};
use serde_json::Value;
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

#[derive(Debug, Deserialize)]
pub struct LiteLlmCallbackPayloadRequest {
    pub attempt_id: Option<Uuid>,
    pub trajectory_id: Option<Uuid>,
    pub callback_type: String,
    pub payload: Value,
}

#[derive(Debug, Serialize)]
pub struct LiteLlmCallbackPayloadResponse {
    pub captured: bool,
    pub callback_payload_id: Uuid,
    pub normalized_ledger: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct NormalizedLiteLlmCallbackFields {
    pub litellm_call_id: Option<String>,
    pub model_group: Option<String>,
    pub deployment_id: Option<String>,
    pub api_key_alias: Option<String>,
    pub response_cost: Option<f64>,
    pub key_spend: Option<f64>,
    pub fallback_depth: Option<i32>,
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

pub async fn record_litellm_callback_payload(
    pool: &Pool,
    req: &LiteLlmCallbackPayloadRequest,
) -> Result<(Uuid, bool), anyhow::Error> {
    if req.callback_type.trim().is_empty() {
        anyhow::bail!("callback_type is required");
    }
    let callback_payload_id = Uuid::new_v4();
    let normalized = normalize_litellm_callback_fields(&req.payload);
    let conn = pool.get().await?;
    conn.execute(
        "INSERT INTO litellm_callback_payloads
         (callback_payload_id, attempt_id, trajectory_id, callback_type, payload)
         VALUES ($1, $2, $3, $4, $5)",
        &[
            &callback_payload_id,
            &req.attempt_id,
            &req.trajectory_id,
            &req.callback_type,
            &req.payload,
        ],
    )
    .await?;

    let normalized_ledger = if let Some(attempt_id) = req.attempt_id {
        let updated = conn
            .execute(
                "UPDATE litellm_call_ledger
                 SET litellm_call_id = COALESCE($2, litellm_call_id),
                     model_group = COALESCE($3, model_group),
                     deployment_id = COALESCE($4, deployment_id),
                     api_key_alias = COALESCE($5, api_key_alias),
                     response_cost = COALESCE($6, response_cost),
                     key_spend = COALESCE($7, key_spend),
                     fallback_depth = COALESCE($8, fallback_depth),
                     raw_callback_payload_id = $9
                 WHERE attempt_id = $1",
                &[
                    &attempt_id,
                    &normalized.litellm_call_id,
                    &normalized.model_group,
                    &normalized.deployment_id,
                    &normalized.api_key_alias,
                    &normalized.response_cost,
                    &normalized.key_spend,
                    &normalized.fallback_depth,
                    &callback_payload_id,
                ],
            )
            .await?;
        updated > 0
    } else {
        false
    };

    Ok((callback_payload_id, normalized_ledger))
}

pub fn normalize_litellm_callback_fields(payload: &Value) -> NormalizedLiteLlmCallbackFields {
    NormalizedLiteLlmCallbackFields {
        litellm_call_id: string_path_any(
            payload,
            &[
                &["litellm_call_id"],
                &["call_id"],
                &["id"],
                &["kwargs", "litellm_call_id"],
                &["standard_logging_object", "id"],
            ],
        ),
        model_group: string_path_any(
            payload,
            &[
                &["model_group"],
                &["model_group_alias"],
                &["kwargs", "model_group"],
                &["standard_logging_object", "model_group"],
            ],
        ),
        deployment_id: string_path_any(
            payload,
            &[
                &["deployment_id"],
                &["litellm_params", "metadata", "deployment_id"],
                &["standard_logging_object", "metadata", "deployment_id"],
            ],
        ),
        api_key_alias: string_path_any(
            payload,
            &[
                &["api_key_alias"],
                &["key_alias"],
                &["standard_logging_object", "metadata", "user_api_key_alias"],
            ],
        ),
        response_cost: f64_path_any(
            payload,
            &[
                &["response_cost"],
                &["cost"],
                &["standard_logging_object", "response_cost"],
            ],
        ),
        key_spend: f64_path_any(
            payload,
            &[
                &["key_spend"],
                &["spend"],
                &["standard_logging_object", "metadata", "user_api_key_spend"],
            ],
        ),
        fallback_depth: i32_path_any(
            payload,
            &[
                &["fallback_depth"],
                &["num_retries"],
                &["standard_logging_object", "metadata", "fallback_depth"],
            ],
        ),
    }
}

fn string_path_any(payload: &Value, paths: &[&[&str]]) -> Option<String> {
    paths
        .iter()
        .find_map(|path| value_at_path(payload, path).and_then(Value::as_str))
        .map(str::to_string)
        .filter(|value| !value.trim().is_empty())
}

fn f64_path_any(payload: &Value, paths: &[&[&str]]) -> Option<f64> {
    paths
        .iter()
        .find_map(|path| value_at_path(payload, path).and_then(Value::as_f64))
}

fn i32_path_any(payload: &Value, paths: &[&[&str]]) -> Option<i32> {
    paths
        .iter()
        .find_map(|path| value_at_path(payload, path).and_then(Value::as_i64))
        .and_then(|value| i32::try_from(value).ok())
}

fn value_at_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for segment in path {
        current = current.get(*segment)?;
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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

    #[test]
    fn callback_normalizer_extracts_known_litellm_fields() {
        let payload = json!({
            "standard_logging_object": {
                "id": "call-123",
                "model_group": "glm-4.5-air",
                "response_cost": 0.012,
                "metadata": {
                    "deployment_id": "deploy-a",
                    "user_api_key_alias": "eval-key",
                    "user_api_key_spend": 3.5,
                    "fallback_depth": 2
                }
            }
        });
        let fields = normalize_litellm_callback_fields(&payload);
        assert_eq!(fields.litellm_call_id.as_deref(), Some("call-123"));
        assert_eq!(fields.model_group.as_deref(), Some("glm-4.5-air"));
        assert_eq!(fields.deployment_id.as_deref(), Some("deploy-a"));
        assert_eq!(fields.api_key_alias.as_deref(), Some("eval-key"));
        assert_eq!(fields.response_cost, Some(0.012));
        assert_eq!(fields.key_spend, Some(3.5));
        assert_eq!(fields.fallback_depth, Some(2));
    }
}
