use deadpool_postgres::Pool;
use uuid::Uuid;

pub mod callback_normalization;
pub mod callbacks;
pub mod outcome;

pub use callbacks::{
    record_litellm_callback_payload, LiteLlmCallbackPayloadRequest, LiteLlmCallbackPayloadResponse,
    NormalizedLiteLlmCallbackFields,
};
pub use outcome::{
    baseline_arm_from_header, validate_outcome_request, BaselineArm, HarnessOutcomeRequest,
    HarnessOutcomeResponse, HumanAttentionBucket, OutcomeLabelerType, BASELINE_ARM_HEADER,
    DEFAULT_BASELINE_ARM,
};

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
