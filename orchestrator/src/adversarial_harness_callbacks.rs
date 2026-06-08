use deadpool_postgres::Pool;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub use crate::adversarial_harness_callback_normalization::{
    normalize_litellm_callback_fields, NormalizedLiteLlmCallbackFields,
};

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
