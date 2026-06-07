use deadpool_postgres::Pool;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

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
