use serde_json::Value;

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
