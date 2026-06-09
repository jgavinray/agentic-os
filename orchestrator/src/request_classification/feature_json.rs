use serde_json::{json, Map, Value};

use crate::request_classification_composition::CompositeAnalysis;
use crate::request_classification_features::RequestFeatures;
use crate::request_classification_types::RequestDomain;

pub(crate) fn features_to_json(
    features: &RequestFeatures,
    detected_domains: &[RequestDomain],
    composite: &CompositeAnalysis,
) -> Value {
    let mut object = Map::new();
    object.insert("char_count".to_string(), json!(features.char_count));
    object.insert("line_count".to_string(), json!(features.line_count));
    object.insert(
        "estimated_tokens".to_string(),
        json!(features.estimated_tokens),
    );
    object.insert("has_code_block".to_string(), json!(features.has_code_block));
    object.insert("has_yaml".to_string(), json!(features.has_yaml));
    object.insert("has_json".to_string(), json!(features.has_json));
    object.insert(
        "has_stack_trace".to_string(),
        json!(features.has_stack_trace),
    );
    object.insert("has_logs".to_string(), json!(features.has_logs));
    object.insert(
        "has_shell_command".to_string(),
        json!(features.has_shell_command),
    );
    object.insert("has_url".to_string(), json!(features.has_url));
    object.insert("has_file_path".to_string(), json!(features.has_file_path));
    object.insert(
        "has_secret_candidate".to_string(),
        json!(features.has_secret_candidate),
    );
    object.insert(
        "contains_error_words".to_string(),
        json!(features.contains_error_words),
    );
    object.insert(
        "contains_destructive_verbs".to_string(),
        json!(features.contains_destructive_verbs),
    );
    object.insert(
        "asks_for_latest".to_string(),
        json!(features.asks_for_latest),
    );
    object.insert(
        "asks_for_file_generation".to_string(),
        json!(features.asks_for_file_generation),
    );
    object.insert(
        "detected_domain_terms".to_string(),
        json!(detected_domains
            .iter()
            .map(|domain| domain.as_str())
            .collect::<Vec<_>>()),
    );
    object.insert(
        "has_kubernetes_terms".to_string(),
        json!(features.has_kubernetes_terms),
    );
    object.insert(
        "has_docker_terms".to_string(),
        json!(features.has_docker_terms),
    );
    object.insert("has_llm_terms".to_string(), json!(features.has_llm_terms));
    object.insert(
        "has_networking_terms".to_string(),
        json!(features.has_networking_terms),
    );
    object.insert(
        "has_security_terms".to_string(),
        json!(features.has_security_terms),
    );
    object.insert(
        "has_config_shape".to_string(),
        json!(features.has_config_shape),
    );
    object.insert(
        "has_diff_or_patch".to_string(),
        json!(features.has_diff_or_patch),
    );
    object.insert(
        "has_test_failure".to_string(),
        json!(features.has_test_failure),
    );
    object.insert("is_composite".to_string(), json!(composite.is_composite));
    object.insert(
        "decomposition_candidate".to_string(),
        json!(composite.decomposition_candidate),
    );
    object.insert("decomposition_reason".to_string(), json!(composite.reason));
    object.insert(
        "sub_intent_count".to_string(),
        json!(composite.sub_intents.len()),
    );
    object.insert(
        "sub_intents".to_string(),
        json!(composite
            .sub_intents
            .iter()
            .map(|intent| intent.as_str())
            .collect::<Vec<_>>()),
    );
    Value::Object(object)
}
