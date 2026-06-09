use crate::request_classification_features::RequestFeatures;
use crate::request_classification_rule_utils::{contains_any, push_if};
use crate::request_classification_types::{RequestArtifactType, RequestDomain};

pub(crate) fn detected_domains(features: &RequestFeatures, lower: &str) -> Vec<RequestDomain> {
    let mut domains = Vec::new();
    push_if(
        &mut domains,
        features.has_kubernetes_terms,
        RequestDomain::Kubernetes,
    );
    push_if(
        &mut domains,
        features.has_docker_terms,
        RequestDomain::Docker,
    );
    push_if(
        &mut domains,
        features.has_llm_terms,
        RequestDomain::LlmInference,
    );
    push_if(
        &mut domains,
        features.has_networking_terms,
        RequestDomain::Networking,
    );
    push_if(
        &mut domains,
        features.has_security_terms,
        RequestDomain::Security,
    );
    push_if(
        &mut domains,
        contains_any(lower, &["medical", "doctor", "diagnosis", "medicine"]),
        RequestDomain::Medical,
    );
    push_if(
        &mut domains,
        contains_any(lower, &["legal", "lawyer", "lawsuit", "contract"]),
        RequestDomain::Legal,
    );
    push_if(
        &mut domains,
        contains_any(lower, &["finance", "tax", "investment", "stock", "loan"]),
        RequestDomain::Finance,
    );
    push_if(
        &mut domains,
        features.has_shell_command,
        RequestDomain::Shell,
    );
    domains
}

pub(crate) fn classify_domain(
    features: &RequestFeatures,
    lower: &str,
    detected_domains: &[RequestDomain],
) -> RequestDomain {
    for high_stakes in [
        RequestDomain::Medical,
        RequestDomain::Legal,
        RequestDomain::Finance,
    ] {
        if detected_domains.contains(&high_stakes) {
            return high_stakes;
        }
    }
    if features.char_count == 0 {
        RequestDomain::Unknown
    } else if let Some(domain) = detected_domains.first() {
        *domain
    } else if contains_any(lower, &["shell", "terminal", "bash", "zsh"]) {
        RequestDomain::Shell
    } else {
        RequestDomain::Generic
    }
}

pub(crate) fn classify_artifact(features: &RequestFeatures, lower: &str) -> RequestArtifactType {
    if features.char_count == 0 {
        RequestArtifactType::Unknown
    } else if features.has_logs || features.has_stack_trace || features.has_test_failure {
        RequestArtifactType::Logs
    } else if features.has_json {
        RequestArtifactType::Json
    } else if features.has_yaml {
        RequestArtifactType::Yaml
    } else if contains_any(
        lower,
        &["sql", "select ", "insert into", "update ", "delete from"],
    ) {
        RequestArtifactType::Sql
    } else if contains_any(lower, &["markdown", ".md", "# "]) {
        RequestArtifactType::Markdown
    } else if features.has_code_block || features.has_diff_or_patch {
        RequestArtifactType::Code
    } else if contains_any(lower, &["image", "screenshot", ".png", ".jpg", ".jpeg"]) {
        RequestArtifactType::Image
    } else if features.has_file_path || features.asks_for_file_generation {
        RequestArtifactType::File
    } else {
        RequestArtifactType::PlainText
    }
}
