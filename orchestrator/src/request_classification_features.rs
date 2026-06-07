use crate::request_classification_feature_vocab as vocab;

#[derive(Debug, Clone, Copy)]
pub(crate) struct RequestFeatures {
    pub(crate) char_count: usize,
    pub(crate) line_count: usize,
    pub(crate) estimated_tokens: usize,
    pub(crate) has_code_block: bool,
    pub(crate) has_yaml: bool,
    pub(crate) has_json: bool,
    pub(crate) has_stack_trace: bool,
    pub(crate) has_logs: bool,
    pub(crate) has_shell_command: bool,
    pub(crate) has_url: bool,
    pub(crate) has_file_path: bool,
    pub(crate) has_secret_candidate: bool,
    pub(crate) contains_error_words: bool,
    pub(crate) contains_destructive_verbs: bool,
    pub(crate) asks_for_latest: bool,
    pub(crate) asks_for_file_generation: bool,
    pub(crate) has_kubernetes_terms: bool,
    pub(crate) has_docker_terms: bool,
    pub(crate) has_llm_terms: bool,
    pub(crate) has_networking_terms: bool,
    pub(crate) has_security_terms: bool,
    pub(crate) has_config_shape: bool,
    pub(crate) has_diff_or_patch: bool,
    pub(crate) has_test_failure: bool,
}

pub(crate) fn extract_features(
    text: &str,
    lower: &str,
    metadata_keys_lower: &str,
) -> RequestFeatures {
    let trimmed = text.trim();
    let char_count = text.chars().count();
    let line_count = if trimmed.is_empty() {
        0
    } else {
        text.lines().count()
    };

    let has_code_block = text.contains("```");
    let has_json = has_code_block && lower.contains("```json")
        || trimmed.starts_with('{') && trimmed.ends_with('}')
        || contains_any(lower, vocab::JSON_MARKERS);
    let has_yaml = has_code_block && (lower.contains("```yaml") || lower.contains("```yml"))
        || contains_any(lower, vocab::YAML_MARKERS)
        || looks_like_yaml(text);
    let has_stack_trace = contains_any(lower, vocab::STACK_TRACE_MARKERS);
    let has_logs = contains_any(lower, vocab::LOG_MARKERS);
    let has_shell_command = contains_any(lower, vocab::SHELL_COMMAND_MARKERS);
    let has_url = contains_any(lower, vocab::URL_MARKERS);
    let has_file_path = has_url || contains_any(lower, vocab::FILE_PATH_MARKERS);
    let has_secret_candidate = contains_any(lower, vocab::SECRET_TEXT_MARKERS)
        || contains_any(metadata_keys_lower, vocab::SECRET_METADATA_MARKERS);
    let contains_error_words = contains_any(lower, vocab::ERROR_WORDS);
    let contains_destructive_verbs = contains_any(lower, vocab::DESTRUCTIVE_VERBS);
    let asks_for_latest = contains_any(lower, vocab::LATEST_MARKERS);
    let asks_for_file_generation = contains_any(lower, vocab::FILE_GENERATION_MARKERS);
    let has_kubernetes_terms = contains_any(lower, vocab::KUBERNETES_TERMS);
    let has_docker_terms = contains_any(lower, vocab::DOCKER_TERMS);
    let has_llm_terms = contains_any(lower, vocab::LLM_TERMS);
    let has_networking_terms = contains_any(lower, vocab::NETWORKING_TERMS);
    let has_security_terms = contains_any(lower, vocab::SECURITY_TERMS);
    let has_config_shape = has_yaml || has_json || contains_any(lower, vocab::CONFIG_TERMS);
    let has_diff_or_patch = contains_any(lower, vocab::DIFF_OR_PATCH_MARKERS);
    let has_test_failure = contains_any(lower, vocab::TEST_FAILURE_MARKERS);

    RequestFeatures {
        char_count,
        line_count,
        estimated_tokens: estimate_tokens(char_count),
        has_code_block,
        has_yaml,
        has_json,
        has_stack_trace,
        has_logs,
        has_shell_command,
        has_url,
        has_file_path,
        has_secret_candidate,
        contains_error_words,
        contains_destructive_verbs,
        asks_for_latest,
        asks_for_file_generation,
        has_kubernetes_terms,
        has_docker_terms,
        has_llm_terms,
        has_networking_terms,
        has_security_terms,
        has_config_shape,
        has_diff_or_patch,
        has_test_failure,
    }
}

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

fn looks_like_yaml(text: &str) -> bool {
    let mut yamlish_lines = 0;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        if trimmed.contains(": ") || trimmed.starts_with("- ") {
            yamlish_lines += 1;
        }
    }
    yamlish_lines >= 2
}

fn estimate_tokens(char_count: usize) -> usize {
    if char_count == 0 {
        0
    } else {
        char_count.div_ceil(4)
    }
}
