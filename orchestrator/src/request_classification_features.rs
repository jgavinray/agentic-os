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
        || contains_any(lower, &[" json ", ".json", "application/json"]);
    let has_yaml = has_code_block && (lower.contains("```yaml") || lower.contains("```yml"))
        || contains_any(lower, &[" yaml", ".yaml", ".yml", "apiVersion:"])
        || looks_like_yaml(text);
    let has_stack_trace = contains_any(
        lower,
        &[
            "traceback",
            "stack trace",
            "stack backtrace",
            "panicked at",
            "caused by:",
        ],
    );
    let has_logs = contains_any(
        lower,
        &[
            "\"level\":\"error\"",
            "\"level\":\"warn\"",
            " level=error",
            " level=warn",
            "[error]",
            "[warn]",
            "timestamp",
        ],
    );
    let has_shell_command = contains_any(
        lower,
        &[
            "docker compose",
            "kubectl ",
            "curl ",
            "cargo test",
            "git ",
            "psql ",
            "sudo ",
            "npm ",
            "yarn ",
            "rm -",
            "$ ",
        ],
    );
    let has_url = contains_any(lower, &["http://", "https://", "localhost:", "127.0.0.1"]);
    let has_file_path = has_url
        || contains_any(
            lower,
            &[
                "/users/",
                "/tmp/",
                "src/",
                "docs/",
                ".rs",
                ".py",
                ".ts",
                ".tsx",
                ".js",
                ".json",
                ".yaml",
                ".yml",
                ".md",
                "cargo.toml",
                "dockerfile",
            ],
        );
    let has_secret_candidate = contains_any(
        lower,
        &[
            "authorization: bearer",
            "api_key",
            "apikey",
            "access_token",
            "secret=",
            "password=",
            "sk-",
            "aws_secret_access_key",
        ],
    ) || contains_any(
        metadata_keys_lower,
        &[
            "authorization",
            "api_key",
            "apikey",
            "access_token",
            "secret",
            "password",
        ],
    );
    let contains_error_words = contains_any(
        lower,
        &[
            "error",
            "failed",
            "failure",
            "panic",
            "exception",
            "timeout",
            "refused",
            "invalid",
        ],
    );
    let contains_destructive_verbs = contains_any(
        lower,
        &[
            "rm -rf",
            "drop table",
            "delete from",
            "truncate table",
            "kubectl delete",
            "docker rm",
            "destroy",
            "wipe",
            "format disk",
            "git reset --hard",
        ],
    );
    let asks_for_latest = contains_any(
        lower,
        &[
            "latest",
            "current",
            "today",
            "right now",
            "up to date",
            "up-to-date",
            "look up",
            "search the web",
        ],
    );
    let asks_for_file_generation = contains_any(
        lower,
        &[
            "create a file",
            "write a file",
            "generate a file",
            "save this",
            "draft a commit",
            "produce a patch",
        ],
    );
    let has_kubernetes_terms = contains_any(
        lower,
        &[
            "kubernetes",
            "kubectl",
            "k8s",
            "pod",
            "deployment",
            "namespace",
            "helm",
        ],
    );
    let has_docker_terms = contains_any(
        lower,
        &[
            "docker",
            "dockerfile",
            "compose.yaml",
            "compose.yml",
            "container",
            "image",
        ],
    );
    let has_llm_terms = contains_any(
        lower,
        &[
            "llm",
            "prompt",
            "tokens",
            "inference",
            "vllm",
            "qwen",
            "claude",
            "openai",
            "litellm",
            "model",
        ],
    );
    let has_networking_terms = contains_any(
        lower,
        &[
            "dns",
            "tcp",
            "http",
            "https",
            "ingress",
            "proxy",
            "tls",
            "port ",
            "localhost",
        ],
    );
    let has_security_terms = contains_any(
        lower,
        &[
            "auth",
            "token",
            "secret",
            "password",
            "jwt",
            "exploit",
            "vulnerability",
            "csrf",
            "xss",
        ],
    );
    let has_config_shape = has_yaml
        || has_json
        || contains_any(
            lower,
            &[
                "config",
                "configuration",
                "env var",
                "environment variable",
                ".env",
                "compose.yaml",
                "values.yaml",
            ],
        );
    let has_diff_or_patch = contains_any(
        lower,
        &[
            "diff --git",
            "--- a/",
            "+++ b/",
            "@@",
            "apply_patch",
            "patch",
        ],
    );
    let has_test_failure = contains_any(
        lower,
        &[
            "test failed",
            "tests failed",
            "failures:",
            "assertion failed",
            "cargo test",
            "pytest",
            "expected:",
        ],
    );

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
