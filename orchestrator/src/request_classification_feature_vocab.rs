pub(crate) const JSON_MARKERS: &[&str] = &[" json ", ".json", "application/json"];
pub(crate) const YAML_MARKERS: &[&str] = &[" yaml", ".yaml", ".yml", "apiVersion:"];
pub(crate) const STACK_TRACE_MARKERS: &[&str] = &[
    "traceback",
    "stack trace",
    "stack backtrace",
    "panicked at",
    "caused by:",
];
pub(crate) const LOG_MARKERS: &[&str] = &[
    "\"level\":\"error\"",
    "\"level\":\"warn\"",
    " level=error",
    " level=warn",
    "[error]",
    "[warn]",
    "timestamp",
];
pub(crate) const SHELL_COMMAND_MARKERS: &[&str] = &[
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
];
pub(crate) const URL_MARKERS: &[&str] = &["http://", "https://", "localhost:", "127.0.0.1"];
pub(crate) const FILE_PATH_MARKERS: &[&str] = &[
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
];
pub(crate) const SECRET_TEXT_MARKERS: &[&str] = &[
    "authorization: bearer",
    "api_key",
    "apikey",
    "access_token",
    "secret=",
    "password=",
    "sk-",
    "aws_secret_access_key",
];
pub(crate) const SECRET_METADATA_MARKERS: &[&str] = &[
    "authorization",
    "api_key",
    "apikey",
    "access_token",
    "secret",
    "password",
];
pub(crate) const ERROR_WORDS: &[&str] = &[
    "error",
    "failed",
    "failure",
    "panic",
    "exception",
    "timeout",
    "refused",
    "invalid",
];
pub(crate) const DESTRUCTIVE_VERBS: &[&str] = &[
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
];
pub(crate) const LATEST_MARKERS: &[&str] = &[
    "latest",
    "current",
    "today",
    "right now",
    "up to date",
    "up-to-date",
    "look up",
    "search the web",
];
pub(crate) const FILE_GENERATION_MARKERS: &[&str] = &[
    "create a file",
    "write a file",
    "generate a file",
    "save this",
    "draft a commit",
    "produce a patch",
];
pub(crate) const KUBERNETES_TERMS: &[&str] = &[
    "kubernetes",
    "kubectl",
    "k8s",
    "pod",
    "deployment",
    "namespace",
    "helm",
];
pub(crate) const DOCKER_TERMS: &[&str] = &[
    "docker",
    "dockerfile",
    "compose.yaml",
    "compose.yml",
    "container",
    "image",
];
pub(crate) const LLM_TERMS: &[&str] = &[
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
];
pub(crate) const NETWORKING_TERMS: &[&str] = &[
    "dns",
    "tcp",
    "http",
    "https",
    "ingress",
    "proxy",
    "tls",
    "port ",
    "localhost",
];
pub(crate) const SECURITY_TERMS: &[&str] = &[
    "auth",
    "token",
    "secret",
    "password",
    "jwt",
    "exploit",
    "vulnerability",
    "csrf",
    "xss",
];
pub(crate) const CONFIG_TERMS: &[&str] = &[
    "config",
    "configuration",
    "env var",
    "environment variable",
    ".env",
    "compose.yaml",
    "values.yaml",
];
pub(crate) const DIFF_OR_PATCH_MARKERS: &[&str] = &[
    "diff --git",
    "--- a/",
    "+++ b/",
    "@@",
    "apply_patch",
    "patch",
];
pub(crate) const TEST_FAILURE_MARKERS: &[&str] = &[
    "test failed",
    "tests failed",
    "failures:",
    "assertion failed",
    "cargo test",
    "pytest",
    "expected:",
];
