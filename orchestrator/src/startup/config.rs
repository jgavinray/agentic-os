use std::collections::HashSet;
use std::env;

use crate::startup::env as startup_env;
use crate::{feature_extraction, sampling, state};

pub struct StartupConfig {
    pub db_url: String,
    pub capture_db_url: Option<String>,
    pub qdrant_url: String,
    pub litellm_url: String,
    pub litellm_key: String,
    pub api_keys: Vec<(String, String)>,
    pub default_model: String,
    pub summarizer_url: String,
    pub summarizer_key: Option<String>,
    pub summarizer_model: String,
    pub total_recall_url: Option<String>,
    pub vllm_metrics_url: Option<String>,
    pub summarizer_max_tokens: u64,
    pub default_task: String,
    pub cache_ttl_ms: u64,
    pub context_decay_rate: f64,
    pub rate_limit_per_minute: u32,
    pub rate_limit_burst: u32,
    pub execution_feedback_enabled: bool,
    pub trajectory_capture_enabled: bool,
    pub trajectory_idle_timeout_sec: u64,
    pub sampling_config: sampling::SamplingConfig,
    pub failure_history_token_budget: usize,
    pub feature_extraction_enabled: bool,
    pub tool_mediation_enabled: bool,
    pub classification_routing_enabled: bool,
    pub route_model_small: Option<String>,
    pub route_model_strong: Option<String>,
    pub prefix_cache_canary_enabled: bool,
    pub prefix_cache_canary_namespace_allowlist: HashSet<String>,
    pub summarizer_enabled: bool,
    pub operational_constraints_token_budget: usize,
    pub background_work_concurrency: usize,
    pub litellm_request_timeout_secs: u64,
    pub embed_model_path: String,
}

pub fn load_startup_config() -> Result<StartupConfig, anyhow::Error> {
    let db_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let capture_db_url = startup_env::optional_trimmed_env("CAPTURE_DATABASE_URL");
    let qdrant_url = env::var("QDRANT_URL").expect("QDRANT_URL must be set");
    let litellm_url = env::var("LITELLM_URL")
        .expect("LITELLM_URL must be set")
        .trim_end_matches('/')
        .to_string();
    let litellm_key = env::var("LITELLM_KEY").expect("LITELLM_KEY must be set");
    // BUG-12: canonical model name matches litessh-prompt.md
    let default_model =
        env::var("DEFAULT_MODEL").unwrap_or_else(|_| "qwen36-35b-heretic".to_string());
    let summarizer_url = env::var("SUMMARIZER_BASE_URL")
        .unwrap_or_else(|_| litellm_url.clone())
        .trim_end_matches('/')
        .to_string();
    let summarizer_key = startup_env::optional_trimmed_env("SUMMARIZER_KEY").or_else(|| {
        if summarizer_url == litellm_url {
            Some(litellm_key.clone())
        } else {
            None
        }
    });

    Ok(StartupConfig {
        db_url,
        capture_db_url,
        qdrant_url,
        litellm_url,
        litellm_key,
        api_keys: startup_env::api_keys_from_env(),
        default_model: default_model.clone(),
        summarizer_url,
        summarizer_key,
        summarizer_model: env::var("SUMMARIZER_MODEL").unwrap_or_else(|_| default_model.clone()),
        total_recall_url: startup_env::optional_trimmed_url_env("TOTAL_RECALL_URL"),
        vllm_metrics_url: startup_env::optional_trimmed_env("VLLM_METRICS_URL"),
        summarizer_max_tokens: startup_env::parsed_env_min(
            "SUMMARIZER_MAX_TOKENS",
            state::SUMMARIZER_MAX_TOKENS,
            1,
        ),
        default_task: env::var("DEFAULT_TASK").unwrap_or_else(|_| "engineering".to_string()),
        cache_ttl_ms: startup_env::parsed_env("CONTEXT_CACHE_TTL_MS", state::CONTEXT_CACHE_TTL_MS),
        context_decay_rate: startup_env::parsed_env(
            "CONTEXT_DECAY_RATE",
            state::DEFAULT_CONTEXT_DECAY_RATE,
        ),
        rate_limit_per_minute: startup_env::parsed_env("RATE_LIMIT_PER_MINUTE", 60),
        rate_limit_burst: startup_env::parsed_env("RATE_LIMIT_BURST", 30),
        execution_feedback_enabled: startup_env::bool_env("EXECUTION_FEEDBACK_ENABLED", true),
        trajectory_capture_enabled: startup_env::bool_env("TRAJECTORY_CAPTURE_ENABLED", true),
        trajectory_idle_timeout_sec: startup_env::parsed_env(
            "TRAJECTORY_IDLE_TIMEOUT_SEC",
            crate::trajectory::DEFAULT_TRAJECTORY_IDLE_TIMEOUT_SEC,
        ),
        sampling_config: sampling::SamplingConfig::from_env()?,
        failure_history_token_budget: startup_env::parsed_env(
            "FAILURE_HISTORY_TOKEN_BUDGET",
            state::DEFAULT_FAILURE_HISTORY_TOKEN_BUDGET,
        ),
        feature_extraction_enabled: feature_extraction::feature_extraction_enabled_from_env(),
        tool_mediation_enabled: startup_env::bool_env("TOOL_MEDIATION_ENABLED", true),
        classification_routing_enabled: startup_env::bool_env(
            "CLASSIFICATION_ROUTING_ENABLED",
            false,
        ),
        route_model_small: startup_env::optional_trimmed_env("ROUTE_MODEL_SMALL"),
        route_model_strong: startup_env::optional_trimmed_env("ROUTE_MODEL_STRONG"),
        prefix_cache_canary_enabled: startup_env::bool_env("PREFIX_CACHE_CANARY_ENABLED", false),
        prefix_cache_canary_namespace_allowlist: startup_env::comma_list_env(
            "PREFIX_CACHE_CANARY_NAMESPACE_ALLOWLIST",
        ),
        summarizer_enabled: startup_env::bool_env("SUMMARIZER_ENABLED", true),
        operational_constraints_token_budget:
            feature_extraction::operational_constraints_token_budget_from_env(),
        background_work_concurrency: startup_env::parsed_env_min(
            "BACKGROUND_WORK_CONCURRENCY",
            state::DEFAULT_BACKGROUND_WORK_CONCURRENCY,
            1,
        ),
        litellm_request_timeout_secs: startup_env::parsed_env_min(
            "LITELLM_REQUEST_TIMEOUT_SECS",
            300,
            1,
        ),
        embed_model_path: env::var("EMBED_MODEL_PATH").expect("EMBED_MODEL_PATH must be set"),
    })
}
