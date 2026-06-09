use std::collections::HashSet;
use std::env;
use std::str::FromStr;

pub fn bool_env(name: &str, default: bool) -> bool {
    env::var(name)
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(default)
}

pub fn optional_trimmed_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub fn optional_trimmed_url_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().trim_end_matches('/').to_string())
        .filter(|value| !value.is_empty())
}

pub fn parsed_env<T>(name: &str, default: T) -> T
where
    T: FromStr,
{
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

pub fn parsed_env_min<T>(name: &str, default: T, min: T) -> T
where
    T: FromStr + Ord,
{
    parsed_env(name, default).max(min)
}

pub fn api_keys_from_env() -> Vec<(String, String)> {
    // API_KEYS is semicolon-delimited entries: `token,namespace;token2,namespace2`.
    env::var("API_KEYS")
        .unwrap_or_else(|_| "agent-os,agentic-os".to_string())
        .split(';')
        .map(|s| {
            let s = s.trim();
            let mut parts = s.splitn(2, ',');
            let token = parts.next().unwrap_or(s).trim().to_string();
            let namespace = parts.next().unwrap_or(&token).trim().to_string();
            (token, namespace)
        })
        .filter(|(token, _)| !token.is_empty())
        .collect()
}

pub fn comma_list_env(name: &str) -> HashSet<String> {
    env::var(name)
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}
