use serde_json::Value;

use crate::feature_metadata_paths::string_path;

pub(crate) fn summarizer_shares_litellm_upstream(metadata: &Value) -> bool {
    let Some(summarizer_url) = string_path(metadata, &["summarizer_base_url"])
        .or_else(|| string_path(metadata, &["payload", "summarizer_base_url"]))
    else {
        return false;
    };
    let Some(litellm_url) = string_path(metadata, &["litellm_url"])
        .or_else(|| string_path(metadata, &["payload", "litellm_url"]))
        .or_else(|| string_path(metadata, &["upstream_litellm_url"]))
        .or_else(|| string_path(metadata, &["payload", "upstream_litellm_url"]))
    else {
        return false;
    };
    normalize_url_for_compare(summarizer_url) == normalize_url_for_compare(litellm_url)
}

pub(crate) fn summarizer_has_dedicated_upstream(metadata: &Value) -> bool {
    let Some(summarizer_url) = string_path(metadata, &["summarizer_base_url"])
        .or_else(|| string_path(metadata, &["payload", "summarizer_base_url"]))
    else {
        return false;
    };
    let Some(litellm_url) = string_path(metadata, &["litellm_url"])
        .or_else(|| string_path(metadata, &["payload", "litellm_url"]))
        .or_else(|| string_path(metadata, &["upstream_litellm_url"]))
        .or_else(|| string_path(metadata, &["payload", "upstream_litellm_url"]))
    else {
        return false;
    };
    normalize_url_for_compare(summarizer_url) != normalize_url_for_compare(litellm_url)
}

fn normalize_url_for_compare(value: &str) -> String {
    value.trim().trim_end_matches('/').to_ascii_lowercase()
}
