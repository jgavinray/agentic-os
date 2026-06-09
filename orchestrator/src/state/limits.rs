/// Default max_tokens for proxied completion requests when the client omits it.
pub const DEFAULT_MAX_TOKENS: u64 = 8192;

/// Maximum max_tokens this backend can safely serve for Claude Code traffic.
pub const MAX_MAX_TOKENS: u64 = 32768;

/// Default concurrency for best-effort derived background work.
pub const DEFAULT_BACKGROUND_WORK_CONCURRENCY: usize = 4;

pub fn configured_default_max_tokens() -> u64 {
    std::env::var("DEFAULT_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .max(1)
}

pub fn configured_max_max_tokens() -> u64 {
    std::env::var("MAX_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(MAX_MAX_TOKENS)
        .max(1)
}
