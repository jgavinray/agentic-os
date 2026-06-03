CREATE TABLE IF NOT EXISTS vllm_cache_observations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    session_id TEXT,
    namespace TEXT NOT NULL,
    repo TEXT NOT NULL,
    task TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    routed_model TEXT NOT NULL,
    request_event_id UUID,
    context_pack_id UUID,
    attempt_id UUID,
    metrics_url TEXT NOT NULL,
    prefix_cache_queries_delta BIGINT NOT NULL DEFAULT 0,
    prefix_cache_hits_delta BIGINT NOT NULL DEFAULT 0,
    prompt_tokens_total_delta BIGINT NOT NULL DEFAULT 0,
    prompt_tokens_cached_delta BIGINT NOT NULL DEFAULT 0,
    prompt_tokens_local_compute_delta BIGINT NOT NULL DEFAULT 0,
    prompt_tokens_local_cache_hit_delta BIGINT NOT NULL DEFAULT 0,
    prompt_tokens_external_kv_delta BIGINT NOT NULL DEFAULT 0,
    request_input_tokens BIGINT NOT NULL DEFAULT 0,
    request_output_tokens BIGINT NOT NULL DEFAULT 0,
    provider_cached_tokens BIGINT NOT NULL DEFAULT 0,
    provider_cache_created_tokens BIGINT NOT NULL DEFAULT 0,
    provider_cache_read_tokens BIGINT NOT NULL DEFAULT 0,
    kv_cache_usage_before DOUBLE PRECISION,
    kv_cache_usage_after DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS vllm_cache_observations_session_idx
    ON vllm_cache_observations(session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS vllm_cache_observations_repo_idx
    ON vllm_cache_observations(repo, created_at DESC);

CREATE INDEX IF NOT EXISTS vllm_cache_observations_namespace_idx
    ON vllm_cache_observations(namespace, created_at DESC);

CREATE INDEX IF NOT EXISTS vllm_cache_observations_attempt_idx
    ON vllm_cache_observations(attempt_id);
