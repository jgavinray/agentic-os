ALTER TABLE litellm_call_ledger
    ADD COLUMN IF NOT EXISTS attempt_id UUID,
    ADD COLUMN IF NOT EXISTS terminal_status TEXT,
    ADD COLUMN IF NOT EXISTS error_kind TEXT,
    ADD COLUMN IF NOT EXISTS error_message TEXT,
    ADD COLUMN IF NOT EXISTS first_token_ms BIGINT,
    ADD COLUMN IF NOT EXISTS total_latency_ms BIGINT,
    ADD COLUMN IF NOT EXISTS context_pack_hash TEXT,
    ADD COLUMN IF NOT EXISTS cache_backend TEXT,
    ADD COLUMN IF NOT EXISTS cache_policy_enabled BOOLEAN,
    ADD COLUMN IF NOT EXISTS cache_bypass_reason TEXT,
    ADD COLUMN IF NOT EXISTS policy_version TEXT,
    ADD COLUMN IF NOT EXISTS selected_route TEXT,
    ADD COLUMN IF NOT EXISTS selection_reason TEXT,
    ADD COLUMN IF NOT EXISTS fallback_from_model TEXT,
    ADD COLUMN IF NOT EXISTS fallback_to_model TEXT,
    ADD COLUMN IF NOT EXISTS provider_cached_tokens BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS provider_cache_created_tokens BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS provider_cache_read_tokens BIGINT NOT NULL DEFAULT 0;

CREATE UNIQUE INDEX IF NOT EXISTS litellm_call_ledger_attempt_id_key
    ON litellm_call_ledger(attempt_id);

CREATE INDEX IF NOT EXISTS litellm_call_ledger_attempt_id_idx
    ON litellm_call_ledger(attempt_id);

CREATE INDEX IF NOT EXISTS litellm_call_ledger_context_pack_hash_idx
    ON litellm_call_ledger(context_pack_hash);

CREATE INDEX IF NOT EXISTS litellm_call_ledger_terminal_status_idx
    ON litellm_call_ledger(terminal_status);

CREATE INDEX IF NOT EXISTS litellm_call_ledger_policy_route_idx
    ON litellm_call_ledger(policy_version, selected_route);
