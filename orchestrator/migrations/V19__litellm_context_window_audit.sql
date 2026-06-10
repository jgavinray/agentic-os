ALTER TABLE litellm_call_ledger
    ADD COLUMN IF NOT EXISTS upstream_error_body_sha256 TEXT,
    ADD COLUMN IF NOT EXISTS upstream_error_excerpt TEXT,
    ADD COLUMN IF NOT EXISTS context_limit_tokens BIGINT,
    ADD COLUMN IF NOT EXISTS context_input_tokens BIGINT,
    ADD COLUMN IF NOT EXISTS context_requested_output_tokens BIGINT,
    ADD COLUMN IF NOT EXISTS context_retry_output_tokens BIGINT,
    ADD COLUMN IF NOT EXISTS retry_reason TEXT;

CREATE INDEX IF NOT EXISTS litellm_call_ledger_retry_reason_idx
    ON litellm_call_ledger(retry_reason);
