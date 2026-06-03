ALTER TABLE litellm_call_ledger
    ADD COLUMN IF NOT EXISTS reasoning_policy TEXT,
    ADD COLUMN IF NOT EXISTS reasoning_policy_source TEXT;

CREATE INDEX IF NOT EXISTS litellm_call_ledger_reasoning_policy_idx
    ON litellm_call_ledger(reasoning_policy);
