-- Adversarial harness MVP measurement substrate.
--
-- Raw model calls, prompts, and responses remain outside the portable memory
-- log. This migration adds append-only outcome labels, raw LiteLLM callback
-- payload capture, and baseline-arm attribution for comparable eval runs.

CREATE TABLE IF NOT EXISTS harness_outcome_events (
    outcome_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trajectory_id UUID NOT NULL,
    source_event_ids UUID[] NOT NULL DEFAULT '{}',
    baseline_arm TEXT NOT NULL CHECK (baseline_arm IN (
        'raw_vllm_via_litellm',
        'orchestrator_policy_disabled',
        'orchestrator_policy_enabled',
        'orchestrator_policy_plus_observational_review',
        'frontier_escalation',
        'unknown'
    )),
    accepted BOOLEAN NOT NULL,
    invalid_output BOOLEAN NOT NULL DEFAULT false,
    manual_prompt_edits_required INTEGER NOT NULL DEFAULT 0
        CHECK (manual_prompt_edits_required >= 0),
    human_attention_bucket TEXT NOT NULL CHECK (
        human_attention_bucket IN ('none', 'low', 'medium', 'high', 'unknown')
    ),
    labeler_type TEXT NOT NULL CHECK (
        labeler_type IN ('eval_operator', 'client_hook', 'human', 'posthoc')
    ),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_harness_outcome_events_trajectory
    ON harness_outcome_events(trajectory_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_harness_outcome_events_baseline_created
    ON harness_outcome_events(baseline_arm, created_at DESC);

CREATE TABLE IF NOT EXISTS litellm_callback_payloads (
    callback_payload_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    attempt_id UUID,
    trajectory_id UUID,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    callback_type TEXT NOT NULL,
    payload JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_litellm_callback_payloads_attempt
    ON litellm_callback_payloads(attempt_id, received_at DESC)
    WHERE attempt_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_litellm_callback_payloads_trajectory
    ON litellm_callback_payloads(trajectory_id, received_at DESC)
    WHERE trajectory_id IS NOT NULL;

ALTER TABLE litellm_call_ledger
    ADD COLUMN IF NOT EXISTS baseline_arm TEXT,
    ADD COLUMN IF NOT EXISTS litellm_call_id TEXT,
    ADD COLUMN IF NOT EXISTS model_group TEXT,
    ADD COLUMN IF NOT EXISTS deployment_id TEXT,
    ADD COLUMN IF NOT EXISTS api_key_alias TEXT,
    ADD COLUMN IF NOT EXISTS response_cost DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS key_spend DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS fallback_depth INTEGER,
    ADD COLUMN IF NOT EXISTS raw_callback_payload_id UUID;

CREATE INDEX IF NOT EXISTS litellm_call_ledger_baseline_arm_idx
    ON litellm_call_ledger(baseline_arm);

CREATE INDEX IF NOT EXISTS litellm_call_ledger_litellm_call_id_idx
    ON litellm_call_ledger(litellm_call_id)
    WHERE litellm_call_id IS NOT NULL;
