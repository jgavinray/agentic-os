-- Additional deterministic operational health counters derived from agent_events.
--
-- These columns keep new signal classes queryable without changing the audit
-- log contract. They are derived state and remain rebuildable from agent_events.

ALTER TABLE agent_feature_records
    ADD COLUMN IF NOT EXISTS context_pack_empty_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS context_pack_truncated_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS high_input_token_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS slow_upstream_model_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS empty_tool_use_message_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS abandoned_before_model_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS single_model_abandoned_no_tools_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS summarizer_shared_upstream_count BIGINT NOT NULL DEFAULT 0;
