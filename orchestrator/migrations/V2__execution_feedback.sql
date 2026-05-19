-- Adds nullable event-chain columns for execution feedback.
--
-- This migration is forward-only. Existing rows remain valid with NULL
-- correlation_id and parent_event_id values; rolling back would only remove
-- optional metadata used by newer orchestrator versions.

ALTER TABLE agent_events
    ADD COLUMN IF NOT EXISTS correlation_id UUID,
    ADD COLUMN IF NOT EXISTS parent_event_id UUID;

CREATE INDEX IF NOT EXISTS idx_agent_events_correlation_id
    ON agent_events(correlation_id)
    WHERE correlation_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_agent_events_parent_event_id
    ON agent_events(parent_event_id)
    WHERE parent_event_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_agent_events_failure_signature
    ON agent_events(repo, ((metadata->'payload'->>'signature')))
    WHERE event_type = 'failure_signature';
