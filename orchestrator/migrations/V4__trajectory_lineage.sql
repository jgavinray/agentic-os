-- Add deterministic trajectory reconstruction primitives to the existing event log.

ALTER TABLE agent_events
    ADD COLUMN IF NOT EXISTS trajectory_id UUID,
    ADD COLUMN IF NOT EXISTS attempt_index INT,
    ADD COLUMN IF NOT EXISTS event_role TEXT;

ALTER TABLE agent_events
    DROP CONSTRAINT IF EXISTS ck_agent_events_attempt_index_positive;

ALTER TABLE agent_events
    ADD CONSTRAINT ck_agent_events_attempt_index_positive
    CHECK (attempt_index IS NULL OR attempt_index >= 1);

ALTER TABLE agent_events
    DROP CONSTRAINT IF EXISTS ck_agent_events_event_role;

ALTER TABLE agent_events
    ADD CONSTRAINT ck_agent_events_event_role
    CHECK (
        event_role IS NULL OR event_role IN (
            'request',
            'context_pack',
            'model_response',
            'tool_call',
            'tool_result',
            'validation',
            'patch',
            'failure',
            'remediation',
            'trajectory_result'
        )
    );

CREATE INDEX IF NOT EXISTS idx_agent_events_trajectory
    ON agent_events(trajectory_id, attempt_index, created_at, id)
    WHERE trajectory_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_agent_events_trajectory_result
    ON agent_events(trajectory_id)
    WHERE event_role = 'trajectory_result';

CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_events_trajectory_result_once
    ON agent_events(trajectory_id)
    WHERE event_role = 'trajectory_result';

CREATE INDEX IF NOT EXISTS idx_agent_events_session_trajectory_latest
    ON agent_events(session_id, created_at DESC)
    WHERE trajectory_id IS NOT NULL;
