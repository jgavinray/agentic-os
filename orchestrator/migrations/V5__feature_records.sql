-- Derived operational feature rows built from agent_events.
--
-- agent_events remains the audit log and source of truth. This table is
-- intentionally rebuildable and idempotently upserted by the deterministic
-- feature extractor.

CREATE TABLE IF NOT EXISTS agent_feature_records (
    feature_id TEXT PRIMARY KEY,
    trajectory_id UUID,
    session_id TEXT NOT NULL,
    repo TEXT NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,

    event_count BIGINT NOT NULL DEFAULT 0,
    user_message_count BIGINT NOT NULL DEFAULT 0,
    assistant_message_count BIGINT NOT NULL DEFAULT 0,
    failed_attempt_count BIGINT NOT NULL DEFAULT 0,
    summary_count BIGINT NOT NULL DEFAULT 0,

    tool_loop_count BIGINT NOT NULL DEFAULT 0,
    repeated_read_loop_count BIGINT NOT NULL DEFAULT 0,
    repeated_bash_loop_count BIGINT NOT NULL DEFAULT 0,
    user_interruption_count BIGINT NOT NULL DEFAULT 0,
    missing_auth_count BIGINT NOT NULL DEFAULT 0,
    wrong_endpoint_count BIGINT NOT NULL DEFAULT 0,
    summarization_failure_count BIGINT NOT NULL DEFAULT 0,
    migration_failure_count BIGINT NOT NULL DEFAULT 0,
    other_failure_count BIGINT NOT NULL DEFAULT 0,

    known_endpoint TEXT,
    known_auth_header TEXT,
    known_migration_fix TEXT,

    loop_detected BOOLEAN NOT NULL DEFAULT false,
    user_interrupted BOOLEAN NOT NULL DEFAULT false,

    failure_classes JSONB NOT NULL DEFAULT '[]'::jsonb,
    recommended_constraints JSONB NOT NULL DEFAULT '[]'::jsonb,
    suppressed_constraints JSONB NOT NULL DEFAULT '[]'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_feature_records_trajectory
    ON agent_feature_records(trajectory_id)
    WHERE trajectory_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_feature_records_session_window
    ON agent_feature_records(repo, session_id, window_start, window_end)
    WHERE trajectory_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_agent_feature_records_repo_updated
    ON agent_feature_records(repo, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_feature_records_session_window_latest
    ON agent_feature_records(repo, session_id, window_end DESC);
