CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE agent_sessions (
    id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    task TEXT NOT NULL,
    actor TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE agent_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    repo TEXT NOT NULL,
    actor TEXT NOT NULL,
    event_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    evidence TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    summarized BOOLEAN NOT NULL DEFAULT false,
    summary_level INT NOT NULL DEFAULT 0,
    CONSTRAINT fk_session
        FOREIGN KEY (session_id)
        REFERENCES agent_sessions(id)
        ON DELETE CASCADE
);

CREATE TABLE error_index (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    repo TEXT NOT NULL,
    task TEXT NOT NULL,
    error_type TEXT NOT NULL,
    description TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'medium',
    frequency BIGINT NOT NULL DEFAULT 1,
    last_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_error_index UNIQUE (repo, task, error_type, description)
);

CREATE TABLE token_usage (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    requested_model TEXT NOT NULL,
    actual_model TEXT NOT NULL,
    namespace TEXT NOT NULL,
    repo TEXT NOT NULL,
    processed_tokens BIGINT NOT NULL DEFAULT 0,
    cached_tokens BIGINT NOT NULL DEFAULT 0,
    generated_tokens BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_agent_events_repo_created
    ON agent_events(repo, created_at DESC);

CREATE INDEX idx_agent_events_session
    ON agent_events(session_id);

CREATE INDEX idx_agent_events_fts
    ON agent_events USING gin(
        to_tsvector('english',
            coalesce(summary, '') || ' ' || coalesce(evidence, ''))
    );

CREATE INDEX idx_agent_events_type
    ON agent_events(event_type, repo);

CREATE INDEX idx_agent_events_summary_level
    ON agent_events(repo, summary_level, summarized, created_at DESC);

CREATE INDEX idx_error_index_repo
    ON error_index(repo);

CREATE INDEX idx_error_index_type
    ON error_index(error_type);

CREATE INDEX idx_error_index_freq
    ON error_index(frequency DESC);

CREATE INDEX idx_token_usage_model
    ON token_usage(requested_model, created_at DESC);

CREATE INDEX idx_token_usage_namespace
    ON token_usage(namespace, created_at DESC);
