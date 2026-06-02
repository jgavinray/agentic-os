CREATE TABLE IF NOT EXISTS context_artifacts (
    id UUID PRIMARY KEY,
    repo TEXT NOT NULL,
    scope TEXT NOT NULL
        CHECK (scope IN ('global', 'repo', 'session', 'trajectory', 'request')),
    artifact_type TEXT NOT NULL
        CHECK (artifact_type IN (
            'service_topology',
            'repo_map',
            'durable_project_memory',
            'repo_decisions',
            'failure_history',
            'active_instruction',
            'session_state',
            'tool_trace',
            'unknown'
        )),
    status TEXT NOT NULL
        CHECK (status IN (
            'active',
            'superseded',
            'stale',
            'resolved',
            'disputed',
            'quarantined'
        )),
    content_raw TEXT,
    content_compact TEXT NOT NULL,
    content_rendered TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    invalidation_key TEXT NOT NULL,
    source_event_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_file_paths JSONB NOT NULL DEFAULT '[]'::jsonb,
    token_estimate INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (repo, scope, artifact_type, invalidation_key)
);

CREATE INDEX IF NOT EXISTS idx_context_artifacts_active
    ON context_artifacts(repo, scope, artifact_type, updated_at DESC)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_context_artifacts_hash
    ON context_artifacts(content_hash);
