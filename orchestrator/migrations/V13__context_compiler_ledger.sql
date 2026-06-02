CREATE TABLE IF NOT EXISTS context_compiler_ledger (
    id UUID PRIMARY KEY,
    repo TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    candidate_source TEXT NOT NULL,
    candidate_id TEXT,
    decision TEXT NOT NULL
        CHECK (decision IN ('included', 'suppressed', 'superseded', 'dropped')),
    reason TEXT NOT NULL,
    artifact_id UUID,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_context_compiler_ledger_repo_created
    ON context_compiler_ledger(repo, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_context_compiler_ledger_artifact
    ON context_compiler_ledger(repo, artifact_type, decision, created_at DESC);
