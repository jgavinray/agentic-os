-- Move failure signature lookup to failed outcome-event payload metadata.
--
-- Standalone failure_signature events remain historical artifacts, but retrieval
-- no longer reads them and the old partial index is intentionally removed.

DROP INDEX IF EXISTS idx_agent_events_failure_signature;

CREATE INDEX IF NOT EXISTS idx_agent_events_inline_failure_signature
    ON agent_events(repo, ((metadata->'payload'->>'signature')))
    WHERE event_type IN (
        'tool_result',
        'compile_result',
        'test_result',
        'lint_result',
        'validation_result'
    )
      AND metadata->>'success' = 'false'
      AND metadata->'payload'->>'signature' IS NOT NULL;
