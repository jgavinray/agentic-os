-- Derived request classification rows built from agent_events.
--
-- agent_events remains the audit log and source of truth. This table is
-- rebuildable derived state for deterministic pre-LLM routing analysis.

CREATE TABLE IF NOT EXISTS agent_request_classifications (
    event_id TEXT NOT NULL
        REFERENCES agent_events(id)
        ON DELETE CASCADE,
    repo TEXT NOT NULL,
    session_id TEXT NOT NULL,
    trajectory_id UUID,
    event_created_at TIMESTAMPTZ NOT NULL,
    classified_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    classification_schema_version INTEGER NOT NULL,
    routing_policy_version TEXT NOT NULL,
    classifier_source TEXT NOT NULL,

    intent TEXT NOT NULL
        CHECK (intent IN (
            'explain', 'debug', 'generate_config', 'modify_config',
            'summarize', 'classify', 'search', 'plan',
            'operate_tool', 'unknown'
        )),
    domain TEXT NOT NULL
        CHECK (domain IN (
            'shell', 'kubernetes', 'llm_inference', 'docker',
            'networking', 'security', 'medical', 'legal',
            'finance', 'generic', 'unknown'
        )),
    secondary_domains TEXT[] NOT NULL DEFAULT '{}'::TEXT[]
        CHECK (secondary_domains <@ ARRAY[
            'shell', 'kubernetes', 'llm_inference', 'docker',
            'networking', 'security', 'medical', 'legal',
            'finance', 'generic', 'unknown'
        ]::TEXT[]),
    artifact_type TEXT NOT NULL
        CHECK (artifact_type IN (
            'plain_text', 'code', 'logs', 'yaml', 'json',
            'sql', 'markdown', 'image', 'file', 'unknown'
        )),
    risk TEXT[] NOT NULL DEFAULT '{}'::TEXT[]
        CHECK (risk <@ ARRAY[
            'none', 'secret_present', 'destructive_command',
            'external_current_info_required', 'high_stakes',
            'prompt_injection', 'unsafe_security', 'unknown'
        ]::TEXT[]),
    complexity TEXT NOT NULL
        CHECK (complexity IN (
            'l0_trivial', 'l1_simple', 'l2_moderate',
            'l3_complex', 'l4_tool_required', 'l5_high_risk',
            'unknown'
        )),
    recommended_route TEXT NOT NULL
        CHECK (recommended_route IN (
            'deterministic_template', 'small_local_model',
            'strong_local_model', 'web_required', 'tool_required',
            'ask_clarification', 'refuse_or_guardrail', 'unknown'
        )),
    response_contract TEXT NOT NULL
        CHECK (response_contract IN (
            'direct_answer', 'structured_json', 'markdown_summary',
            'patch_required', 'validation_required',
            'clarification_question', 'refusal', 'unknown'
        )),

    features JSONB NOT NULL DEFAULT '{}'::JSONB,

    PRIMARY KEY (event_id, classification_schema_version, routing_policy_version)
);

CREATE INDEX IF NOT EXISTS idx_agent_request_classifications_repo_created
    ON agent_request_classifications(repo, event_created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_request_classifications_repo_route_created
    ON agent_request_classifications(repo, recommended_route, event_created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_request_classifications_repo_intent_domain
    ON agent_request_classifications(repo, intent, domain);

CREATE INDEX IF NOT EXISTS idx_agent_request_classifications_repo_complexity
    ON agent_request_classifications(repo, complexity);

CREATE INDEX IF NOT EXISTS idx_agent_request_classifications_risk
    ON agent_request_classifications USING GIN(risk);
