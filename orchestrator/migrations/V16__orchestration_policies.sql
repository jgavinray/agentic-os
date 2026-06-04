-- Append-only orchestration policy ledger.
--
-- This table records the deterministic policy envelope derived for request
-- events and tool-authorization events. It is intentionally append-only:
-- policies are historical facts about what the orchestrator decided at a
-- specific point in a trajectory, not mutable current state.
--
-- Every derived policy is appended; no ON CONFLICT / UPDATE.
--
-- JSONB columns store arrays of snake_case string labels from Rust as_str()
-- methods. The database intentionally does not enforce JSONB enum labels with
-- CHECK constraints because policy labels are validated by closed Rust enums.
-- That keeps migrations stable when arrays grow while preserving type safety at
-- the source-code boundary.

CREATE TABLE IF NOT EXISTS agent_orchestration_policies (
    -- Unique append-only row id for this derived policy.
    policy_id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Durable event id when this policy was attached to a request/tool event.
    event_id                TEXT,
    -- Session/repo are duplicated here so analytics can query policy decisions
    -- without joining through agent_events first.
    session_id              TEXT NOT NULL,
    repo                    TEXT NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Classification and route versions that produced the input labels.
    classification_schema_version INTEGER NOT NULL,
    routing_policy_version      TEXT NOT NULL,
    -- Version of the policy schema in this table.
    policy_schema_version       INTEGER NOT NULL DEFAULT 1,

    -- Primary request labels copied from RequestClassification.
    intent              TEXT NOT NULL,
    recommended_route   TEXT NOT NULL,

    -- Vector fields are arrays of snake_case labels.
    context_sources     JSONB NOT NULL DEFAULT '[]'::jsonb,
    allowed_tools       JSONB NOT NULL DEFAULT '[]'::jsonb,
    required_tools      JSONB NOT NULL DEFAULT '[]'::jsonb,
    blocked_tools       JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Scalar policy dimensions.
    edit_policy         TEXT NOT NULL,
    validation_policy   TEXT NOT NULL,
    git_policy          TEXT NOT NULL,
    runtime_policy      TEXT NOT NULL,

    -- Additional vector/scalar policy dimensions.
    scope_policy        JSONB NOT NULL DEFAULT '[]'::jsonb,
    prompt_refinement_policy TEXT NOT NULL DEFAULT 'none',
    risk_policy         JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Source of the policy derivation. Currently deterministic rules only.
    source              TEXT NOT NULL DEFAULT 'deterministic_rules'
);

-- Common access patterns:
-- - per-session trajectory inspection,
-- - per-repo policy analysis,
-- - direct event lookup,
-- - intent/prompt-refinement analytics.
CREATE INDEX IF NOT EXISTS idx_orchestration_policies_session_created
    ON agent_orchestration_policies (session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_orchestration_policies_repo_created
    ON agent_orchestration_policies (repo, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_orchestration_policies_event_id
    ON agent_orchestration_policies (event_id) WHERE event_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_orchestration_policies_intent_created
    ON agent_orchestration_policies (intent, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_orchestration_policies_prompt_refinement_created
    ON agent_orchestration_policies (prompt_refinement_policy, created_at DESC);
