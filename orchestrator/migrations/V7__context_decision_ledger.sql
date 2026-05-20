-- Shadow Context Decision Ledger: envelope table
--
-- One row per context-pack assembly attempt, including cache hits and
-- assembly failures. Append-only.

CREATE TABLE IF NOT EXISTS context_decision_envelopes (
    decision_id                  TEXT NOT NULL,
    source_decision_id           TEXT,
    repo                         TEXT NOT NULL,
    session_id                   TEXT,
    trajectory_id                TEXT,
    request_route                TEXT NOT NULL,
    task_type                    TEXT NOT NULL,
    created_at_micros            BIGINT NOT NULL,
    context_policy_version       TEXT NOT NULL,
    feature_schema_version       INTEGER NOT NULL,
    total_token_budget           INTEGER NOT NULL,
    candidate_tokens_total       INTEGER NOT NULL,
    fixed_overhead_tokens        INTEGER NOT NULL,
    final_estimated_context_tokens INTEGER NOT NULL,
    cache_status                 TEXT NOT NULL,
    candidate_count              INTEGER NOT NULL,
    injected_candidate_count     INTEGER NOT NULL,
    assembly_duration_micros     BIGINT NOT NULL,
    assembly_status              TEXT NOT NULL,
    PRIMARY KEY (decision_id)
);

-- One row per candidate that reached deterministic ranker input.
-- Append-only.

CREATE TABLE IF NOT EXISTS context_decision_candidates (
    candidate_id                 TEXT NOT NULL,
    decision_id                  TEXT NOT NULL
        REFERENCES context_decision_envelopes(decision_id),
    source_type                  TEXT NOT NULL,
    source_record_id             TEXT NOT NULL,
    event_type                   TEXT,
    summary_level                TEXT NOT NULL,
    age_seconds                  INTEGER NOT NULL,
    same_repo                    BOOLEAN NOT NULL,
    same_session                 BOOLEAN NOT NULL,
    same_trajectory              BOOLEAN NOT NULL,
    estimated_token_cost         INTEGER NOT NULL,
    token_cost_bucket            TEXT NOT NULL,
    retrieval_score_bucket       TEXT NOT NULL,
    failure_class                TEXT,
    operational_constraint_type  TEXT,
    context_section              TEXT NOT NULL,
    duplicate_coverage           TEXT NOT NULL,
    deterministic_score_bucket   TEXT NOT NULL,
    deterministic_rank           INTEGER NOT NULL,
    budget_before_candidate      INTEGER NOT NULL,
    budget_after_candidate       INTEGER NOT NULL,
    injected                     BOOLEAN NOT NULL,
    drop_reason                  TEXT,
    drop_reason_other_detail     TEXT,
    representation_selected      TEXT NOT NULL,
    PRIMARY KEY (candidate_id)
);

-- One row per decision, upserted asynchronously, keyed by decision_id.

CREATE TABLE IF NOT EXISTS context_decision_outcomes (
    decision_id                          TEXT PRIMARY KEY,
    trajectory_outcome                   TEXT NOT NULL,
    user_correction_after_decision       BOOLEAN,
    retry_after_decision                 BOOLEAN,
    validation_result                    TEXT NOT NULL,
    tool_loop_detected_after_decision    BOOLEAN,
    request_latency_bucket               TEXT NOT NULL,
    input_token_bucket                   TEXT NOT NULL,
    output_token_bucket                  TEXT NOT NULL,
    outcome_joined_at_micros             BIGINT,
    outcome_window_closed                BOOLEAN NOT NULL,
    feature_schema_version               INTEGER NOT NULL
);
