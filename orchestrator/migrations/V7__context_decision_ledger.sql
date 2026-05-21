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
    request_route                TEXT NOT NULL
        CHECK (request_route IN (
            'chat_completions', 'messages', 'unknown'
        )),
    task_type                    TEXT NOT NULL
        CHECK (task_type IN (
            'debug', 'feature', 'refactor', 'architecture', 'deploy', 'unknown'
        )),
    created_at_micros            BIGINT NOT NULL,
    context_policy_version       TEXT NOT NULL,
    feature_schema_version       INTEGER NOT NULL,
    total_token_budget           INTEGER NOT NULL,
    candidate_tokens_total       INTEGER NOT NULL,
    fixed_overhead_tokens        INTEGER NOT NULL,
    final_estimated_context_tokens INTEGER NOT NULL,
    cache_status                 TEXT NOT NULL
        CHECK (cache_status IN (
            'miss', 'hit_fresh', 'hit_stale', 'bypass', 'unknown'
        )),
    candidate_count              INTEGER NOT NULL,
    injected_candidate_count     INTEGER NOT NULL,
    assembly_duration_micros     BIGINT NOT NULL,
    assembly_status              TEXT NOT NULL
        CHECK (assembly_status IN (
            'success', 'partial', 'failed', 'unknown'
        )),
    PRIMARY KEY (decision_id)
);

-- One row per candidate that reached deterministic ranker input.
-- Append-only.

CREATE TABLE IF NOT EXISTS context_decision_candidates (
    candidate_id                 TEXT NOT NULL,
    decision_id                  TEXT NOT NULL
        REFERENCES context_decision_envelopes(decision_id),
    source_type                  TEXT NOT NULL
        CHECK (source_type IN (
            'raw_event', 'summary', 'failure_history',
            'operational_constraint', 'retrieval_hit', 'feature_record', 'unknown'
        )),
    source_record_id             TEXT NOT NULL,
    event_type                   TEXT
        CHECK (event_type IN (
            'user_message', 'assistant_message', 'tool_call',
            'tool_result', 'error', 'summary', 'checkpoint', 'unknown'
        )),
    summary_level                TEXT NOT NULL
        CHECK (summary_level IN (
            'none', 'compact', 'full', 'rolled_up', 'unknown'
        )),
    age_seconds                  INTEGER NOT NULL,
    same_repo                    BOOLEAN NOT NULL,
    same_session                 BOOLEAN NOT NULL,
    same_trajectory              BOOLEAN NOT NULL,
    estimated_token_cost         INTEGER NOT NULL,
    token_cost_bucket            TEXT NOT NULL
        CHECK (token_cost_bucket IN (
            'bucket_0_50', 'bucket_51_200', 'bucket_201_500',
            'bucket_501_1000', 'bucket_1001_2000', 'bucket_2001_plus', 'unknown'
        )),
    retrieval_score_bucket       TEXT NOT NULL
        CHECK (retrieval_score_bucket IN (
            'none', 'q1', 'q2', 'q3', 'q4', 'unknown'
        )),
    failure_class                TEXT
        CHECK (failure_class IN (
            'tool_loop', 'user_interruption', 'missing_auth',
            'wrong_endpoint', 'summarization_failure', 'migration_failure',
            'context_pack_empty', 'context_pack_truncated',
            'high_input_tokens', 'slow_upstream_model',
            'empty_tool_use_message', 'abandoned_before_model',
            'single_model_abandoned_no_tools', 'summarizer_shared_upstream', 'other'
        )),
    operational_constraint_type  TEXT
        CHECK (operational_constraint_type IN (
            'use_known_auth', 'use_known_endpoint', 'use_known_migration_fix',
            'avoid_tool_loop', 'fix_context_retrieval', 'reduce_context_bloat',
            'separate_summarizer_upstream', 'handle_user_interruption',
            'handle_summarization_failure', 'unknown'
        )),
    context_section              TEXT NOT NULL
        CHECK (context_section IN (
            'recent_events', 'session_summary', 'repo_summary',
            'project_summary', 'failure_history', 'operational_constraints', 'unknown'
        )),
    duplicate_coverage           TEXT NOT NULL
        CHECK (duplicate_coverage IN (
            'none', 'partial', 'full', 'unknown'
        )),
    deterministic_score_bucket   TEXT NOT NULL
        CHECK (deterministic_score_bucket IN (
            'very_low', 'low', 'medium', 'high', 'very_high', 'unknown'
        )),
    deterministic_rank           INTEGER NOT NULL,
    budget_before_candidate      INTEGER NOT NULL,
    budget_after_candidate       INTEGER NOT NULL,
    injected                     BOOLEAN NOT NULL,
    drop_reason                  TEXT
        CHECK (drop_reason IN (
            'priority_cap', 'token_budget', 'duplicate', 'stale',
            'low_relevance', 'section_disabled', 'recovered',
            'superseded', 'other', 'unknown'
        )),
    drop_reason_other_detail     TEXT CHECK (char_length(drop_reason_other_detail) <= 256),
    representation_selected      TEXT NOT NULL
        CHECK (representation_selected IN (
            'none', 'compact', 'full', 'unknown'
        )),
    PRIMARY KEY (candidate_id)
);

-- One row per decision, upserted asynchronously, keyed by decision_id.

CREATE TABLE IF NOT EXISTS context_decision_outcomes (
    decision_id                          TEXT PRIMARY KEY,
    trajectory_outcome                   TEXT NOT NULL
        CHECK (trajectory_outcome IN (
            'succeeded', 'unresolved', 'abandoned', 'reverted', 'unknown'
        )),
    user_correction_after_decision       BOOLEAN,
    retry_after_decision                 BOOLEAN,
    validation_result                    TEXT NOT NULL
        CHECK (validation_result IN (
            'passed', 'failed', 'not_run', 'unknown'
        )),
    tool_loop_detected_after_decision    BOOLEAN,
    request_latency_bucket               TEXT NOT NULL
        CHECK (request_latency_bucket IN (
            'bucket__500ms', 'bucket_500ms_1s', 'bucket_1s_3s',
            'bucket_3s_10s', 'bucket_10s_plus', 'unknown'
        )),
    input_token_bucket                   TEXT NOT NULL
        CHECK (input_token_bucket IN (
            'bucket_0_1000', 'bucket_1000_4000', 'bucket_4000_8000',
            'bucket_8000_32000', 'bucket_32000_plus', 'unknown'
        )),
    output_token_bucket                  TEXT NOT NULL
        CHECK (output_token_bucket IN (
            'bucket_0_128', 'bucket_128_512', 'bucket_512_2048',
            'bucket_2048_8192', 'bucket_8192_plus', 'unknown'
        )),
    outcome_joined_at_micros             BIGINT,
    outcome_window_closed                BOOLEAN NOT NULL,
    feature_schema_version               INTEGER NOT NULL
);
