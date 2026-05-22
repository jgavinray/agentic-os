-- Correct Phase 1+2 context decision ledger enum constraints without
-- rewriting the already committed V7 migration.

ALTER TABLE context_decision_candidates
    DROP CONSTRAINT IF EXISTS context_decision_candidates_deterministic_score_bucket_check,
    ADD CONSTRAINT context_decision_candidates_deterministic_score_bucket_check
        CHECK (deterministic_score_bucket IN (
            'none', 'very_low', 'low', 'medium', 'high', 'very_high', 'unknown'
        ));

ALTER TABLE context_decision_outcomes
    DROP CONSTRAINT IF EXISTS context_decision_outcomes_request_latency_bucket_check,
    ADD CONSTRAINT context_decision_outcomes_request_latency_bucket_check
        CHECK (request_latency_bucket IN (
            'bucket_0_500ms', 'bucket_500ms_1s', 'bucket_1s_3s',
            'bucket_3s_10s', 'bucket_10s_plus', 'unknown'
        ));
