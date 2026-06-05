-- Add first-class implementation intent to derived request classifications.
--
-- V9 is already applied in deployed databases, so extend the intent check
-- constraint forward-only instead of editing the original migration.

ALTER TABLE agent_request_classifications
    DROP CONSTRAINT IF EXISTS agent_request_classifications_intent_check,
    ADD CONSTRAINT agent_request_classifications_intent_check
        CHECK (intent IN (
            'explain', 'debug', 'implement', 'generate_config', 'modify_config',
            'summarize', 'classify', 'search', 'plan',
            'operate_tool', 'unknown'
        ));
