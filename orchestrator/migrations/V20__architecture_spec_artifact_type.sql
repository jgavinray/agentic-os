-- Allow operator-registered specification artifacts. Spec grounding registers
-- architecture/design documents through POST /context/artifacts so they ride
-- the stable cacheable system prefix on every request for the repo.
ALTER TABLE context_artifacts
    DROP CONSTRAINT context_artifacts_artifact_type_check;

ALTER TABLE context_artifacts
    ADD CONSTRAINT context_artifacts_artifact_type_check CHECK (
        artifact_type = ANY (
            ARRAY[
                'service_topology'::text,
                'repo_map'::text,
                'durable_project_memory'::text,
                'repo_decisions'::text,
                'failure_history'::text,
                'active_instruction'::text,
                'session_state'::text,
                'tool_trace'::text,
                'architecture_spec'::text,
                'unknown'::text
            ]
        )
    );
