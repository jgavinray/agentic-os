# ADR 0004: Deterministic Trajectory Lineage

## Status

Accepted

## Context

agentic-os already records execution outcomes, validation results, retry telemetry, failure signatures, layered memory, retrieval metadata, and sampling parameters. These events explain important pieces of engineering work, but they did not deterministically identify which full user-intent lifecycle each event belonged to.

Without lifecycle lineage, future eval replay, trajectory scoring, routing policy, and useful-deliverables-per-token analysis would have to infer chains after the fact.

## Decision

Add trajectory lineage to the existing event log. Trajectory-participating events may carry `trajectory_id`, `attempt_index`, and `event_role` as nullable queryable columns on `agent_events`.

Use bounded enums for `event_role`, `validator_type`, and final trajectory status. Reject invalid values at the application layer and with database constraints where applicable.

Write `context_pack` events before the `model_response` they inform. Store a `context_pack_id` back-reference on the model response.

Emit one `trajectory_result` event per `trajectory_id`, enforced with a partial unique index. Aggregate totals from event metadata deterministically. Token totals sum only non-null upstream counts; missing counts set `token_totals_complete=false`.

Run an idle sweep only when `TRAJECTORY_CAPTURE_ENABLED=true`.

## Consequences

Trajectory reconstruction remains a normal event-log query. No new storage backend, graph database, trajectory table, semantic trajectory search, scoring model, XGBoost integration, reinforcement learning, or autonomous retry policy is introduced.

The event log now has enough deterministic lineage for future eval and policy systems to consume raw trajectories without reclassifying boundaries through an LLM.

When `TRAJECTORY_CAPTURE_ENABLED=false`, trajectory metadata and results are not written and existing behavior is preserved.
