# Trajectory Lineage

agentic-os captures trajectory lineage as an event chain in the existing `agent_events` log. A trajectory is one user intent plus the downstream context pack, model response, tool results, validations, patches, remediations, failures, and final operational result produced while trying to satisfy that intent.

No graph database, scoring table, learned router, retry policy, or prompt-body archive is involved.

## Event Model

Trajectory-participating events use three queryable columns:

- `trajectory_id`: UUID for the full user-intent lifecycle.
- `attempt_index`: positive integer for the remediation phase. The first attempt is `1`.
- `event_role`: bounded operational role, separate from `event_type`.

Allowed `event_role` values:

- `request`
- `context_pack`
- `model_response`
- `tool_call`
- `tool_result`
- `validation`
- `patch`
- `failure`
- `remediation`
- `trajectory_result`

The same values are duplicated into event metadata when capture is enabled so historical JSON payload consumers can inspect lineage without depending on column projection.

## Boundary Rules

A new trajectory starts when a user message is ingested and no active trajectory exists for the session, the previous trajectory has already ended, or the previous trajectory has been idle longer than `TRAJECTORY_IDLE_TIMEOUT_SEC`.

A user message on the same session finalizes any open previous trajectory and starts a new `trajectory_id`. Follow-on context packs, model responses, tool results, validations, patches, failures, and remediations inherit the active `trajectory_id`.

The default idle timeout is 600 seconds.

## Attempt Index

`attempt_index=1` is used for the initial attempt. A retry increments the attempt only when a model response follows a failed terminal event in the same trajectory. Failed terminal events are failed validations, failed or reverted patches, failure events, and failed remediations.

Multiple model responses can share the same `attempt_index` when they happen before another failed terminal event.

## Context Pack Lineage

When a context pack is assembled for a captured model request, a `context_pack` event is written before the `model_response` it informs.

The context pack payload includes:

- `context_pack_id`
- `trajectory_id`
- `attempt_index`
- `total_context_tokens`
- `retrieved_event_ids`
- `retrieval_strategy`
- `semantic_hits`
- `keyword_hits`
- `reranked_hits`
- `memory_levels_used`
- `injected_failure_signatures`
- `token_budget`
- `truncated`

The downstream `model_response` stores `context_pack_id` as a back-reference. Full prompt bodies are not archived.

## Model Metadata

`model_response` metadata records:

- `model_name`
- `provider`
- `input_tokens`
- `output_tokens`
- `latency_ms`
- `finish_reason`
- `sampling_params`
- `forwarded_sampling_params`
- `context_pack_id`

Token counts come from upstream response usage fields when present. Missing or unparsable token counts are stored as `null`; they are not estimated.

## Validation And Patch Payloads

Validation-style events use bounded `validator_type` values:

- `compile`
- `test`
- `lint`
- `type_check`
- `schema`
- `static_analysis`
- `other`

Invalid validator types are rejected. Validation payloads include `validator_name`, `validator_type`, `command`, `success`, `exit_code`, `duration_ms`, `failure_signature`, and `failure_excerpt`.

Patch payloads include `files_touched`, `lines_added`, `lines_removed`, `patch_applied`, and `patch_reverted`.

## Tool Mediation Lineage

Tool menu shaping metadata is attached to request and model-response metadata when `TOOL_MEDIATION_ENABLED=true`.

The `/tools/authorize` endpoint can also write `tool_authorization_decision` events when the client supplies `session_id`. If the client supplies `trajectory_id` and `attempt_index`, the decision event participates in the trajectory using `event_role=tool_call`.

This keeps canonical-tool preference decisions, such as denying `Bash("cat README.md")` when `Read` is available, reconstructable from the same event chain without adding a new storage backend or event-role enum.

## Trajectory Result

At completion, agentic-os writes exactly one `trajectory_result` event per `trajectory_id`. Idempotency is enforced by a partial unique index on `agent_events(trajectory_id)` where `event_role='trajectory_result'`.

Result payload fields:

- `trajectory_id`
- `final_status`
- `success`
- `retry_count`
- `total_model_calls`
- `total_tool_calls`
- `total_validations`
- `total_input_tokens`
- `total_output_tokens`
- `token_totals_complete`
- `total_latency_ms`
- `models_used`
- `validators_used`
- `files_touched`
- `failure_signatures`
- `remediation_count`
- `final_attempt_index`

Allowed `final_status` values are `succeeded`, `abandoned`, `unresolved`, and `reverted`.

`success=true` only when `final_status=succeeded`.

## Status Rules

`succeeded`: the trajectory contains an applied patch, no later reverted patch, and the most recent relevant validation succeeded.

`reverted`: the trajectory contains an applied patch followed by a reverted patch.

`abandoned`: a new user message starts a new trajectory before the previous one reaches a successful or reverted state.

`unresolved`: the trajectory reaches idle timeout before a successful or reverted state.

These are deterministic operational heuristics, not claims about the user's intent.

## Token Totals

`token_totals_complete=true` only when every `model_response` in the trajectory has non-null input and output token counts.

If any token count is missing, totals sum only non-null values and set `token_totals_complete=false`.

## Retrieval Helpers

The DB layer exposes:

- `get_trajectory(trajectory_id)`: all trajectory events ordered by `attempt_index`, `created_at`, and `id`.
- `get_trajectory_attempts(trajectory_id)`: distinct attempt indexes in order.
- `get_trajectory_result(trajectory_id)`: the single result event, or `None`.

Reconstruction is deterministic because all lineage information is stored on the event rows.

## Metrics

Trajectory metrics are bounded:

- `trajectory_results_total{status}`
- `trajectory_attempts_total`
- `trajectory_validation_failures_total{validator_type}`
- `trajectory_tokens_total{direction}`

Allowed `direction` values are `input` and `output`. Raw trajectory IDs, filenames, signatures, token counts, and user-controlled values are not metric labels.

## Feature Flag

`TRAJECTORY_CAPTURE_ENABLED=true` by default.

When set to false:

- trajectory metadata is not written,
- `trajectory_result` events are not emitted,
- the idle sweep is not started,
- trajectory metrics remain at zero,
- existing event behavior is preserved.

## Future Use

Trajectory lineage makes future scoring, eval replay, routing policy research, and training-data generation possible by making complete engineering attempts observable and reconstructable first. Those systems remain out of scope for this work.
