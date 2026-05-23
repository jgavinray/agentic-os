# Request Classification And Routing

This track defines the pre-LLM classification layer for agentic-os.

Core thesis:

The LLM should not be the dispatcher. The orchestrator should classify,
constrain, and route work before invoking any model.

This work is separate from `docs/XGBoost/`. The XGBoost context ledger records
what context was considered and selected for future learned ranking. Request
classification records what a user/request event appears to need before routing.

## Files

Read in this order:

1. `ARCHITECTURE.md` - cross-phase invariants and taxonomy.
2. `PHASE_1_schema_taxonomy.md` - database schema, enums, Rust structs.
3. `PHASE_2_feature_extraction.md` - pure deterministic feature extraction.
4. `PHASE_3_backfill_boot_cli.md` - boot-time backfill and operator CLI.
5. `PHASE_4_shadow_routing_metrics.md` - route recommendation and metrics.
6. `PHASE_5_live_policy.md` - narrow live enforcement for safe cases.

## Phase Summary

Phase 1 creates the storage and bounded taxonomy.

Phase 2 extracts deterministic features and labels from `agent_events`.

Phase 3 makes the system self-sustaining at boot and adds a CLI escape hatch.

Phase 4 records shadow route recommendations without changing live behavior.

Phase 5 promotes only high-confidence deterministic routes into live policy.

## Runtime Behavior

Request classification is deterministic and rebuildable. `agent_events` remains
the source of truth; `agent_request_classifications` stores derived structured
rows only.

On orchestrator startup, classification backfill runs after migrations and
harness feedback repair, before feature extraction and before the HTTP listener
binds. The backfill is idempotent for each `(event_id,
classification_schema_version, routing_policy_version)` tuple.

New classifiable events are also classified inline when they are inserted.

Classifiable events are:

- `user_message`
- events with `event_role = 'request'`
- request-shaped `context_pack` events

Live policy is disabled by default. When
`REQUEST_CLASSIFICATION_LIVE_POLICY_ENABLED=true`, only objective high-confidence
routes can affect responses:

- `refuse_or_guardrail`
- `web_required`
- `ask_clarification`
- `deterministic_template`

Small-model, strong-model, and tool-required recommendations remain shadow-only.

## Operator Commands

```bash
orchestrator-maint classify-requests \
  [--repo REPO] \
  [--session SESSION_ID] \
  [--since TIMESTAMP] \
  [--dry-run] \
  [--repair] \
  [--batch-size N]

orchestrator-maint request-classification-report \
  [--repo REPO] \
  [--since TIMESTAMP]
```

Use `--repair` to recompute already-classified rows after deterministic rules
change. Startup backfill also repairs obviously stale rows where stored features
show `char_count = 0` but the current source event has request text.

## Environment

```text
REQUEST_CLASSIFICATION_STARTUP_BACKFILL_ENABLED=true
REQUEST_CLASSIFICATION_STARTUP_BACKFILL_BATCH_SIZE=500
REQUEST_CLASSIFICATION_LIVE_POLICY_ENABLED=false
REQUEST_CLASSIFICATION_POLICY_VERSION=v1
```

## Non-Goals

- No learned classifier in the critical path.
- No autonomous model routing for ambiguous requests.
- No raw prompt archival outside existing `agent_events`.
- No raw request text, paths, commands, URLs, or auth values as metric labels.
- No replacement of `agent_feature_records`.
- No production XGBoost integration in this track.
