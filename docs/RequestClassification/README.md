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

## Non-Goals

- No learned classifier in the critical path.
- No autonomous model routing for ambiguous requests.
- No raw prompt archival outside existing `agent_events`.
- No raw request text, paths, commands, URLs, or auth values as metric labels.
- No replacement of `agent_feature_records`.
- No production XGBoost integration in this track.
