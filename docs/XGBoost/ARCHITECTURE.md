# Shadow Context Decision Ledger — Architecture

This document is the anchor for all implementation phases. It defines invariants that must hold across every phase. Read it before doing any work in any phase.

## Goal

Add a deterministic, privacy-safe ledger that records how context packs are assembled so future learned rankers can be trained offline.

This work records data only. It must not change prompt assembly, context ranking, model routing, or live behavior.

## Non-Goals

- No production learned ranking
- No shadow scoring in this pass
- No LLM calls
- No prompt-body archival
- No raw user message storage
- No raw tool-output storage beyond existing agent_events
- No changes to deterministic context selection
- No backfill of old context assemblies
- No cross-repo export by default
- No new network dependency for ledger writes

## Core Design

The `context_ledger` subsystem has three responsibilities:

1. Accept complete decision bundles from the context assembler.
2. Persist envelopes and candidates transactionally.
3. Export deterministic training rows later.

The context assembler calls a narrow ledger interface with already-built decision data. The ledger must not re-run retrieval, ranking, or selection from raw text.

Writes use an in-process bounded channel. The queue item is the complete decision bundle. The writer either persists the whole bundle in one transaction or drops the whole bundle. Partial ledger writes are not allowed.

If the queue is full or enqueue exceeds the latency budget, drop the whole bundle and increment a metric. Request handling continues unchanged.

## Determinism Rules (invariant across all phases)

1. All enum fields must have an `unknown` variant.
2. Unknown enum input maps to `unknown` and increments `ledger_unknown_enum_total{field}`.
3. Unknown enum input must not drop the row.
4. Sort orders must be total and documented.
5. Ties must break by `candidate_id` ASC.
6. Map iteration order must not affect output.
7. Raw floats must not be stored as features.
8. Floating values must be bucketed before storage.
9. Stored timestamps must be UTC integer microseconds.
10. Time-derived features use integer seconds.
11. Feature bucket boundaries are constants in code.
12. Changing any enum, bucket, or feature definition requires bumping `FEATURE_SCHEMA_VERSION`.

## Privacy Rules (invariant across all phases)

Ledger tables must not store:

- prompt bodies
- raw user text
- raw assistant text
- raw tool output
- file contents
- auth values
- endpoints as labels
- free-form summaries as features

Allowed unbounded strings:

- opaque ids such as `decision_id`, `candidate_id`, `source_record_id`
- `repo`
- `session_id`
- `trajectory_id`

Allowed bounded free text:

- `drop_reason_other_detail`, max 256 chars, truncated silently, never exported as a feature

Everything else must be integer, boolean, timestamp micros, JSON of bounded feature fields, or closed enum.

## Candidate Definition

A candidate is "considered" only if it reaches the deterministic ranker input.

Items rejected before ranker input are not ledger candidates in this pass.

## Cache Hit Rule

For cache hits, write an envelope with `cache_status`.

If the cache hit reuses a previous context pack, the envelope includes `source_decision_id`.

`source_decision_id` is a soft reference only. It must not be a foreign key. Retention may delete the source decision, and readers must tolerate dangling references.

Candidate rows are not required for cache-hit envelopes when no fresh ranker input was produced.

## Module Boundary Rule

The ledger writer module must not import the context assembler's internal selection logic. The assembler imports the ledger's narrow public interface only. This boundary is enforced by a test.

## Tables

Three tables. Full schemas are defined in Phase 1+2. Other phases reference these by name only.

- `context_decision_envelopes` — one row per context-pack assembly attempt. Append-only.
- `context_decision_candidates` — one row per candidate that reached ranker input. Append-only.
- `context_decision_outcomes` — one row per decision, upserted asynchronously. Idempotent upserts.

## Implementation Order

Phases run sequentially. Do not skip ahead. Do not implement future phases.

1. **Phase 1+2 (fused):** Schema migration + enums + constants + structs + feature extraction function signature.
2. **Phase 3:** Feature extraction body + candidate id derivation + bucket helpers.
3. **Phase 4:** Bounded queue + transactional writer + metrics for write path.
4. **Phase 5:** Wire one narrow context-assembler call site.
5. **Phase 6:** Outcome joiner + window closer.
6. **Phase 7:** Export command.
7. **Phase 8:** Docs + metrics wrap + full test pass.

Do not implement shadow scoring in this work at all. Not even stubs. Not even "for later" hooks.

## Success Definition

After deployment, for every new context-pack decision, operators can reconstruct:

- what was considered
- what was injected
- what was dropped
- why it was dropped
- what each item cost
- which feature schema and context policy were used
- what outcome later attached

And the system can export deterministic, privacy-safe rows suitable for offline learned ranker development without affecting live prompt assembly.
