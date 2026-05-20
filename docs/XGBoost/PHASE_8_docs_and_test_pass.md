# Phase 8: Docs + Metrics Wrap + Full Test Pass

Read `ARCHITECTURE.md` first. Every rule in it applies here.

Phases 1+2 through 7 must be complete. The ledger is recording envelopes, candidates, and outcomes; the export command works. Do not introduce new functionality in this phase.

## Scope

Implement only the following in this phase:

1. Verify all metrics from prior phases are wired and emit correctly with bounded labels.
2. Write operator-facing documentation.
3. Run the full test suite and fix any cross-phase invariant violations.

## Out of Scope For This Phase

- New schema
- New enum variants
- New metrics beyond verification
- New CLI flags
- Shadow scoring (still not in this work)

## Metrics Verification

Confirm every metric below exists, is emitted at the correct site, has only bounded labels, and has no user-controlled label values.

- `context_ledger_bundles_total{outcome}` — outcomes: `queued`, `written`, `dropped_queue_full`, `dropped_enqueue_timeout`, `dropped_error`, `dropped_shutdown`
- `context_ledger_candidates_total{source_type,injected}`
- `context_ledger_unknown_enum_total{field}`
- `context_ledger_write_duration_seconds`
- `context_ledger_queue_depth`
- `context_ledger_outcomes_joined_total{outcome}`
- `context_ledger_outcomes_closed_total{had_signal}`

For each metric, produce a one-line description of when it fires and what the labels mean. Save this to the docs.

## Documentation

Write `docs/context_ledger.md` covering:

1. **Purpose.** Two sentences. What the ledger is for, what it records.
2. **Tables.** One section per table. Field list with one-line descriptions. Mark which fields are nullable. Mark which fields are append-only vs upserted.
3. **Enums and buckets.** List every enum, its variants, and a one-line description per variant. List every bucket and its boundary values.
4. **Versioning.** Explain `FEATURE_SCHEMA_VERSION` and `CONTEXT_POLICY_VERSION`. State that any enum or bucket boundary change requires a version bump and that a test enforces this.
5. **Privacy invariants.** Restate the privacy rules from the architecture doc. List the allowlist of unbounded strings.
6. **Operational behavior.** Describe the queue, the drop policy, the writer transaction semantics, the outcome window and closer cadence.
7. **Running the export.** Show every CLI flag with an example. Show the manifest format. Show a sample row.
8. **Metrics.** The list from the verification section.
9. **Failure modes.** Document what happens when:
   - The queue is full.
   - The DB is unavailable.
   - An unknown enum variant arrives.
   - The closer runs while the joiner is upserting.
   - The export is run while live writes are happening.
10. **What the ledger is not.** A short section reiterating non-goals: not a ranker, not a scorer, not a source of truth for prompts, not a backfill target.

## Full Test Pass

Run the complete test suite from all prior phases. Every test must pass. List of tests by phase:

- **Phase 1+2:** schema privacy test, enum-version-bump test.
- **Phase 3:** candidate id stability, field-ordering, replay equivalence, map ordering, bucket boundary, unknown input, no-float-in-output.
- **Phase 4:** queue saturation, enqueue latency, transactionality success, transactionality failure, cache-hit dangling reference, cache-hit empty candidates, unknown enum metric, shutdown drain, producer non-blocking.
- **Phase 5:** end-to-end success, inject/drop XOR, dense rank, budget accounting, failed-assembly, cache-hit, module boundary, call site coverage.
- **Phase 6:** idempotency, window closure no events, window closure partial events, no silent reopen, no prediction, no inference, bucket-only, race.
- **Phase 7:** export determinism, no raw text, no identifiers, no raw timestamps, dry-run, open-outcomes exclusion, cross-repo default, schema version filter, sort order, privacy schema.

If any test fails, identify the phase that introduced the regression, fix it there, and re-run all tests. Do not paper over failures with skipped tests or relaxed assertions.

## Additional Verification Tasks

1. **Privacy schema sweep.** Re-run the privacy schema test from Phase 1+2 against the current schema. Confirm no new unbounded string fields have been added across phases without being added to the allowlist.

2. **Enum surface sweep.** Confirm every enum still has an `unknown` variant. Confirm the enum hash file from Phase 1+2 matches the current enums (or has been updated alongside a `FEATURE_SCHEMA_VERSION` bump).

3. **Module boundary sweep.** Re-run the module boundary test from Phase 5. Confirm no new imports from the assembler into the ledger have crept in.

4. **No-shadow-scoring sweep.** Grep the codebase for any references to predictive scoring, learned ranking, or shadow scoring within the ledger module. Confirm none exist.

5. **Production safety sweep.** Confirm the assembler's prompt assembly output is unchanged by the ledger. A test that runs the assembler with the ledger enabled and disabled and diffs the resulting prompts is appropriate here.

## Stop Condition

Stop after:

- All tests from all phases pass.
- All five additional verification sweeps pass.
- `docs/context_ledger.md` is complete.
- The metrics verification one-liners are written.

Do not add new functionality. Do not refactor.

Report:

- Test count by phase, all passing.
- Docs file path.
- Result of each verification sweep.
- Final confirmation: "Ledger is recording. Assembler behavior is unchanged. Ready for offline ranker development."
