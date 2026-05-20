# Phase 4: Bounded Queue + Transactional Writer

Read `ARCHITECTURE.md` first. Every rule in it applies here.

Phases 1+2 and 3 must be complete. Structs, enums, constants, feature extraction, candidate id, and bucket helpers all exist. Do not modify them.

## Scope

Implement only the following in this phase:

1. An in-process bounded channel for `DecisionBundle` items.
2. A writer task that drains the channel and persists bundles transactionally.
3. The narrow public interface the assembler will eventually call (just the interface — wiring is Phase 5).
4. All metrics for the write path.
5. Tests listed below.

## Out of Scope For This Phase

- Wiring to the context assembler
- The outcome joiner
- The export command
- Any modification to Phase 1-3 code

## Public Interface

Expose exactly one function for producers:

```
fn try_record_decision(bundle: DecisionBundle) -> RecordOutcome
```

Where `RecordOutcome` is an enum:

- `Queued` — bundle accepted into the channel
- `DroppedQueueFull` — channel was full at enqueue time
- `DroppedEnqueueTimeout` — enqueue exceeded the latency budget
- `DroppedShutdown` — writer is shutting down

Rules for this function:

- Non-blocking. Returns within the latency budget regardless of channel state.
- Latency budget: 1 ms p99 target. Use `try_send` semantics on the channel; do not use blocking `send`.
- Never panics on the producer side.
- Never logs at producer-call time (logging may happen on the writer side asynchronously).

This is the only function the assembler will eventually import from the ledger module. Nothing else in this phase is part of the public interface.

## Channel

Use an in-process bounded channel. Capacity comes from an environment variable with a documented default (suggested: `CONTEXT_LEDGER_QUEUE_CAPACITY=1024`).

Rules:

- Channel item type is `DecisionBundle`, the whole bundle as one item.
- Bounded. Never grow without limit.
- No network dependency. No Redis, no external queue. In-process only.
- No disk spill. If full, drop.

## Writer Task

A background task drains the channel and persists bundles.

Rules for the writer:

- One bundle per database transaction. Envelope insert + all candidate inserts in the same transaction.
- On transaction success: increment `context_ledger_bundles_total{outcome="written"}`.
- On transaction failure: log at WARN, increment `context_ledger_bundles_total{outcome="dropped_error"}`. Do not retry the bundle. Do not block the writer. Move on to the next bundle.
- The writer does not call feature extraction. Feature extraction happens on the producer side before enqueue. The writer only persists.
- On shutdown signal: stop accepting new items, drain remaining items up to a documented timeout, then exit. Items not drained within the timeout are dropped and counted.

## Metrics

Define and emit the following metrics. All labels are bounded enums; no user-controlled values as labels.

- `context_ledger_bundles_total{outcome}` — counter. Outcomes: `queued`, `written`, `dropped_queue_full`, `dropped_enqueue_timeout`, `dropped_error`, `dropped_shutdown`.
- `context_ledger_candidates_total{source_type,injected}` — counter. Incremented per candidate persisted. `injected` is `"true"` or `"false"` as label strings.
- `context_ledger_unknown_enum_total{field}` — counter. Already used by Phase 3 bucket helpers. Ensure the writer increments it for any enum normalization done at write time.
- `context_ledger_write_duration_seconds` — histogram. Wall time per transaction.
- `context_ledger_queue_depth` — gauge. Current channel depth.

## Cache-Hit Handling

The writer must handle envelopes with zero candidates correctly. Per the architecture doc, cache-hit envelopes may have no candidate rows. The transaction must still commit.

The writer must accept envelopes whose `source_decision_id` references a `decision_id` that no longer exists in the table. `source_decision_id` is not a foreign key. The writer must not validate it against existing rows. Retention may have deleted the source.

## Tests Required In This Phase

All tests in this phase run against the writer in isolation, with mock `DecisionBundle` inputs. Do not test through the assembler — that is Phase 5.

1. **Queue saturation test.** Fill the channel to capacity. Call `try_record_decision` once more. Assert it returns `DroppedQueueFull` within the latency budget. Assert the producer is not blocked. Assert `context_ledger_bundles_total{outcome="dropped_queue_full"}` incremented.

2. **Enqueue latency test.** Under nominal load, measure `try_record_decision` duration over many calls. Assert p99 is under the documented budget.

3. **Transactionality test (success).** Submit a bundle with envelope + N candidates. Wait for write. Assert the envelope row and all N candidate rows are present in the DB.

4. **Transactionality test (failure).** Inject a DB error on the candidate insert step. Submit a bundle. Wait. Assert neither the envelope nor any candidate row is present. Assert `context_ledger_bundles_total{outcome="dropped_error"}` incremented.

5. **Cache-hit dangling reference test.** Submit a bundle whose envelope has `source_decision_id` set to a value that does not exist in the envelopes table. Assert the bundle persists successfully. Assert no FK error.

6. **Cache-hit empty candidates test.** Submit a bundle with envelope and zero candidates. Assert the envelope persists. Assert the transaction commits.

7. **Unknown enum metric test.** Submit a bundle with an enum value that requires normalization to `unknown`. Assert the row persists with the `unknown` variant. Assert `context_ledger_unknown_enum_total{field=...}` incremented for the correct field.

8. **Shutdown drain test.** Fill the channel partially. Signal shutdown. Assert all items are drained within the documented timeout, OR remaining items are counted as `dropped_shutdown`. Assert no items are silently lost.

9. **Producer non-blocking test.** Spin up the writer with a deliberately slow DB. Call `try_record_decision` repeatedly. Assert producer call latency stays within budget even as the writer falls behind.

## Stop Condition

Stop after the writer is working in isolation and all nine tests pass.

Do not start Phase 5. Do not import or call any context assembler code. Do not modify the assembler.

Report:

- The public interface signature (`try_record_decision`).
- Channel capacity default and env var name.
- Writer module path.
- Confirmation that all nine tests pass.
- Measured p99 enqueue latency from test 2.
