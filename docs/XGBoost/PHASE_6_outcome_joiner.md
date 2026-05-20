# Phase 6: Outcome Joiner + Window Closer

Read `ARCHITECTURE.md` first. Every rule in it applies here.

Phases 1+2, 3, 4, and 5 must be complete. The ledger is recording envelopes and candidates from live traffic. Do not modify any of that.

## Scope

Implement only the following in this phase:

1. The outcome joiner — a background process that observes system events and upserts outcome rows into `context_decision_outcomes`.
2. The window closer — marks decisions closed after the join window even when no events arrived.
3. Idempotency for both.
4. Tests listed below.

## Out of Scope For This Phase

- The export command
- Any new schema
- Any modification to envelopes or candidates
- Any LLM call
- Any predictive or scoring logic — outcomes derive only from observed system events

## Outcome Joiner

A background process watches for outcome-relevant events and upserts into `context_decision_outcomes` keyed by `decision_id`.

Events the joiner must observe (use existing event sources; do not invent new ones):

- Trajectory completion events → `trajectory_outcome`
- User correction events → `user_correction_after_decision`
- Retry events → `retry_after_decision`
- Validation events → `validation_result`
- Tool loop detection events → `tool_loop_detected_after_decision`
- Request completion events → `request_latency_bucket`, `input_token_bucket`, `output_token_bucket`

Rules:

- Upserts are idempotent. Running the joiner twice over the same events produces the same outcome rows.
- The joiner does not infer outcomes from absence of events. Missing data is missing, not "succeeded by default."
- The joiner does not write predictions. Only observed signals.
- On first upsert for a decision, create the row with `outcome_joined_at_micros` set to the current time and `outcome_window_closed = false`.
- On subsequent upserts, update the relevant fields and update `outcome_joined_at_micros`.
- All numeric values are bucketed via the helpers from Phase 3 before storage. No raw latencies or token counts.

## Window Closer

A periodic task that closes decisions after the join window has elapsed.

Rules:

- The join window is `OUTCOME_JOIN_WINDOW_MICROS` from the constants module (default 24 hours).
- For each decision where `created_at_micros + OUTCOME_JOIN_WINDOW_MICROS < now()` and no outcome row exists: create an outcome row with all enum fields set to `unknown`, all nullable booleans set to NULL, all bucket fields set to the bucket variant representing "unknown" (or the explicit `unknown` variant where defined), `outcome_window_closed = true`, `outcome_joined_at_micros` set to current time.
- For each decision where an outcome row exists with `outcome_window_closed = false` and the window has elapsed: set `outcome_window_closed = true`. Do not modify other fields. Do not overwrite real outcomes with `unknown`.
- The closer runs on a schedule (cron or interval task). Frequency is documented (suggested: every 10 minutes).
- The closer is idempotent. Running it twice produces the same rows.

## Joiner / Closer Coordination

The joiner and closer may run concurrently. Use database-level upsert semantics (e.g., `INSERT ... ON CONFLICT ... DO UPDATE`) to avoid races. Last-writer-wins is acceptable as long as:

- A closed window is never silently reopened.
- A real outcome value is never overwritten by `unknown`.

Document the conflict resolution rule in code.

## Metrics

Add:

- `context_ledger_outcomes_joined_total{outcome}` — counter. Outcomes are `trajectory_outcome` values: `succeeded`, `unresolved`, `abandoned`, `reverted`, `unknown`.
- `context_ledger_outcomes_closed_total{had_signal}` — counter. `had_signal` is `"true"` if any real outcome data was present at closure, `"false"` if the window closed with no signal at all.

## Tests Required In This Phase

1. **Idempotency test.** Run the joiner over the same event log twice. Assert the outcome rows are identical after both runs.

2. **Window closure with no events test.** Create an envelope with `created_at_micros` set to a value older than the join window. Run the closer. Assert an outcome row exists with all enums set to `unknown` and `outcome_window_closed = true`.

3. **Window closure with partial events test.** Create an envelope older than the window. Emit only a trajectory completion event. Run the joiner, then the closer. Assert the outcome row has the real `trajectory_outcome` and `unknown` for fields with no events. Assert `outcome_window_closed = true`.

4. **No silent reopen test.** Close a decision via the closer. Then emit a late outcome event for that decision. Assert `outcome_window_closed` remains `true`. Document the behavior of late events: either they update the row and the closed flag stays true, or they are dropped — pick one and assert it.

5. **No prediction test.** Inspect the joiner and closer code. Assert no learned model or predicted value is written to outcome fields. (Static check or code review enforced by lint.)

6. **No inference test.** For a decision with zero events, run the joiner without the closer. Assert no outcome row is created. Outcomes are not inferred from absence.

7. **Bucket-only test.** Inspect outcome rows after joiner runs. Assert no raw latency, token count, or float values are stored. All numeric fields are bucket enums.

8. **Race test.** Run joiner and closer concurrently on the same decision near the window boundary. Assert no row corruption. Assert the final state is consistent with the conflict resolution rule documented in code.

## Stop Condition

Stop after the joiner and closer are running, outcomes are being written from live traffic, and all eight tests pass.

Do not start Phase 7. Do not implement the export command.

Report:

- Joiner module path.
- Closer module path and schedule.
- Documented conflict resolution rule.
- Documented late-event behavior (test 4).
- Confirmation that all eight tests pass.
