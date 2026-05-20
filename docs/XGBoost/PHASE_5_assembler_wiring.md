# Phase 5: Wire One Narrow Context-Assembler Call Site

Read `ARCHITECTURE.md` first. Every rule in it applies here.

Phases 1+2, 3, and 4 must be complete. The schema, structs, feature extraction, candidate id, bucket helpers, queue, and writer all exist and are tested in isolation. Do not modify them.

## Scope

Implement only the following in this phase:

1. Identify all places in the context assembler that produce a context pack. There may be multiple paths (success, cache hit, fallback). All of them must produce a `DecisionBundle`.
2. At each call site, after the assembler has finalized its decisions, construct a `DecisionBundle` and call `try_record_decision(bundle)`.
3. Do not modify any selection, ranking, or budget logic in the assembler.
4. Tests listed below.

## Out of Scope For This Phase

- Any change to assembler selection logic
- Any change to ranker behavior
- Any change to budget logic
- The outcome joiner
- The export command
- New metrics beyond what Phase 4 already defined
- New enum variants

If a change to assembler internals seems necessary, stop and report. Do not modify them.

## Identifying All Call Sites

Before writing any code, enumerate all functions in the assembler that produce a context pack. Look for:

- The primary assembly path.
- The cache-hit return path.
- Any fallback or degraded-mode path.
- Any error-path that still returns a partial context pack.

Produce a written list of every site. Each site must either:

- Construct a `DecisionBundle` and call `try_record_decision`, or
- Be documented in code comments explaining why no bundle is emitted (only acceptable reason: this path never produces a context pack at all).

Missing a call site is the most likely bug in this phase. The written list is the safeguard.

## Bundle Construction Rules

At each call site, construct the `DecisionBundle` from data the assembler already has. Do not re-run retrieval. Do not re-rank. Do not re-extract features from raw text.

For each candidate that reached the deterministic ranker input, call `extract_candidate_features` from Phase 3 to produce its feature vector. Populate the `DecisionCandidate` struct with:

- `candidate_id` derived via `derive_candidate_id`.
- `decision_id` from the envelope.
- All fields per the schema definition.
- `injected = true` and `representation_selected != none` if the candidate was selected.
- `injected = false`, `drop_reason` set, `representation_selected = none` if the candidate was dropped.

For the envelope:

- `assembly_status = success` if the assembler completed normally.
- `assembly_status = partial` if the assembler returned a degraded pack.
- `assembly_status = failed` if the assembler raised but the call site still wants to record the attempt. In this case `candidate_count = 0`, `injected_candidate_count = 0`, no candidate rows.
- `cache_status` reflects which path produced the pack.
- `source_decision_id` set when the pack was reused from cache and the prior `decision_id` is known.

## Assembly Status Rules (recap)

- `success`: budget equation must hold (`candidate_tokens_total + fixed_overhead_tokens == final_estimated_context_tokens`). All candidates that reached ranker input have rows.
- `partial`: budget equation is best-effort. Some candidates may be missing from the rows; `candidate_count` reflects only what was captured.
- `failed`: envelope only, no candidate rows. `candidate_count = 0`.

## Module Boundary

The ledger module exposes `try_record_decision` and the public types (`DecisionBundle`, `DecisionEnvelope`, `DecisionCandidate`, the public enums) only. The assembler imports those.

The ledger module must not import anything from the assembler. This is enforced by test 7 below.

## Tests Required In This Phase

1. **End-to-end success test.** Run the assembler on a fixture input that produces a normal context pack. Assert one envelope row is written. Assert one candidate row per candidate that reached ranker input. Assert the budget equation holds. Assert `injected_candidate_count` matches the number of rows with `injected = true`.

2. **Inject/drop XOR test.** From the rows persisted in test 1, assert every candidate row satisfies `(injected = true) XOR (drop_reason IS NOT NULL)`.

3. **Dense rank test.** From the rows persisted in test 1, assert `deterministic_rank` values are dense (1, 2, 3, ... with no gaps) and unique within the decision.

4. **Budget accounting test.** From the rows persisted in test 1, assert sum of `estimated_token_cost` over injected candidates plus `fixed_overhead_tokens` equals `final_estimated_context_tokens` in the envelope.

5. **Failed-assembly test.** Force the assembler into a failure path that still records an attempt. Assert one envelope row with `assembly_status = failed`, `candidate_count = 0`, and zero candidate rows.

6. **Cache-hit test.** Force a cache hit. Assert one envelope row with `cache_status = hit_fresh` or `hit_stale` and `source_decision_id` set. Assert zero candidate rows (since no fresh ranker input was produced). Assert no error from the dangling reference rule.

7. **Module boundary test.** A static check (compile-time, lint, or grep-based test) that fails if any file in the `context_ledger` module imports from the assembler module. The assembler may import from the ledger; the ledger may not import from the assembler.

8. **Call site coverage test.** A meta-test or documented manual checklist asserting every assembler path that produces a context pack has a corresponding `try_record_decision` call. The written list from the scope section is the source of truth.

## Hand-Verification Step

After implementation, search the assembler codebase for all functions that return a context pack type. For each one, verify either:

- A `try_record_decision` call exists on the path, or
- A code comment explains why no bundle is emitted.

If any path is missing both, fix before declaring the phase done.

## Stop Condition

Stop after the assembler emits decision bundles for live traffic and all eight tests pass.

Do not start Phase 6. Do not implement the outcome joiner. Do not implement export.

Report:

- The written list of all assembler call sites and which now emit bundles.
- Diff summary: files changed in this phase. Should be the assembler call sites and nothing else.
- Confirmation that all eight tests pass.
- Confirmation that the hand-verification step found no missing call sites.
