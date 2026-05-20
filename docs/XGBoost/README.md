# Shadow Context Decision Ledger — Implementation Package

This directory contains the chunked spec for handing the ledger work to a 35B MoE.

## Files

- `ARCHITECTURE.md` — Cross-phase invariants. Prepend to every phase prompt.
- `PHASE_1_2_schema_enums_structs.md` — Migration, enums, constants, structs, feature extraction signature.
- `PHASE_3_feature_extraction.md` — Feature extraction body, candidate id, bucket helpers.
- `PHASE_4_queue_and_writer.md` — Bounded queue, transactional writer, metrics.
- `PHASE_5_assembler_wiring.md` — One narrow call site in the assembler.
- `PHASE_6_outcome_joiner.md` — Outcome joiner and window closer.
- `PHASE_7_export.md` — Export CLI command.
- `PHASE_8_docs_and_test_pass.md` — Docs, metrics verification, full test pass.

## How To Use

For each phase, send the model:

1. The full `ARCHITECTURE.md` as system context or first user message.
2. The phase-specific prompt as the actual task.

Example for Phase 3:

```
[system or first user message]
<contents of ARCHITECTURE.md>

[task]
<contents of PHASE_3_feature_extraction.md>
```

Do not send multiple phases at once. Do not send the next phase until the current one is complete and verified.

## Between Phases

Before starting phase N+1:

1. Run the tests from phase N. Confirm they pass.
2. Skim the diff. Look specifically for:
   - Enum variants that lost their `unknown` variant
   - Floats stored where buckets should be
   - Unbounded string fields not in the privacy allowlist
   - New imports between ledger and assembler modules
3. Run the privacy schema sweep manually if the model added any new fields.
4. Tick off the invariant checklist (below) for any rules touched by this phase.

## Invariant Checklist

Keep this in a text file outside the model. Tick off each as you verify it in the code so far. The model cannot see this; you are the only verifier.

Determinism:
- [ ] All enums have `unknown` variant
- [ ] Unknown enum input maps to `unknown` + metric, never drops row
- [ ] Sort orders are total and documented
- [ ] Ties break by `candidate_id` ASC
- [ ] Map iteration does not affect output
- [ ] No raw floats stored as features
- [ ] All stored timestamps are UTC integer microseconds
- [ ] Bucket boundaries are code constants
- [ ] `FEATURE_SCHEMA_VERSION` bumps with any enum/bucket change

Privacy:
- [ ] No prompt bodies in any table
- [ ] No raw user/assistant text in any table
- [ ] No raw tool output in any table
- [ ] No file contents in any table
- [ ] No auth values in any table
- [ ] No endpoints as metric labels
- [ ] No free-form summaries as features
- [ ] Allowlist of unbounded strings matches architecture doc
- [ ] `drop_reason_other_detail` capped at 256 chars, never in export

Module boundary:
- [ ] Ledger does not import assembler internals
- [ ] Assembler imports only the ledger's narrow interface

Shadow scoring:
- [ ] No predictive scoring code anywhere in the ledger module
- [ ] No "for later" hooks or stubs

## Hand-Verification Checkpoints

Three places to stop and verify by hand:

1. **After Phase 3:** Compute one `candidate_id` by hand using the documented formula. Compare to what the code produces for identical inputs. Must match exactly.

2. **After Phase 5:** Search the assembler codebase for every function that returns a context pack type. For each one, confirm a `try_record_decision` call exists on the path, or a code comment explains why not.

3. **After Phase 8:** Diff the assembler's prompt output with the ledger enabled vs disabled on identical inputs. Must be byte-identical. If it differs, the ledger has changed behavior somewhere — find it and fix it.

## Recovery

If a phase fails or the model produces something obviously wrong:

1. Do not proceed to the next phase.
2. Identify which invariant was violated.
3. Re-prompt the model with just that phase's spec plus a note: "the previous attempt violated invariant X. Fix only that. Do not regenerate the entire phase."
4. Re-run tests.

If the same phase fails three times, the spec for that phase is likely under-constrained for the model. Reduce scope further: split the phase in half, verify the first half, then send the second half as a new prompt.

## What Not To Do

- Do not let the model "improve" the assembler.
- Do not let the model add shadow scoring stubs.
- Do not let the model add retry logic to the writer.
- Do not let the model add network dependencies for ledger writes.
- Do not let the model add new enum variants without bumping `FEATURE_SCHEMA_VERSION`.
- Do not skip the hand-verification checkpoints.
- Do not run multiple phases in parallel.
