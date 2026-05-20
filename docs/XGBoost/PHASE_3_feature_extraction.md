# Phase 3: Feature Extraction Body + Candidate ID Derivation + Bucket Helpers

Read `ARCHITECTURE.md` first. Every rule in it applies here.

Phase 1+2 must be complete. The enums, constants, structs, and feature extraction function signature already exist. Do not redefine them.

## Scope

Implement only the following in this phase:

1. Body of the `extract_candidate_features` function from Phase 1+2.
2. `derive_candidate_id` function — deterministic id derivation.
3. Bucket-assignment helper functions, one per bucketed field.
4. Tests listed below.

## Out of Scope For This Phase

- The bounded queue
- The writer
- Any wiring to the context assembler
- The outcome joiner
- The export command
- Modifying any enum, constant, or struct from Phase 1+2

## Candidate ID Derivation

Implement `derive_candidate_id` as a pure function:

```
fn derive_candidate_id(
    decision_id: &str,
    source_type: SourceType,
    source_record_id: &str,
    context_section: ContextSection,
) -> String
```

Rules:

- Use a stable hash function. SHA-256 truncated to 128 bits, hex-encoded, is acceptable. Document the choice in a comment.
- Field ordering in the hash input is fixed: `decision_id || "\x1f" || source_type_str || "\x1f" || source_record_id || "\x1f" || context_section_str`. The `\x1f` separator prevents field-boundary ambiguity.
- `source_type_str` and `context_section_str` are the lowercase enum variant names. Do not use integer discriminants — those can shift if enum order changes.
- The function is pure. No clock, no randomness, no global state.

## Bucket Helpers

Implement one bucket-assignment function per bucketed field:

- `token_cost_to_bucket(cost: u64) -> TokenCostBucket`
- `age_seconds_to_bucket(age: u64) -> AgeBucket` (if `AgeBucket` is its own enum) or returns the appropriate variant directly
- `retrieval_score_to_bucket(score: Option<f64>, decision_scores: &[f64]) -> RetrievalScoreBucket` — `none` if input is `None`, otherwise quartile within `decision_scores` (which must be the set of retrieval scores for all candidates in the same decision)
- `deterministic_score_to_bucket(score: f64) -> DeterministicScoreBucket`
- `latency_micros_to_bucket(latency: u64) -> RequestLatencyBucket`
- `input_tokens_to_bucket(tokens: u64) -> InputTokenBucket`
- `output_tokens_to_bucket(tokens: u64) -> OutputTokenBucket`

Rules for all bucket helpers:

- Boundaries come from the constants module. Do not redefine them.
- Boundary semantics must be documented in code: are boundaries inclusive on the low end or the high end? Pick one and apply consistently. Suggested: boundary value `B` means "values `<= B` fall in this bucket, values `> B` fall in the next."
- All helpers are pure functions. No side effects, no logging.
- Unknown or out-of-range inputs map to `unknown` and increment `ledger_unknown_enum_total{field}`. Do not raise.

## Feature Extraction Body

Implement the body of `extract_candidate_features`. The function:

- Takes the typed inputs from Phase 1+2 (source record reference, decision context).
- Calls the bucket helpers to derive bucketed feature values.
- Computes deterministic boolean features (`same_repo`, `same_session`, `same_trajectory`) by comparing against the decision context.
- Returns a fully-populated `CandidateFeatureVector`.

Rules:

- No floating point values in the output. All numeric features are integers or bucket enums.
- No raw strings from source records in the output. Only ids, enums, integers, booleans.
- Map iteration must use sorted keys if any internal map is traversed. Output must be deterministic for identical inputs regardless of input map ordering.
- The function is pure. No clock, no randomness, no I/O.

## Tests Required In This Phase

1. **Candidate id stability test.** Given fixed inputs, `derive_candidate_id` returns the same value across processes and runs. Include a hardcoded expected hash value in the test as a regression anchor — compute it once, hardcode it, fail loudly if it ever changes.

2. **Candidate id field-ordering test.** Verify that swapping `source_record_id` and `decision_id` in the input produces a different id. Verify that adding a trailing or leading whitespace to any field changes the id. This catches field-boundary bugs.

3. **Feature extraction replay equivalence test.** Build a `DecisionContext` and a source record. Call `extract_candidate_features` twice. Assert the two output vectors are byte-identical when serialized.

4. **Feature extraction map ordering test.** Build inputs where any internal map can be constructed with two different insertion orders. Call `extract_candidate_features` against each variant. Assert outputs are byte-identical.

5. **Bucket boundary test.** For each bucket helper, test the value exactly at each boundary and one above/below. Verify the documented semantics (inclusive on which end) hold.

6. **Unknown input test.** Pass deliberately invalid or out-of-range values to each bucket helper. Assert the output is `unknown` and the `ledger_unknown_enum_total` metric increments.

7. **No-float-in-output test.** Reflect over `CandidateFeatureVector` fields. Assert no field has a floating-point type.

## Hand-Verification Step

Before declaring this phase done, manually compute one `candidate_id` by hand from the formula and compare against what the code produces for the same inputs. If they disagree, the implementation has chosen a different hash or ordering than the spec. Do not proceed to Phase 4 until they match.

## Stop Condition

Stop after all functions are implemented and all tests pass.

Do not start Phase 4. Do not implement the writer. Do not implement the queue. Do not wire to the assembler.

Report:

- File paths for the new functions.
- The hardcoded expected hash value from test 1.
- Confirmation that all tests pass.
- Confirmation that the hand-verification step matched.
