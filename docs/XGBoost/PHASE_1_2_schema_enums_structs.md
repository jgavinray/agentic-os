# Phase 1+2: Schema + Enums + Constants + Structs

Read `ARCHITECTURE.md` first. Every rule in it applies here.

## Scope

Implement only the following in this phase:

1. Database migration creating three tables: `context_decision_envelopes`, `context_decision_candidates`, `context_decision_outcomes`.
2. All enum definitions with their variants, including `unknown` on every enum.
3. Constants module with version markers and bucket boundaries.
4. In-memory structs that mirror the table schemas field-for-field.
5. The feature extraction function signature only — function body is `unimplemented!()` or equivalent placeholder.

## Out of Scope For This Phase

- Function bodies (except trivial accessors)
- The bounded queue
- The writer
- Any wiring to the context assembler
- The outcome joiner
- The export command
- Any tests beyond the two specified below

## Table 1: `context_decision_envelopes`

One row per context-pack assembly attempt, including cache hits and assembly failures. Append-only.

Required fields:

- `decision_id` — opaque id, primary key
- `source_decision_id` — nullable, soft reference only, NOT a foreign key
- `repo`
- `session_id` — nullable
- `trajectory_id` — nullable
- `request_route` — enum
- `task_type` — enum
- `created_at_micros` — UTC integer microseconds
- `context_policy_version` — string
- `feature_schema_version` — integer
- `total_token_budget` — integer
- `candidate_tokens_total` — integer
- `fixed_overhead_tokens` — integer
- `final_estimated_context_tokens` — integer
- `cache_status` — enum: `miss`, `hit_fresh`, `hit_stale`, `bypass`, `unknown`
- `candidate_count` — integer
- `injected_candidate_count` — integer
- `assembly_duration_micros` — integer
- `assembly_status` — enum: `success`, `partial`, `failed`, `unknown`

Invariants the schema must support (enforced by code, not DB constraints, except where noted):

- When `assembly_status = success`: `candidate_tokens_total + fixed_overhead_tokens == final_estimated_context_tokens`.
- When `assembly_status = failed`: `candidate_count = 0` and `injected_candidate_count = 0`.
- `source_decision_id` is never a foreign key.

## Table 2: `context_decision_candidates`

One row per candidate that reached deterministic ranker input. Append-only.

Required fields:

- `candidate_id` — deterministic id, primary key
- `decision_id` — foreign key to `context_decision_envelopes.decision_id`
- `source_type` — enum: `raw_event`, `summary`, `failure_history`, `operational_constraint`, `retrieval_hit`, `feature_record`, `unknown`
- `source_record_id`
- `event_type` — enum nullable, includes `unknown`
- `summary_level` — enum: `none`, `compact`, `full`, `rolled_up`, `unknown`
- `age_seconds` — integer
- `same_repo` — boolean
- `same_session` — boolean
- `same_trajectory` — boolean
- `estimated_token_cost` — integer
- `token_cost_bucket` — enum
- `retrieval_score_bucket` — enum: `none`, `q1`, `q2`, `q3`, `q4`, `unknown`
- `failure_class` — enum nullable, includes `unknown`
- `operational_constraint_type` — enum nullable, includes `unknown`
- `context_section` — enum
- `duplicate_coverage` — enum: `none`, `partial`, `full`, `unknown`
- `deterministic_score_bucket` — enum: `none`, `very_low`, `low`, `medium`, `high`, `very_high`, `unknown`
- `deterministic_rank` — integer
- `budget_before_candidate` — integer
- `budget_after_candidate` — integer
- `injected` — boolean
- `drop_reason` — enum nullable: `priority_cap`, `token_budget`, `duplicate`, `stale`, `low_relevance`, `section_disabled`, `recovered`, `superseded`, `other`, `unknown`
- `drop_reason_other_detail` — nullable string, max 256 chars
- `representation_selected` — enum: `none`, `compact`, `full`, `unknown`

Invariants the schema must support:

- `(injected = true) XOR (drop_reason IS NOT NULL)` for every row.
- Injected candidates have `representation_selected != none`.
- Dropped candidates have `representation_selected = none`.
- `deterministic_rank` is dense and unique within a `decision_id`.

## Table 3: `context_decision_outcomes`

One row per decision, upserted asynchronously, keyed by `decision_id`.

Required fields:

- `decision_id` — primary key
- `trajectory_outcome` — enum: `succeeded`, `unresolved`, `abandoned`, `reverted`, `unknown`
- `user_correction_after_decision` — nullable boolean
- `retry_after_decision` — nullable boolean
- `validation_result` — enum: `passed`, `failed`, `not_run`, `unknown`
- `tool_loop_detected_after_decision` — nullable boolean
- `request_latency_bucket` — enum
- `input_token_bucket` — enum
- `output_token_bucket` — enum
- `outcome_joined_at_micros` — nullable integer
- `outcome_window_closed` — boolean
- `feature_schema_version` — integer

## Constants Module

Define these as code constants in a single module:

- `FEATURE_SCHEMA_VERSION` — integer, start at 1
- `CONTEXT_POLICY_VERSION` — string, start at "1.0.0"
- `TOKEN_COST_BUCKET_BOUNDARIES` — array of integer thresholds. Suggested: `[50, 200, 500, 1000, 2000]` producing buckets `0-50`, `51-200`, `201-500`, `501-1000`, `1001-2000`, `2001+`. Document the mapping from threshold array to bucket variant.
- `AGE_BUCKET_BOUNDARIES_SECONDS` — array. Suggested: `[60, 600, 3600, 21600, 86400, 604800]` producing `<1m`, `<10m`, `<1h`, `<6h`, `<24h`, `<7d`, `older`.
- `DETERMINISTIC_SCORE_BUCKET_BOUNDARIES` — array producing `very_low`, `low`, `medium`, `high`, `very_high`. Document the boundary values.
- `LATENCY_BUCKET_BOUNDARIES_MICROS` — array producing buckets for `request_latency_bucket`.
- `INPUT_TOKEN_BUCKET_BOUNDARIES` — array for `input_token_bucket`.
- `OUTPUT_TOKEN_BUCKET_BOUNDARIES` — array for `output_token_bucket`.
- `OUTCOME_JOIN_WINDOW_MICROS` — integer, default 24 hours in microseconds (`86_400_000_000`).

All bucket boundary arrays must be const, not configurable at runtime. Changing any of them requires bumping `FEATURE_SCHEMA_VERSION` (enforced by the test described below).

## Enum Definitions

All enums are closed and include an `unknown` variant. List all variants explicitly in code. Do not derive enum sets from external data.

Define these enums (variants listed above in table sections):

- `RequestRoute` — closed enum, must include `unknown`. List all routes in use.
- `TaskType` — closed enum, must include `unknown`.
- `CacheStatus`
- `AssemblyStatus`
- `SourceType`
- `EventType`
- `SummaryLevel`
- `TokenCostBucket`
- `RetrievalScoreBucket`
- `FailureClass`
- `OperationalConstraintType`
- `ContextSection`
- `DuplicateCoverage`
- `DeterministicScoreBucket`
- `DropReason`
- `RepresentationSelected`
- `TrajectoryOutcome`
- `ValidationResult`
- `RequestLatencyBucket`
- `InputTokenBucket`
- `OutputTokenBucket`

## Structs

Define in-memory structs that mirror each table schema exactly, field-for-field, with matching types. Name them:

- `DecisionEnvelope`
- `DecisionCandidate`
- `DecisionOutcome`

Also define:

- `DecisionBundle { envelope: DecisionEnvelope, candidates: Vec<DecisionCandidate> }` — the unit the writer persists or drops as a whole.

## Feature Extraction Signature

Define the function signature only. Body is a placeholder.

```
fn extract_candidate_features(<typed inputs>) -> CandidateFeatureVector
```

The inputs must be types you have already defined (source record reference, decision context, etc.). The return type `CandidateFeatureVector` is a struct containing only the bounded feature fields used in the candidate row (buckets, booleans, enums, integers). Define this struct.

Do not implement the body. Leave it as `unimplemented!()` or equivalent.

## Tests Required In This Phase

Only two tests in this phase. Both are static checks against the schema and enum definitions.

1. **Schema privacy test.** Walks the column definitions of all three tables. Fails if any column is typed as unbounded string except those in the architecture allowlist (`decision_id`, `candidate_id`, `source_record_id`, `repo`, `session_id`, `trajectory_id`, `source_decision_id`, `context_policy_version`, `drop_reason_other_detail`). The test must enumerate the allowlist explicitly, not infer it.

2. **Enum-version-bump test.** Hashes the set of enum variants for every enum defined in this module and compares against a checked-in hash file. If the hashes differ from the checked-in values, the test fails with a message instructing the developer to bump `FEATURE_SCHEMA_VERSION` and update the hash file.

## Stop Condition

Stop after the migration runs cleanly, all enums and constants are defined, all structs compile, the feature extraction function signature exists with a placeholder body, and the two tests above pass.

Do not start Phase 3. Do not implement feature extraction logic. Do not implement the writer. Do not write any additional tests beyond the two specified.

Report:

- Migration file path.
- List of enum names defined.
- Constants module path.
- Struct names defined.
- Confirmation that both tests pass.
