# Phase 7: Export Command

Read `ARCHITECTURE.md` first. Every rule in it applies here.

Phases 1+2 through 6 must be complete. The ledger is recording envelopes, candidates, and outcomes from live traffic. Do not modify any of that.

## Scope

Implement only the following in this phase:

1. A CLI command that exports deterministic, privacy-safe training rows.
2. The manifest produced alongside the export.
3. Tests listed below.

## Out of Scope For This Phase

- Any modification to schema, joiner, writer, or assembler
- New metrics
- New enum variants
- Cross-repo export as default behavior
- Any predictive scoring or filtering

## CLI Command

Add a subcommand:

```
orchestrator-maint export-context-ledger [FLAGS]
```

Flags:

- `--repo REPO` — restrict to one repo. If omitted, defaults to the current repo. Cross-repo requires explicit `--cross-repo`.
- `--session SESSION` — restrict to one session.
- `--trajectory TRAJECTORY` — restrict to one trajectory.
- `--since TIMESTAMP` — RFC3339 timestamp, lower bound on `created_at_micros`.
- `--until TIMESTAMP` — RFC3339 timestamp, upper bound on `created_at_micros`.
- `--feature-schema-version N` — restrict to rows produced under a specific schema version. If omitted, exports all schema versions present and warns if multiple are present.
- `--include-open-outcomes` — include decisions where `outcome_window_closed = false`. Default behavior excludes them.
- `--cross-repo` — allow rows from multiple repos. Without this flag, multi-repo data is filtered to current repo only.
- `--dry-run` — print row count and manifest, write no rows.
- `--output PATH` — output file path. Required unless `--dry-run`.

## Default Behavior

- Single repo only unless `--cross-repo` is set.
- Exclude rows where `outcome_window_closed = false` unless `--include-open-outcomes` is set.
- No source identifiers in output: no `source_record_id`, no `candidate_id`, no `decision_id`.
- No partition identifiers in output: no `repo`, no `session_id`, no `trajectory_id`.
- No raw timestamps in output rows. The only timestamp anywhere in the output is `export_timestamp` in the manifest.
- Output contains only structured features and labels: the bucket enums, booleans, integer counts (where bucketing does not apply), and outcome enums.

## Output Format

Use a stable, deterministic format. Suggested: newline-delimited JSON (JSONL), one row per candidate-outcome join.

Per-row content (one row per `(envelope, candidate, outcome)` join):

- All fields from `CandidateFeatureVector` (Phase 3 output)
- `injected` (boolean)
- `drop_reason` (enum, nullable, never `drop_reason_other_detail`)
- `representation_selected` (enum)
- `trajectory_outcome` (enum)
- `user_correction_after_decision` (nullable boolean)
- `retry_after_decision` (nullable boolean)
- `validation_result` (enum)
- `tool_loop_detected_after_decision` (nullable boolean)
- `request_latency_bucket` (enum)
- `input_token_bucket` (enum)
- `output_token_bucket` (enum)
- `feature_schema_version` (integer)

Excluded from rows (privacy):

- All ids
- All raw timestamps
- `drop_reason_other_detail`
- `context_policy_version` (lives in manifest, not per-row)

## Determinism Rules For Output

- Row order is determined by a documented sort key. Suggested: sort by `(envelope.created_at_micros, candidate.deterministic_rank)`. Document this in code.
- JSON serialization uses sorted keys. No map iteration order leaks into output.
- The same query against unchanged data produces byte-identical bytes (excluding the manifest's `export_timestamp`).

## Manifest

Output alongside the rows (or as a header in the same file, document the choice). Contents:

- `feature_schema_version` — integer or list if multiple versions present
- `context_policy_version_range` — list of distinct policy versions observed in the export
- `row_count` — integer
- `filters` — record of every flag passed to the CLI
- `export_timestamp` — RFC3339 timestamp, the only non-deterministic field in the entire output
- `included_open_outcomes` — boolean
- `cross_repo` — boolean

## Tests Required In This Phase

1. **Export determinism test.** Run the export twice against unchanged ledger data. Diff the row content (excluding the manifest `export_timestamp`). Assert byte-identical.

2. **No raw text test.** Run the export. Inspect every row. Assert no field contains raw user text, raw tool output, or any value from the privacy-restricted list in the architecture doc. (Walk the schema; assert no unbounded strings appear except in the documented bounded allowlist — and the allowlist for export is empty, all bounded strings are excluded too.)

3. **No identifiers test.** Inspect every row. Assert no `decision_id`, `candidate_id`, `source_record_id`, `repo`, `session_id`, or `trajectory_id` appears.

4. **No raw timestamps test.** Inspect every row. Assert no microsecond or RFC3339 timestamp appears in row content. The only timestamp is `export_timestamp` in the manifest.

5. **Dry-run test.** Run with `--dry-run`. Assert no output file is written. Assert stdout contains row count and manifest.

6. **Open-outcomes exclusion test.** Create a decision with `outcome_window_closed = false`. Run export with default flags. Assert that decision is not in output. Run with `--include-open-outcomes`. Assert it is in output.

7. **Cross-repo default test.** Seed data from two repos. Run export with no `--cross-repo` flag. Assert only the current repo's rows appear. Run with `--cross-repo`. Assert both repos' rows appear.

8. **Schema version filter test.** Seed data under two schema versions. Run export with `--feature-schema-version 1`. Assert only version 1 rows appear. Run with no version flag. Assert a warning is emitted about mixed versions and all rows are exported.

9. **Sort order test.** Run the export. Parse the rows. Assert the sort key is monotonic per the documented rule.

10. **Privacy schema test.** A test that walks the exported row schema (the fields actually serialized) and fails if any field is in the architecture's privacy-restricted list. This is a static check; it catches a developer adding `decision_id` to the export.

## Stop Condition

Stop after the command runs cleanly, all ten tests pass, and exported files match the determinism guarantee.

Do not start Phase 8. Do not add new functionality.

Report:

- CLI command path.
- Output format chosen (JSONL or other).
- Documented sort key.
- Manifest format example.
- Confirmation that all ten tests pass.
