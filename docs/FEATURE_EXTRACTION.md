# Trajectory Feature Extraction

## Purpose

Feature extraction turns raw operational history into compact deterministic state. `agent_events` remains the source of truth and audit log. `agent_feature_records` is derived state that can be rebuilt from `agent_events` for one trajectory, session, repo, time range, or the full dataset.

The subsystem follows three rules:

- Capture raw facts in `agent_events`.
- Extract compact feature rows in `agent_feature_records`.
- Inject only bounded corrective constraints into context packs.

No LLM calls, probabilistic classification, learned routing, retry policy, or semantic search are used by detection, extraction, or constraint building.

## Detection Tags

Structured detection tags live under `metadata.detection_tags` as a list. Feature extraction consumes these tags only; live extraction does not match free-form prose because prompt or summarizer wording changes would silently break rules.

Every tag includes:

```json
{
  "type": "tool_loop",
  "tool": "Read",
  "source": "hook_parser",
  "tag_schema_version": 1
}
```

Required fields are `type`, `source`, and `tag_schema_version`. The current schema version is `1`. The extractor skips unknown schema versions and increments `feature_tag_schema_version_unknown_total`. Backward-compatible optional fields are ignored by older extractors.

Supported tag types:

- `tool_loop`, with `tool`
- `user_interruption`
- `missing_auth`
- `wrong_endpoint`
- `summarization_failure`
- `migration_failure`
- `correction_acknowledged`

Supported live producer sources:

- `hook_parser`
- `tool_loop_detector`
- `validation_parser`
- `failed_attempt_classifier`
- `summarizer`
- `remediation_parser`

Historical bootstrap tags use `bootstrap_migration` as their source so provenance stays visible.

## Producer Conflicts

`metadata.detection_tags` is a list because multiple deterministic producers can observe the same event. Different tag types are retained and can contribute to different counters. Tags from the same producer with the same type on the same event are deduplicated and logged at debug level.

During extraction, the same event contributes once per condition. For example, `hook_parser` and `summarizer` can both retain a `tool_loop` tag for `Read`, but the feature row counts one `Read` loop condition for that event. Events with any recognized tag are not classified as `other_failure`.

## Feature Records

When `trajectory_id` is present, extraction groups by trajectory. Otherwise it groups by `session_id` plus `FEATURE_WINDOW_SEC`, default `3600`.

Feature rows include event counts, failure counters, known facts, booleans such as `loop_detected`, `failure_classes`, `recommended_constraints`, and `suppressed_constraints`.

`other_failure_count` increments for `failed_attempt` events with zero recognized detection tags. No constraint is generated for `other_failure`; it exists so operators can see unclassified failure patterns and add new deterministic producers.

## Constraints

Only detections within `CONSTRAINT_FRESHNESS_WINDOW_SEC`, default `1800`, can produce active constraints. Older detections still increment lifetime counters but are recorded as suppressed with `reason=freshness_window` when they would otherwise qualify.

Recovery suppression applies when a recovery indicator for the same failure class occurs after the most recent qualifying detection and the recovery itself is still inside the freshness window. A new qualifying detection after recovery resets suppression.

Recovery indicators:

- `tool_loop`: successful `tool_result` for the same tool
- `user_interruption`: later `correction_acknowledged` tag
- `missing_auth`: successful authenticated request event
- `wrong_endpoint`: successful request to the known correct endpoint
- `summarization_failure`: successful summarization event
- `migration_failure`: successful migration event

The builder emits at most `MAX_OPERATIONAL_CONSTRAINTS`, default `5`, in fixed priority order:

1. `use_known_auth`
2. `use_known_endpoint`
3. `use_known_migration_fix`
4. `avoid_tool_loop`
5. `handle_user_interruption`
6. `handle_summarization_failure`

Read and Bash loop signals combine into one `avoid_tool_loop` constraint before the cap is evaluated.

Suppression reasons are:

- `freshness_window`
- `recovery_detected`
- `priority_cap`
- `token_budget`

## Context Packs

Context packs include an `Operational Constraints` section immediately above `Failure History`. If Failure History is absent, Operational Constraints still occupies that slot before later general sections such as Open Questions. Empty sections are omitted.

Format:

```text
Operational Constraints:
- Use `Bearer ...` when calling protected orchestrator endpoints.
- Do not use `localhost` for host-side orchestrator testing. The correct endpoint for this environment is `...`.
```

The section contains only compact constraint text: no raw events, metadata, summaries, or counters. `OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET`, default `300`, drops lower-priority constraints first when needed.

## Failure Handling

Inline extraction is best-effort. If extraction, constraint building, or persistence fails:

- A WARN log is emitted.
- The user-facing request proceeds.
- The Operational Constraints section is omitted if constraints cannot be built.
- `feature_extraction_failures_total{stage}` increments for `extraction`, `constraint_build`, or `persistence`.

## Backfill

Run:

```bash
orchestrator-maint extract-features [--repo REPO] [--session SESSION] [--trajectory TRAJECTORY] [--since TIMESTAMP] [--dry-run] [--batch-size N] [--skip-bootstrap-tagging]
```

Backfill first applies deterministic bootstrap tagging to historical events unless `--skip-bootstrap-tagging` is passed. Bootstrap prose matching is allowed only here. It writes structured tags equivalent to live producer output for the same conditions, with `source=bootstrap_migration` as the only provenance difference.

Bootstrap tagging and feature persistence are idempotent. Re-running backfill does not create duplicate tags or feature rows.

## Metrics

New metrics:

- `trajectory_features_total`
- `feature_failure_classes_total{failure_class}`
- `operational_constraints_injected_total{constraint_type}`
- `operational_constraints_suppressed_total{constraint_type,reason}`
- `feature_extraction_duration_seconds`
- `feature_extraction_failures_total{stage}`
- `feature_tag_schema_version_unknown_total`

Labels are bounded enumerations. User text, IP addresses, filenames, endpoints, auth values, summaries, and other user-controlled values are not metric labels.

## Feature Flag

`FEATURE_EXTRACTION_ENABLED=false` disables inline extraction and Operational Constraints injection. Existing behavior is preserved. Manual backfill remains available.

## Future Policy

Feature rows are deliberately compact, deterministic training substrate for future learned routing or policy systems. This work does not implement learned policy, XGBoost, reinforcement learning, autonomous retries, or trajectory scoring.
