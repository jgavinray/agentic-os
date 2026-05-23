# Phase 3: Backfill, Boot Integration, And CLI

## Goal

Persist request classifications for historical and new events, and make the
orchestrator self-sustaining at boot.

## Scope

Implement:

- Idempotent backfill over `agent_events`.
- Boot-time classification backfill before serving traffic.
- Operator CLI for dry-run and targeted repair.
- Persistence metrics.

The same library function must power boot, CLI, and any live write path.

## Event Selection

Classify request-like events first:

- `user_message`
- `context_pack` only if it represents request context metadata
- trajectory request events with `event_role = 'request'`

Do not classify summary-only maintenance rows unless a later phase explicitly
adds that behavior.

Skip events that already have a row for the same:

```text
event_id
classification_schema_version
routing_policy_version
```

## Boot Behavior

Startup order:

1. Run migrations.
2. Run harness feedback repair.
3. Run request classification backfill.
4. Run feature extraction backfill.
5. Bind HTTP listener.

If request classification backfill fails, the orchestrator refuses to serve
traffic.

The boot job must be idempotent and bounded by batch size.

## CLI

Add:

```text
orchestrator-maint classify-requests
  [--repo REPO]
  [--session SESSION_ID]
  [--since TIMESTAMP]
  [--dry-run]
  [--batch-size N]
```

CLI output includes:

```text
events_scanned
inserted
updated
skipped
dry_run
batch_size
```

## Metrics

Add:

```text
request_classification_backfill_runs_total{result}
request_classifications_written_total{result}
request_classification_unknown_labels_total{field}
```

Metric labels must be bounded.

## Required Tests

1. Backfill inserts rows for unclassified request events.
2. Backfill is idempotent over unchanged data.
3. Dry-run writes no rows.
4. Repo/session/since filters work.
5. Boot backfill failure prevents serving.
6. CLI and boot use the same classifier entry point.
7. Unknown labels increment bounded unknown-label metrics.

## Stop Condition

Stop when the CLI dry-run works, boot backfill is wired before listener bind,
and persistence tests pass. Do not enforce route recommendations.
