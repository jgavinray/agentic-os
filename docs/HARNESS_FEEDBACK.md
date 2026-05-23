# Harness Feedback

`agent_events` remains the audit log. Harness feedback is derived metadata
attached to events so repeated agent-harness failures can be measured without
turning bad benchmark traces into future memory.

The orchestrator records bounded signals under `metadata.harness_feedback`:

```json
{
  "schema_version": 1,
  "signals": ["path_typo_context_ledger"],
  "signal_count": 1,
  "quarantined": true,
  "quarantine_reason": "poisoned_path",
  "memory_policy": "exclude_from_context",
  "learning_status": "quarantined"
}
```

Signals are deterministic and closed-enum style. Raw prompts, paths, commands,
tool outputs, auth values, and user text are not stored as metric labels.

## Classification

The classifier consumes structured metadata first, such as
`metadata.harness.signals`, `metadata.harness.quarantine`, or existing
`metadata.harness_feedback`. It also applies narrow deterministic text checks
for known harness failure signatures observed in benchmark runs, including:

- misspelled `context_ledger` paths
- requested-path versus actual-path mismatches
- poisoned path attempts using known bad spellings
- false read/edit success claims involving poisoned paths
- final success claims with zero changed files in structured harness metadata
- missing required validations in structured harness metadata
- continued tool use after explicit stop or do-not-retry instructions
- self-correction after a stop/path violation
- failed reads or edits
- GateGuard/fact-forcing blocks and recovery failures
- forbidden Bash fallback
- invalid patch output or fake placeholder hashes
- misleading green tests
- no-progress runs
- explicit poisoned-memory indicators

There are no LLM calls, probabilistic classifiers, network calls, or filesystem
reads in this path.

## Quarantine

Quarantine does not delete or mutate the audit trail. It only marks events with
`memory_policy = "exclude_from_context"` and causes context-pack retrieval,
full-text search, semantic-search hydration, and context event counts to ignore
those rows. Direct audit reads and trajectory inspection can still access them.

Quarantine reasons are bounded:

- `benchmark_session`
- `poisoned_path`
- `repeated_invalid_path`
- `fake_success_claim`
- `tool_recovery_failed`
- `contract_violation`
- `patch_invalid`
- `manual_quarantine`
- `memory_poison_indicator`

Existing events can be reclassified with:

```bash
orchestrator-maint classify-harness-feedback \
  [--repo REPO] [--session SESSION] [--since TIMESTAMP] \
  [--dry-run] [--batch-size N]
```

Use `--dry-run` first to see how many rows would be updated and quarantined.
The orchestrator also runs this repair path automatically: one best-effort
startup pass is enabled by default, and a conservative periodic background
repair loop scans a recent lookback window. These jobs never create new
`agent_events`; they only update metadata on existing rows when the
deterministic classification changes.

Runtime controls:

- `HARNESS_FEEDBACK_STARTUP_BACKFILL_ENABLED` default `true`
- `HARNESS_FEEDBACK_STARTUP_BACKFILL_BATCH_SIZE` default `500`
- `HARNESS_FEEDBACK_BACKGROUND_REPAIR_ENABLED` default `true`
- `HARNESS_FEEDBACK_REPAIR_INTERVAL_SEC` default `300`
- `HARNESS_FEEDBACK_REPAIR_LOOKBACK_SEC` default `2 * interval`, minimum `60`
- `HARNESS_FEEDBACK_REPAIR_BATCH_SIZE` default `500`

## Runtime Guardrail

Clients that can see tool intent before execution can call
`POST /harness/guardrail` with the same `event_type`, `summary`, `evidence`,
and structured `metadata` shape used for event ingestion. The response is a
bounded deterministic decision:

- `allow`: no known hard violation
- `warn`: recordable signal, but not a hard stop
- `block`: do not execute or accept the claimed completion
- `terminate`: stop the current benchmark/session attempt

The orchestrator also attaches the same decision under
`metadata.harness_guardrail` when a persisted event triggers a non-allow
decision. This keeps enforcement decisions observable without adding raw paths
or free-form model text as metric labels.

## Learning

The learning surface is intentionally observational. The current implementation
records compact structured labels and metrics only. It does not train a model,
alter ranking, retry automatically, or modify live prompt assembly beyond the
quarantine filter that prevents poisoned traces from being reused as context.
