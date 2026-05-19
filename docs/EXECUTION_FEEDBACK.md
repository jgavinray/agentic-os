# Execution Feedback

agentic-os captures deterministic engineering outcomes as first-class memory events in the existing `agent_events` log. The subsystem is enabled by default with `EXECUTION_FEEDBACK_ENABLED=true`; set it to `false` to skip execution artifact capture and omit Failure History context without rolling back migrations.

Rows created before execution feedback may have null `correlation_id` and `parent_event_id`; they remain valid and queryable. Legacy `failure_signature` rows from the initial implementation are inert historical artifacts after `V3__inline_failure_signatures.sql`; retrieval ignores them.

## Event Taxonomy

All execution feedback events use the existing event insertion path and store a structured JSON envelope in `metadata`:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `event_type` | string | yes | One of the event types below. |
| `success` | boolean | yes | Deterministic pass/fail outcome. |
| `correlation_id` | UUID | yes | Action-chain identifier generated at user-message ingestion or validation submission. |
| `parent_event_id` | UUID or null | yes | Direct predecessor in the chain when known. |
| `repo` | string | yes | Repository namespace. |
| `task` | string | yes | Task label. |
| `payload` | object | yes | Event-specific fields. |

Event-specific payload schemas:

| Event Type | Required Payload Fields |
| --- | --- |
| `tool_result` | `tool_name: string`, `exit_code: integer`, `stdout_summary: string`, `stderr_summary: string`, `duration_ms: integer` |
| `compile_result` | `language: string`, `target: string`, `exit_code: integer`, `error_count: integer`, `warning_count: integer` |
| `test_result` | `framework: string`, `total: integer`, `passed: integer`, `failed: integer`, `skipped: integer`, `failure_summaries: string[]` |
| `lint_result` | `tool_name: string`, `error_count: integer`, `warning_count: integer`, `findings: string[]` |
| `validation_result` | `validator_name: string`, `pass: boolean`, `failure_reason: string` |
| `patch_result` | `files_touched: string[]`, `outcome: "applied" | "rejected" | "reverted"`, `validation_event_ids: UUID[]` |
| `remediation` | `failure_event_id: UUID`, `resolving_event_id: UUID`, `signature: string` |

Outcome payloads for `tool_result`, `compile_result`, `test_result`, `lint_result`, and `validation_result` also carry a stable inline signature shape:

| Field | Failed Outcome | Successful Outcome |
| --- | --- | --- |
| `signature` | canonical fingerprint string, for example `rust:type-mismatch` | `null` |
| `signature_category` | one of `borrow_checker`, `import_error`, `type_error`, `parse_error`, `unknown` | `null` |
| `fingerprint_version` | integer `FINGERPRINT_VERSION` | `null` |

## Failure Fingerprinting

Fingerprinting is deterministic and uses only regex/string rules. `FINGERPRINT_VERSION` is stored in code and emitted inline on failed outcome payloads. Successful outcome payloads explicitly set signature fields to `null` so consumers can rely on a stable shape.

Initial rules:

| Rule | Matches | Signature |
| --- | --- | --- |
| `rust_borrow_checker` | rustc `E0382`, `E0497`, `E0501`, `E0506`, `E0507` | `rust:borrow-checker` |
| `rust_type_mismatch` | rustc `E0308`, `E0278`, `E0302`, `E0305` | `rust:type-mismatch` |
| `python_import_error` | `ImportError`, `ModuleNotFoundError` | `python:import-error` |
| `typescript_type_error` | `TS####` diagnostics | `typescript:TS####` |
| `json_parse_error` | common JSON parser errors | `json:parse-error` |
| `generic_non_zero_exit` | non-zero exit-code text | `process:non-zero-exit` |

If no rule matches, the signature is `unknown`. To add a rule, add the matcher and register it in the rule registry in `execution_feedback.rs`.

## Patch Chains

Patches are represented by event chains, not separate patch objects. A chain starts with the user message and model response, then continues through `tool_result`, validator events, and a final `patch_result`. Failed outcome events carry their canonical fingerprint metadata inline; no standalone signature event is written.

`correlation_id` links all events in the action sequence. `parent_event_id` links each event to its direct predecessor when known. `db::get_event_chain_by_event_id` reconstructs a chain from any event by using both fields.

## Validation Capture

Path A, stream parsing: completed chat streams are scanned for `tool_result` blocks. Known validator tool names are `cargo`, `pytest`, `npm test`, `eslint`, `tsc`, `mypy`, `ruff`, `terraform`, and `kubectl`. Recognized validators produce `compile_result`, `test_result`, `lint_result`, or `validation_result`; unrecognized tools produce only `tool_result`.

Path B, explicit endpoint: `POST /v1/validations` accepts:

```json
{
  "repo": "agentic-os",
  "task": "fix tests",
  "session_id": "optional-session-id",
  "actor": "validator",
  "validator_name": "pytest",
  "validator_type": "test",
  "success": false,
  "exit_code": 1,
  "content": "1 failed, 2 passed",
  "payload": null,
  "correlation_id": "optional-uuid",
  "parent_event_id": "optional-uuid"
}
```

The response contains `captured`, `event_ids`, and `qdrant_indexed`.

## Failure History Context

Context packs gain a `Failure History` section between `Failed Attempts and Remediations` and `Open Questions`. The section appears only when the current task text contains a recognizable failure signature or raw error text that fingerprints to one, and exact-match lookup finds prior failed outcome events with a matching inline `payload.signature` in the same repo.

For each match, agentic-os reconstructs or searches the event chain for linked `remediation` events and surfaces the failure plus remediation. Lookup is exact-match only; semantic similarity is not used for signatures. The section respects `FAILURE_HISTORY_TOKEN_BUDGET` (default `1000` tokens) and is omitted entirely when empty or when `EXECUTION_FEEDBACK_ENABLED=false`.

## Backfill

Run the maintenance command after upgrading databases that may contain failed outcomes without inline signatures:

```bash
orchestrator-maint backfill-signatures --dry-run
orchestrator-maint backfill-signatures --batch-size 500
```

The backfill reads failed outcome events missing inline fields, derives deterministic fingerprints from stored outcome payload text or the legacy signature row excerpt when available, and writes the metadata back into the existing `metadata.payload` JSON. It is idempotent, resumable, batched, and does not delete legacy `failure_signature` rows.

On startup, the orchestrator emits an informational WARN log if legacy signature rows exist and failed outcomes still need inline backfill. Startup continues normally; retrieval still ignores legacy signature rows.

## Metrics

Execution feedback metrics are documented in [METRICS.md](METRICS.md). Labels are bounded to canonical event types, validator names, outcomes, retry task types, trigger categories, and canonical signatures.

## Operational Notes

Execution feedback writes are best-effort on chat paths. Failures emit error-level structured logs with event type, correlation ID, and event ID when available, then the user-facing request continues. Graceful shutdown is handled by the existing Axum shutdown path and background persistence tasks are allowed to finish during normal request draining.
