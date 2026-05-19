# ADR 0002: Inline Failure Signatures on Outcome Events

## Status

Accepted

## Context

The first execution feedback implementation stored canonical failure fingerprints as standalone `failure_signature` events linked near failed compile, test, lint, validation, or tool outcomes. That doubled writes for fingerprintable failures and made retrieval understand a two-event pattern even though the signature describes the failed outcome itself.

The system has no external consumers yet, so the cleaner model can replace the original design directly.

## Decision

Failed outcome events carry fingerprint metadata inline in `metadata.payload`:

- `signature`
- `signature_category`
- `fingerprint_version`

Successful outcome events carry the same fields with explicit `null` values. Standalone `failure_signature` events are no longer produced, and Failure History retrieval reads failed outcome events by exact `payload.signature` match.

No new tables, columns, storage systems, semantic matching, or compatibility dual-read path are introduced.

## Consequences

Storage is simpler because a failure and its canonical fingerprint live in one event row. Retrieval is simpler because Failure History reads one event type family instead of joining or walking through standalone signature rows before finding the actual outcome. Event chains remain coherent because remediations can point at the failed outcome event itself.

Legacy `failure_signature` rows remain as inert historical artifacts. The `orchestrator-maint backfill-signatures` command rewrites older failed outcomes to include inline signature fields without deleting legacy rows. A later cleanup migration may remove those rows after operational validation.
