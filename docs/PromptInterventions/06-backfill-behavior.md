# Backfill Behavior

The system must support retroactive labeling of existing raw captures.

## Inputs

Backfill must accept these filters:

- since timestamp
- until timestamp
- requested model
- response model
- repository or namespace when available
- dry-run flag
- batch size

## Expected Behavior

Backfill must:

1. scan matching raw exchanges
2. extract prompt/message text deterministically
3. extract optional model, routing, attempt, trajectory, and baseline metadata
4. compute exact and normalized prompt hashes
5. run deterministic detectors
6. assign signal family, signal type, signal strength, burden type, and failure
   relation for every detected intervention
7. redact and bound evidence excerpts
8. deduplicate records for the same exchange, intervention type, evidence hash,
   and taxonomy version
9. write append-only intervention records when not in dry-run mode
10. record a backfill-run summary containing detector version, taxonomy version,
    fingerprint version, filter summary, rows scanned, labels written, status,
    start time, completion time, and notes

## Dry-Run Behavior

Dry-run mode must:

- scan and classify
- report counts by intervention type
- report counts by signal family
- report counts by burden type
- report counts by failure relation
- report counts by confidence bucket
- report rows scanned
- report labels that would be written
- avoid writing intervention records

## Historical Audit Behavior

Backfill must:

- backfill Heretic-backed captures by filtering response model to
  `qwen36-35b-heretic`
- tolerate SSE responses and JSON responses
- tolerate missing forwarded metadata
- avoid requiring cross-database joins to complete the backfill

## Deduplication

Repeated backfill runs over the same raw exchanges must not create duplicate
active records for the same exchange, intervention type, evidence hash, and
taxonomy version.

If a later detector version changes a label, it must append a new record rather
than mutating the old record.
