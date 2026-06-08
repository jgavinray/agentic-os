# Grading Criteria

An implementation satisfies this spec only if all required behavior below is
true.

## Core Correctness

- raw captures remain immutable
- intervention records are append-only
- every intervention record has exact raw-exchange provenance
- exact and normalized prompt hashes are recorded for every intervention
- detector, taxonomy, and fingerprint versions are recorded for every
  intervention
- signal family, signal type, signal strength, burden type, and failure
  relation are recorded for every intervention
- deterministic detector output is stable across repeated runs
- missing optional metadata does not prevent intervention labeling

## Backfill

- backfill dry-run produces counts without writing records
- backfill dry-run reports counts by intervention type, signal family, burden
  type, failure relation, confidence bucket, rows scanned, and labels that would
  be written
- backfill write mode deduplicates repeated runs
- backfill tolerates SSE and JSON responses
- backfill tolerates missing forwarded metadata

## Runtime

- runtime detection is best-effort
- runtime detection does not block model responses
- runtime detection does not fail user requests
- runtime write failures are observable through bounded telemetry

## Evidence Safety

- evidence excerpts are bounded
- evidence excerpts are redacted
- full raw prompts are not stored in intervention records
- raw prompt text, file paths, commands, auth values, and user text are not used
  as metric labels
- workspace-local raw prompts, responses, and label datasets are not committed
  to the open-source repo

## Metrics

- headline metrics exclude superseded records
- headline metrics count only confidence at or above 0.8
- metrics use bounded labels only
- signal metrics are available by family, burden type, and failure relation

## MVP Limits

- model-assisted labeling is not required for MVP
- `local_model` and `frontier_model` labeler types are reserved for later
  offline experiments

## Deliverables

The implementation handoff should produce:

- capture-side persistent storage for intervention records
- capture-side persistent storage for backfill-run summaries
- deterministic prompt/message extraction
- deterministic prompt fingerprinting
- deterministic phrase/rule detectors
- deterministic signal classification for each detected intervention
- redaction and bounded evidence-excerpt generation
- backfill dry-run mode
- backfill write mode
- runtime best-effort detection path
- bounded telemetry counters
- tests for extraction, fingerprinting, detection, signal classification,
  redaction, deduplication, superseding, dry-run behavior, write behavior, and
  runtime failure isolation

The implementation is complete when the examples produce the expected labels
and every grading criterion passes.
