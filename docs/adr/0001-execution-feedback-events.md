# ADR 0001: Capture Engineering Outcomes as Events

## Status

Accepted, superseded in part by [ADR 0002](0002-inline-failure-signatures.md)

## Context

agentic-os already stores durable engineering memory in `agent_events`, indexes those events through the existing retrieval pipeline, and exposes metrics through the existing Prometheus facade. Compile, test, lint, patch, and remediation outcomes need to become durable memory without creating a second source of truth.

## Decision

Engineering outcomes are captured as structured event types in the existing event log. Event chains use nullable `correlation_id` and `parent_event_id` columns added to `agent_events`. The original decision stored failure fingerprints as `failure_signature` events; ADR 0002 replaces that piece with inline metadata on failed outcome events.

No separate execution-feedback table, patch object store, retrieval backend, or metrics path is introduced.

## Consequences

Existing migrations and storage operations continue to own persistence. Existing semantic and FTS retrieval can index execution artifacts, while Failure History uses a small exact-match query for canonical signatures. Old events remain valid with null chain columns.
