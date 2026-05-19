# ADR 0005: Trajectory Feature Extraction

## Status

Accepted

## Context

`agent_events` captures rich operational history: user and assistant messages, tool results, failed attempts, summaries, validation outcomes, retries, corrections, and trajectory lineage. Raw events are essential for auditability, but they are too verbose and irregular to serve directly as future operational policy input.

Repeated operational mistakes such as tool loops, wrong host endpoints, missing authorization headers, summarization failures, migration failures, and unacknowledged user corrections need compact reusable representation. Replaying raw history into every prompt would bloat context and still leave failure avoidance nondeterministic.

## Decision

Add deterministic trajectory feature extraction:

- Store structured detection tags in `agent_events.metadata.detection_tags`.
- Derive compact aggregates into `agent_feature_records`.
- Build bounded Operational Constraints from recent unsuppressed feature signals.
- Inject the constraints above Failure History in context packs.
- Provide `orchestrator-maint extract-features` for bootstrap tagging and backfill.

Detection, extraction, freshness evaluation, recovery suppression, and prioritization are all deterministic and explainable. No LLM calls or learned classifiers are used.

## Rationale

Raw events are not enough because meaning is often embedded in summaries or prose. Compact feature rows give the system stable operational memory without replaying full history.

Rule-based extraction comes before learned policy because the system first needs trustworthy labels and counters. The same rows can later become training inputs for learned routing, but the initial substrate must be deterministic and auditable.

Constraints are bounded to avoid context bloat. Imperative guardrails are capped by priority and token budget, and stale or recovered failures are suppressed.

Pattern detection requires structured tags rather than prose matching. Summary wording and prompt templates change over time; `metadata.detection_tags` gives detectors and extractors a stable contract.

Tags can be produced by multiple sources rather than the summarizer alone. Hooks, validation parsers, failed-attempt classifiers, summarizers, remediation parsers, and tool-loop detectors observe different evidence at different moments. Keeping source provenance makes disagreements and overlaps observable.

Bootstrap migration must maintain parity with live producers. Historical and future feature rows need to form one coherent corpus; otherwise downstream policy systems would learn from two subtly different data distributions.

## Consequences

`agent_feature_records` is derived state and can be rebuilt from `agent_events`. Operators can inspect `other_failure_count` to discover unclassified patterns. The design does not add graph storage, semantic trajectory search, autonomous retries, trajectory scoring, XGBoost, reinforcement learning, or prompt-body archival.
