# Prompt Intervention Telemetry

This is the index for the prompt intervention telemetry handoff packet.

Prompt intervention telemetry measures how much external steering a model path
requires before it produces useful work. It is telemetry, not adversarial
review.

The packet is split into small files so a smaller implementation model can work
one bounded deliverable at a time.

Read in this order:

1. [Overview](PromptInterventions/01-overview.md)
2. [Record Contract](PromptInterventions/02-record-contract.md)
3. [Signal Taxonomy](PromptInterventions/03-signal-taxonomy.md)
4. [Extraction And Fingerprints](PromptInterventions/04-extraction-and-fingerprints.md)
5. [Detector Rules](PromptInterventions/05-detector-rules.md)
6. [Backfill Behavior](PromptInterventions/06-backfill-behavior.md)
7. [Runtime And Metrics](PromptInterventions/07-runtime-and-metrics.md)
8. [Examples](PromptInterventions/08-examples.md)
9. [Grading Criteria](PromptInterventions/09-grading-criteria.md)

Implementation rule:

- Do not store full raw prompts in derived intervention records.
- Do not mutate raw captures.
- Do not infer ChatGPT or human authorship unless provenance explicitly exists.
- Do generate bounded telemetry signals for steering, failure correction, risk
  control, validation pressure, and context pressure.
