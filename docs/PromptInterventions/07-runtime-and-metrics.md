# Runtime And Metrics

After the backfill path exists, the runtime capture path should apply the same
deterministic detector when a raw exchange is recorded.

## Runtime Behavior

Runtime detection must be:

- best-effort only
- non-blocking for model responses
- isolated from user request completion
- excluded from context-pack injection
- consistent with the backfill detector version
- consistent with the backfill taxonomy version
- consistent with the backfill fingerprint version
- consistent with the backfill redaction behavior

Runtime write failures must be observable as bounded telemetry but must not
interrupt request completion.

## Required Counters

Runtime metrics must use bounded labels only.

Required counters:

- intervention records emitted by intervention type and labeler type
- intervention records emitted by signal family
- intervention records emitted by burden type
- intervention records emitted by failure relation
- backfill runs by result
- runtime intervention write attempts by result

## Required Reports

Required derived reports:

- interventions per day
- interventions by type
- interventions by signal family
- interventions by burden type
- interventions by failure relation
- interventions by requested model when available
- interventions by routed model when available
- interventions by baseline arm when available
- top exact prompt hashes by intervention count
- top normalized prompt hashes by intervention count
- interventions per trajectory when trajectory ID is available
- interventions per accepted result when outcome feedback is available
- accepted-without-intervention rate when outcome feedback is available

Metrics and reports must not expose raw prompt text, raw excerpts, raw file
paths, commands, tokens, auth values, or user text as labels.

## Relationship To Outcome Feedback

Prompt interventions do not say whether final work was accepted.

Outcome feedback answers:

> Was the final output accepted?

Prompt intervention telemetry answers:

> How much steering was required along the way?

The two should be correlated later to estimate real value:

- accepted output
- minus retries
- minus prompt repairs
- minus stop/redirects
- minus scope-narrowing interventions
- minus human attention

This prevents a model path from receiving full credit for work that only
became useful after substantial external steering.
