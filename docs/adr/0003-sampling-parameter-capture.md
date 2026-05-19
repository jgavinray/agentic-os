# ADR 0003: Capture Sampling Parameters on Chat Requests

## Status

Accepted

## Context

agentic-os records execution outcomes, failure signatures, retry behavior, token usage, and structured memory. It did not record the sampling parameters used on model requests, leaving future learned routing or policy systems blind to a major controllable input.

The orchestrator can attest to client-requested parameters and the parameters it forwards after its own hook runs. It cannot attest to what LiteLLM or an upstream provider actually honors.

## Decision

Capture `temperature`, `top_p`, `max_tokens`, and `seed` in existing request event metadata for `/v1/chat/completions`. Store two stable objects:

- `sampling_params`: the client-requested fields.
- `forwarded_sampling_params`: the fields after the orchestrator sampling override hook.

Both objects always include all four keys when capture is enabled, using `null` for missing fields. Request event metadata also includes `event_schema_version`.

Add `SAMPLING_CAPTURE_ENABLED=true` and `SAMPLING_OVERRIDE_ENABLED=false`. If override is enabled while capture is disabled, startup fails because unaudited sampling mutation is unsafe.

Add a `SamplingPolicy` hook with `NoOpSamplingPolicy` as the default implementation. The no-op hook returns requests unchanged and increments `sampling_param_overrides_total{parameter="none",reason="noop"}` only when override is enabled. Future policy implementations can replace the no-op without refactoring the request path.

No new tables or columns are introduced. No LiteLLM configuration is changed.

## Consequences

Outcome history now contains the generation conditions needed for future policy learning. Default behavior remains inert: capture is on, override is off, and the hook is bypassed.

Metrics avoid raw sampling values as labels. Raw values remain in event JSON, while Prometheus labels stay bounded.

Adaptive parameter selection, XGBoost integration, exploration mode, provider-specific effective-parameter modeling, and runtime tuning UI remain out of scope.
