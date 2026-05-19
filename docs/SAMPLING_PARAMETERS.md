# Sampling Parameters

agentic-os captures the sampling controls supplied on OpenAI-compatible `/v1/chat/completions` requests so future routing and policy layers can correlate outcomes with generation conditions. Capture is deterministic, enabled by default, and stored in the existing `agent_events.metadata` JSON for the persisted `user_message` request event.

## Captured Fields

Each captured request event includes:

```json
{
  "event_schema_version": 1,
  "endpoint": "/v1/chat/completions",
  "requested_model": "client-supplied-or-default-model",
  "routed_model": "orchestrator-routed-model",
  "sampling_params": {
    "temperature": null,
    "top_p": null,
    "max_tokens": null,
    "seed": null
  },
  "forwarded_sampling_params": {
    "temperature": null,
    "top_p": null,
    "max_tokens": null,
    "seed": null
  }
}
```

`sampling_params` records what the client requested. `forwarded_sampling_params` records the sampling parameter payload after the orchestrator sampling override hook has had a chance to run.

All four fields are always present in both objects when capture is enabled. Missing client fields are stored as `null`.

`event_schema_version` is the request event metadata schema version. It is incremented when the payload structure changes in an incompatible way.

`forwarded_sampling_params` is intentionally not named effective parameters. It reflects what the orchestrator forwarded after its own sampling hook. It does not prove what LiteLLM, the provider, or the upstream model actually honored.

## Feature Flags

| Flag | Default | Behavior |
| --- | --- | --- |
| `SAMPLING_CAPTURE_ENABLED` | `true` | Writes `sampling_params`, `forwarded_sampling_params`, and `event_schema_version` into request event metadata. |
| `SAMPLING_OVERRIDE_ENABLED` | `false` | Invokes the sampling override hook before forwarding to LiteLLM. |

Flag combinations:

| Capture | Override | Result |
| --- | --- | --- |
| enabled | disabled | Capture requested params and store `forwarded_sampling_params` as a field-by-field copy. The override hook is not invoked. |
| enabled | enabled | Capture requested params, invoke the hook exactly once, capture forwarded params after the hook, and forward the modified request. |
| disabled | disabled | No sampling fields are written and the override hook is not invoked. |
| disabled | enabled | Startup fails with a clear configuration error because override without audit capture is unsafe. |

## Override Hook

The current policy is `NoOpSamplingPolicy`. It returns the request unchanged and exists only to define the future control surface.

When override is enabled, the no-op policy increments:

```text
sampling_param_overrides_total{parameter="none",reason="noop"}
```

That metric is a wire check showing that the hook executed. If a future policy modifies one or more parameters, agentic-os increments the same counter with `reason="overridden_by_orchestrator"` and `parameter` set to the bounded parameter name.

## Metrics

Raw sampling values are not Prometheus labels. They live only in event metadata. The metric labels are bounded to the known parameter names plus `none`, and the reason values `noop` and `overridden_by_orchestrator`.

## Scope

This work collects data and establishes the request-path control surface. It does not implement adaptive policy, learned routing, XGBoost integration, exploration mode, provider-specific parameter translation, or claims about upstream model-confirmed effective parameters.

The purpose is to make future policy learning possible: outcome feedback says what happened, failure signatures explain how it failed, retry history shows wasted work, and sampling capture records the generation conditions under which those outcomes occurred.
