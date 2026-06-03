# vLLM Prefix Cache Diagnosis

This stack has three different cache layers that must stay separate:

1. **Context cache**: agentic-os stores compiled context packs so the orchestrator
   does not rebuild the same pack on every request.
2. **Client-visible provider cache accounting**: Anthropic-compatible responses
   can include `cache_creation_input_tokens` and `cache_read_input_tokens`.
   These fields are usage accounting, not the cache itself.
3. **vLLM prefix/KV cache**: vLLM reuses model prefill work when a later request
   has token-identical reusable prefix blocks.

The current Heretic backend is correctly serving `qwen36-35b-heretic`, but it is
not producing vLLM prefix-cache hits.

## Live Evidence

The Spark backend exposes the Heretic model through vLLM:

```text
model: qwen36-35b-heretic
image: ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2
vLLM: 0.1.dev1+gbfde49e28.d20260418
```

It starts with prefix caching enabled:

```text
--enable-prefix-caching
--enable-chunked-prefill
--speculative-config {"method":"dflash",...}
```

The same logs also show the important constraint:

```text
Mamba cache mode is set to 'align' for Qwen3_5MoeForConditionalGeneration
Prefix caching in Mamba cache 'align' mode is currently enabled.
Its support for Mamba layers is experimental.
```

A direct vLLM `/v1/completions` probe with two identical 25,001-token prompts
removed orchestrator, LiteLLM, Claude-compatible translation, and chat-template
variability. vLLM still reported:

```text
prefix_cache_queries_delta: 50002
prefix_cache_hits_delta: 0
prompt_tokens_local_compute_delta: 50002
prompt_tokens_cached_delta: 0
```

That means vLLM queried the prefix cache, missed, and locally recomputed the
prompt tokens.

## Root Cause

The installed image contains this Qwen3.5 guard:

```text
Qwen3.5 currently does not support 'all' prefix caching,
please use '--mamba-cache-mode=align' instead
```

For GDN/Mamba hybrid models, `align` mode only stores narrow boundary state and
is documented as experimental. vLLM upstream has active work for GDN `all` mode,
which stores recurrent state at every block boundary. That is the mode designed
to make repeated long prefixes reusable for Qwen3.5/Qwen3Next-style hybrid
models.

The current runtime cannot switch to `--mamba-cache-mode all`; it would fail at
startup. The current DFlash speculative decode setup also matters because the
GDN `all`-mode work does not yet cover speculative decoding.

## What agentic-os Can Track

agentic-os now records vLLM cache observations around model calls when
`VLLM_METRICS_URL` is configured:

- prefix-cache query tokens
- prefix-cache hit tokens
- total prompt tokens
- cached prompt tokens
- local-compute prompt tokens
- local-cache-hit prompt tokens
- external-KV-transfer prompt tokens
- request input/output tokens
- provider cache counters when the provider reports them

These are persisted in `vllm_cache_observations` and exposed from
`/cache/stats`.

Client-visible Anthropic cache fields must not be fabricated. For local vLLM,
agentic-os can safely surface cache reads only when vLLM reports
`local_cache_hit`, `external_kv_transfer`, or `prompt_tokens_cached`. Local
compute is tracked as local compute, not as provider cache creation.

## Remediation Options

The backend fix has to happen at the vLLM/runtime layer:

1. **Cache-first backend**: run the Heretic model on a vLLM build that supports
   Qwen3.5/Qwen3Next GDN `all`-mode prefix caching, then start it with
   `--mamba-cache-mode all` and no speculative decode until specdec support is
   proven.
2. **Speed-first backend**: keep the current DFlash speculative decode config.
   This preserves decode throughput but currently gives zero observed prefix
   reuse.
3. **Dual backend**: keep one Heretic endpoint optimized for speculative decode
   and one Heretic endpoint optimized for cache reuse. The orchestrator can route
   long stable-prefix agentic traffic to the cache-first endpoint and short or
   decode-heavy traffic to the speed-first endpoint.

The next backend validation must be a repeated-prefix probe against the candidate
runtime. Success criteria are concrete:

```text
prefix_cache_hits_delta > 0
prompt_tokens_cached_delta > 0
prompt_tokens_local_cache_hit_delta > 0
second identical request has lower first-token latency
```

Until those counters move, Claude Code usage will correctly show no cache reads.
