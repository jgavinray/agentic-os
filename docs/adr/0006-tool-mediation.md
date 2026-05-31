# ADR 0006: Deterministic Tool Mediation

## Status

Accepted

## Context

Trajectory lineage makes engineering attempts observable, but tool use still comes mostly from client runtimes. In proxy mode, agentic-os can see model request payloads and persisted tool outcomes, but it does not own every tool loop.

We need a bridge that improves tool discipline now and can become part of a future agent orchestrator. A representative case is file reading: if a client offers both `Read` and `Bash`, the model should use `Read` for file reads rather than `bash cat`, `sed`, `head`, or `tail`.

Relying only on prompt instructions is weak. Enforcement requires either shaping the model's visible tool menu before the model call or authorizing tool calls before the client executes them.

## Decision

Add a deterministic tool mediation subsystem.

The subsystem:

- maps client-provided tool names into bounded capabilities,
- detects simple tool intent from the user request,
- subtractively shapes OpenAI and Anthropic `tools` arrays before forwarding to LiteLLM,
- exposes `POST /tools/authorize` for pre-execution client hooks,
- denies shell fallbacks when a canonical tool for the same capability is available,
- records authorization decisions as event-log entries when session data is provided,
- emits bounded metrics.

The first enforced rule is canonical-tool preference for read/search/list operations. agentic-os does not inject tools the client did not advertise.

## Consequences

This improves behavior for compatible clients without requiring agentic-os to become the full tool executor immediately.

Menu shaping prevents the model from seeing unnecessary broad tools in simple phases. Authorization gives hook-capable clients a hard denial path and a replacement suggestion.

The design remains compatible with future broker and orchestrator modes because the primitives are capability-oriented rather than tied to one client tool schema.

## Non-Goals

- learned routing
- scoring
- reinforcement learning
- autonomous retries
- tool execution ownership
- graph database
- prompt-body archival
- injecting tools that the client cannot execute

## Feature Flag

`TOOL_MEDIATION_ENABLED=true` by default.

When disabled, proxy payloads are not shaped and `/tools/authorize` allows calls with `reason=disabled`.
