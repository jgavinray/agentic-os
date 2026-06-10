# Project: agentic-os

A local-first agent operating environment — a Rust orchestrator that sits between coding agents (Claude Code, opencode) and local LLM inference backends. Provides persistent memory, semantic recall, and automated context packing.

## Architecture

```
Clients → Orchestrator → [PGSQL state, Qdrant memory, LiteLLM router] → local GPU
```

The orchestrator is the "brain stem." Models are interchangeable compute units.

## Code Layout

`orchestrator/src/` — All Rust source in a flat module tree:

| Module | Responsibility |
|--------|---------------|
| `main.rs` | Entry point, routes, graceful shutdown |
| `handlers.rs` | HTTP endpoints — auth, streaming, persistence (largest file) |
| `anthropic.rs` | Anthropic↔OpenAI protocol translation, SSE streaming |
| `db.rs` | Postgres schema, queries, context pack builder |
| `qdrant.rs` | Qdrant vector store/search via TEI embeddings |
| `hybrid.rs` | RRF merge for semantic + FTS search |
| `summarizer.rs` | Background session summarizer |
| `state.rs` | AppState, request types, task categorization |
| `logging.rs` | Structured JSON logging init |

## Running

```bash
./setup-models.sh        # Download embedding model (one-time)
docker compose up -d     # Start full stack (postgres, qdrant, embedding, litellm, orchestrator)
cargo test               # Run tests
```

## Headless Claude Through Orchestrator

Use the clean namespace key when driving Claude Code through the local
orchestrator. Do not use `sk-local-orchestrator` for implementation runs; that
namespace is polluted by prior failed trajectories.

```bash
ANTHROPIC_BASE_URL=http://localhost:8088 \
ANTHROPIC_AUTH_TOKEN=sk-agent-clean-002 \
ANTHROPIC_API_KEY=sk-agent-clean-002 \
claude -p --model opus --dangerously-skip-permissions \
  --strict-mcp-config --mcp-config '{"mcpServers":{}}' \
  -- 'your prompt here'
```

Model selection should be done with `--model opus`. Do not add
`ANTHROPIC_DEFAULT_OPUS_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`, or
`ANTHROPIC_DEFAULT_HAIKU_MODEL` for this path. Do not add `--allowedTools`;
tool availability should be mediated by the orchestrator policy path. Keep
`--strict-mcp-config --mcp-config '{"mcpServers":{}}'` unless MCP tools are
explicitly part of the test; otherwise plugin MCP servers such as ECC GitHub
search become part of the effective tool surface. The `--` separator is
required because `--mcp-config` accepts multiple values and will otherwise
consume the prompt.

## Tool Mediation Validation Requirement

When validating intent-to-tool mediation, do not prove correctness by forcing
`tool_choice` to a specific tool. The requirement is that the orchestrator
classifies the request intent and then exposes the correct narrowed tool menu
to the LLM for that intent through the normal `/v1/messages` or
`/v1/chat/completions` path. Runtime evidence must come from live orchestrator
requests with a broad client-provided tool menu and no forced tool choice, then
querying the captured `tool_mediation_decisions` row to verify intent,
allowed tools, hidden tools, and missing implementation capabilities.

If validating that the LLM can use a tool, use natural prompts that make that
tool appropriate for the request intent, without overriding mediation with
forced `tool_choice`. Forced tool probes only prove provider/tool-call
plumbing and are not evidence that intent mediation is correct.

## Key Patterns

- **Auth**: Constant-time Bearer token comparison via `subtle` crate. `API_KEYS=token,ns;token2,ns2` format.
- **Two HTTP clients**: `http` (5-min timeout for normal) and `http_stream` (no timeout for SSE streaming)
- **Memory namespaces**: Key-derived, fully isolated per namespace
- **Context injection**: Appended to existing system message (never duplicates), floor `max_tokens` at 65536
- **Persistence**: Events stored to Postgres + Qdrant (best-effort, embedding failure doesn't break the event)
- **Summarizer**: Background loop that summarizes sessions after 20+ messages via LLM

## Conventions

- **Naming**: snake_case
- **Errors**: `anyhow::Error` for opaque, `Result` for propagation
- **Tests**: Inline `#[cfg(test)]` modules with unit tests
- **Commits**: conventional commits only (`feat:`, `fix:`, `chore:`, etc.)
- **Docker**: `cgr.dev/chainguard/rust:latest-dev`, multi-stage, glibc
