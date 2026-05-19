# agentic-os

A local-first agent memory orchestrator for LLM clients. It provides OpenAI-compatible and Anthropic-compatible proxy endpoints, persistent engineering memory, semantic and full-text recall, automated context packing, summarization, and Anthropic tool-use passthrough through LiteLLM.

```
Claude Code / opencode / curl
             |
             v
      Rust orchestrator
       |       |       |
       v       v       v
   Postgres  Qdrant  LiteLLM
```

## Quick Start

```bash
cp .env.example .env
# Edit .env: set LITELLM_SALT_KEY and API_KEYS.

./setup-models.sh
docker compose up -d
curl localhost:8088/health/ready
```

## Architecture

The orchestrator is a single-node control plane. It also captures deterministic engineering outcomes such as tool results, test runs, lint failures, patch outcomes, remediations, and inline failure signatures as first-class memory events; see [docs/EXECUTION_FEEDBACK.md](docs/EXECUTION_FEEDBACK.md). It captures chat sampling parameters for future outcome-aware routing; see [docs/SAMPLING_PARAMETERS.md](docs/SAMPLING_PARAMETERS.md). Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the endpoint surface, memory model, retrieval pipeline, summarizer loop, cache behavior, and startup order.

| Component | Role | Port |
| --- | --- | --- |
| Rust Orchestrator | Authenticated API frontend, context compiler, memory coordinator, LiteLLM proxy | 8088 |
| PostgreSQL | Durable sessions, events, summaries, errors, token usage, migration history | 5432 |
| Qdrant | Vector memory for semantic recall | 6333-6334 |
| LiteLLM | Model routing and OpenAI/Anthropic-compatible upstream API | 4000 |
| Local ONNX Embedder | In-process embeddings for Qdrant indexing and search | n/a |

## API Endpoints

All endpoints except health checks require `Authorization: Bearer <key>`.

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health`, `/health/live` | GET | Process liveness. |
| `/health/ready` | GET | Readiness for Postgres, Qdrant, and LiteLLM. |
| `/v1/models` | GET | Model list proxied from LiteLLM. |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions, streaming or non-streaming, with injected memory context. |
| `/v1/messages` | POST | Anthropic-compatible messages passthrough with memory context. |
| `/v1/validations` | POST | Submit a structured validation report for deterministic execution feedback. |
| `/sessions/start` | POST | Create an explicit memory session. |
| `/events/append` | POST | Store a memory event and best-effort vector index it. |
| `/context/pack` | POST | Return a layered context pack for a repo/task. |
| `/cache/stats` | GET | Return context cache size and TTL. |
| `/metrics` | GET | Prometheus exposition format. |
| `/metrics/json` | GET | Legacy JSON metrics snapshot. |
| `/summaries/checkpoint` | POST | Store a checkpoint event with next actions/open questions. |
| `/search` | POST | Semantic Qdrant search over memory. |

## Memory Namespaces

`API_KEYS` is semicolon-delimited:

```bash
API_KEYS=sk-work,work;sk-project,project-alpha
```

`Bearer sk-work` routes to the `work` namespace. `x-agent-repo` and `x-agent-task` can override the repo/task per request.

## Operations

Operational procedures live in [docs/OPERATIONS.md](docs/OPERATIONS.md). Highlights:

- Exactly one orchestrator process may own a Postgres database.
- Schema migrations are embedded in `orchestrator/migrations/`.
- Signature backfill runs with `orchestrator-maint backfill-signatures`.
- Backups run with `scripts/backup.sh`; restores run with `scripts/restore.sh`.
- Metrics are documented in [docs/METRICS.md](docs/METRICS.md), with a dashboard at [docs/grafana/agentic-os.json](docs/grafana/agentic-os.json).

### Adding A Migration

Create a new file named `orchestrator/migrations/V<N>__short_name.sql`, where `N` is the next integer version. Write forward-only SQL, never edit an already-applied migration, then run:

```bash
cd orchestrator
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

## Security Model

agentic-os assumes a single-user local node, typically bound to localhost or reachable through a private network such as Tailscale. By default, CORS allows any origin to preserve local-tool compatibility. Set `ALLOWED_ORIGINS` to a comma-separated origin list before exposing the API to a LAN, tunnel, or browser-accessible shared network.

Rate limiting applies per API key to `/v1/chat/completions` and `/v1/messages`. Health, metrics, and memory endpoints are not rate limited.

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | required | Postgres connection string. |
| `QDRANT_URL` | required | Qdrant base URL. |
| `LITELLM_URL` | required | LiteLLM base URL including `/v1`. |
| `LITELLM_KEY` | required | LiteLLM API key. |
| `LITELLM_SALT_KEY` | required | LiteLLM salt key. |
| `API_KEYS` | `agent-os,agentic-os` | Semicolon-delimited `token,namespace` entries. |
| `DEFAULT_MODEL` | `qwen36-35b-heretic` | Canonical model sent to LiteLLM. |
| `DEFAULT_TASK` | `engineering` | Task label when no header is present. |
| `CONTEXT_CACHE_TTL_MS` | `300000` | Context cache TTL. |
| `CONTEXT_DECAY_RATE` | `0.006` | Hybrid retrieval age decay. |
| `EXECUTION_FEEDBACK_ENABLED` | `true` | Enables execution artifact capture and Failure History context. |
| `FAILURE_HISTORY_TOKEN_BUDGET` | `1000` | Token budget for Failure History context. |
| `SAMPLING_CAPTURE_ENABLED` | `true` | Captures requested and forwarded chat sampling parameters in event metadata. |
| `SAMPLING_OVERRIDE_ENABLED` | `false` | Enables the sampling override hook. Requires sampling capture to stay enabled. |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-key inference refill rate. |
| `RATE_LIMIT_BURST` | `30` | Per-key inference burst. |
| `ALLOWED_ORIGINS` | `*` | CORS origin policy. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Enables OTLP tracing when built with `tracing-otlp`. |

## License

GPL v2
