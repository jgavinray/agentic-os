# agentic-os

A local-first agent memory orchestrator for LLM clients. It provides OpenAI-compatible and Anthropic-compatible proxy endpoints, persistent engineering memory, semantic and full-text recall, automated context packing, summarization, and Anthropic tool-use passthrough through LiteLLM.

## What This Solves

LLM coding clients generate valuable engineering evidence while they work: user intent, retrieved context, model responses, tool calls, validation failures, patches, retry attempts, token usage, and final outcomes. Without durable lineage, that evidence is scattered across transient logs and chat history. You can ask what happened, but not reliably which user intent it belonged to, which context informed a response, which validation caused a retry, or how much model work was spent before success, abandonment, or rollback.

agentic-os makes that workflow observable. It records structured engineering memory locally and ties related events into deterministic trajectories, so future context packing, debugging, eval replay, policy experiments, and useful-deliverables-per-token measurement can build on raw facts rather than guesswork.

## Why It Is Useful

- It helps agents resume work with relevant prior decisions, failures, and remediations instead of starting cold every turn.
- It preserves validation outcomes, failure signatures, patches, retries, sampling parameters, and token usage as queryable memory.
- It reconstructs the lifecycle of a user intent from request through context, model calls, tools, validations, patches, and final result.
- It stays local-first and deliberately avoids learned routing, scoring, graph databases, autonomous retry policy, or prompt-body archival in the capture layer.

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
# Edit .env: set LITELLM_SALT_KEY.
# Compose provides a default orchestrator API key; change compose.yaml if you want a different key.

./setup-models.sh
docker compose up -d
curl localhost:8088/health/ready
```

## Architecture

The orchestrator is a single-node control plane. It also captures deterministic engineering outcomes such as tool results, test runs, lint failures, patch outcomes, remediations, and inline failure signatures as first-class memory events; see [docs/EXECUTION_FEEDBACK.md](docs/EXECUTION_FEEDBACK.md). It extracts compact operational feature rows and injects bounded corrective guardrails as Operational Constraints; see [docs/FEATURE_EXTRACTION.md](docs/FEATURE_EXTRACTION.md). It captures chat sampling parameters for future outcome-aware routing; see [docs/SAMPLING_PARAMETERS.md](docs/SAMPLING_PARAMETERS.md). It groups request, context, model, tool, validation, patch, remediation, and result events into deterministic trajectories; see [docs/TRAJECTORIES.md](docs/TRAJECTORIES.md). Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the endpoint surface, memory model, retrieval pipeline, summarizer loop, cache behavior, and startup order.

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
- Feature extraction backfill runs with `orchestrator-maint extract-features`.
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

agentic-os assumes a single-user local node, typically reachable only from localhost or a private network such as Tailscale. The orchestrator listens on `0.0.0.0:8088` in Docker Compose, and by default CORS allows any origin to preserve local-tool compatibility. Set `ALLOWED_ORIGINS` to a comma-separated origin list before exposing the API to a LAN, tunnel, or browser-accessible shared network.

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
| `EMBED_MODEL_PATH` | required | Local ONNX embedder path. Compose sets this to `/data/models/embed`. |
| `SENTIMENT_MODEL_PATH` | unset | Optional local sentiment model path for negative feedback detection. |
| `SENTIMENT_THRESHOLD` | `0.70` | Negative feedback classifier threshold when the sentiment model is loaded. |
| `CONTEXT_CACHE_TTL_MS` | `300000` | Context cache TTL. |
| `CONTEXT_DECAY_RATE` | `0.006` | Hybrid retrieval age decay. |
| `EXECUTION_FEEDBACK_ENABLED` | `true` | Enables execution artifact capture and Failure History context. |
| `TRAJECTORY_CAPTURE_ENABLED` | `true` | Enables deterministic trajectory lineage metadata, context-pack lineage events, trajectory results, and the idle sweep. |
| `TRAJECTORY_IDLE_TIMEOUT_SEC` | `600` | Idle duration after which an open trajectory is finalized as unresolved. |
| `FAILURE_HISTORY_TOKEN_BUDGET` | `1000` | Token budget for Failure History context. |
| `FEATURE_EXTRACTION_ENABLED` | `true` | Enables deterministic feature extraction and Operational Constraints context. |
| `BACKGROUND_WORK_CONCURRENCY` | `4` | Max concurrent best-effort derived background jobs such as cache refresh, feature extraction, and Qdrant indexing. |
| `FEATURE_WINDOW_SEC` | `3600` | Session fallback grouping window when trajectory lineage is absent. |
| `CONSTRAINT_FRESHNESS_WINDOW_SEC` | `1800` | Maximum age of detections and recoveries used for active constraints. |
| `MAX_OPERATIONAL_CONSTRAINTS` | `5` | Maximum constraints emitted by the deterministic builder. |
| `OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET` | `300` | Token budget for Operational Constraints context. |
| `SAMPLING_CAPTURE_ENABLED` | `true` | Captures requested and forwarded chat sampling parameters in event metadata. |
| `SAMPLING_OVERRIDE_ENABLED` | `false` | Enables the sampling override hook. Requires sampling capture to stay enabled. |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-key inference refill rate. |
| `RATE_LIMIT_BURST` | `30` | Per-key inference burst. |
| `ALLOWED_ORIGINS` | `*` | CORS origin policy. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Enables OTLP tracing when built with `tracing-otlp`. |

## License

GPL v2
