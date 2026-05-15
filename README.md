# agentic-os

A local-first agent operating environment that coordinates LLM inference, memory, orchestration, retrieval, and tool execution into a unified cognitive architecture. It treats LLMs as interchangeable compute units inside a brain stem orchestrator — responsible for context construction, routing, persistence, and state management — so developers get persistent engineering memory, semantic recall, and automated context packing in a single Docker Compose stack.

```
    Claude Code / OpenHands / opencode
                │
                ▼
        ┌────────────────┐
        │  Rust          │
        │  Orchestrator  │
        │  (brain stem)  │
        └────────┬───────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
 ┌──────┐   ┌────────┐   ┌─────────┐
 │PGSQL │   │Qdrant  │   │LiteLLM  │
 │state │   │memory │   │router   │
 └──────┘   └────────┘   └────┬────┘
                              │
                      ┌───────▼──────┐
                      │ local GPU    │
                      │ inference    │
                      └──────────────┘
```

## Quick Start

```bash
# Boot the full stack
cp .env.example .env
docker compose up -d
```

```bash
# Verify health
curl localhost:8088/health/ready
# {"status":"ready","services":["postgres","qdrant","litellm"]}
```

## Use with opencode

```bash
opencode -e http://localhost:8088/v1 -k sk-local-orchestrator "fix the login bug in my-project"
```

The orchestrator headers (`x-agent-repo`, `x-agent-task`) are handled automatically.

## Architecture

| Component | Role | Port |
|-----------|------|------|
| **Rust Orchestrator** | Control plane, memory coordinator, context compiler, OpenAI-compatible frontend | 8088 |
| **PostgreSQL** | Structured memory — sessions, events, workflow state | 5432 |
| **Qdrant** | Semantic recall — vector memory, embeddings | 6333-6334 |
| **LiteLLM** | Model routing and inference abstraction | 4000 |

### How it works

1. Clients (opencode, OpenHands, curl) send requests to the orchestrator
2. The orchestrator forwards inference requests to LiteLLM, which routes to a local GPU
3. Every interaction is persisted to Postgres and indexed in Qdrant for semantic recall
4. Context packing automatically injects relevant past events as system context

## API Endpoints

All endpoints use Bearer token auth (`Authorization: Bearer <API_KEY>`).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | Model listing |
| `/v1/chat/completions` | POST | Chat completions (streaming + non-streaming) |
| `/sessions/start` | POST | Create session |
| `/events/append` | POST | Append event |
| `/context/pack` | POST | Context pack injection |
| `/summaries/checkpoint` | POST | Checkpoint summary |
| `/search` | POST | Semantic search |

## Example requests

```bash
# Chat completions
curl -H "Authorization: Bearer sk-local-orchestrator" \
     -H "Content-Type: application/json" \
     -H "x-agent-repo: my-project" \
     -H "x-agent-task: fix auth bug" \
     -d '{"model":"qwen36-35b-heretic","messages":[{"role":"user","content":"Help me fix the login bug"}]}' \
     localhost:8088/v1/chat/completions
```

## Design constraints

- **Local-first** — runs on a single node; cloud APIs are optional
- **Simplicity** — no Kubernetes, no distributed complexity, no enterprise auth
- **Orchestrator-centric** — clients talk only to the orchestrator; it owns all memory and routing

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Postgres connection string | `postgresql://agent:agentpass@postgres:5432/agentstack` |
| `QDRANT_URL` | Qdrant base URL | `http://qdrant:6333` |
| `LITELLM_URL` | LiteLLM base URL + /v1 path | `http://litellm:4000/v1` |
| `LITELLM_KEY` | LiteLLM API key | `sk-local-agentstack` |
| `API_KEY` | Orchestrator auth | `sk-local-orchestrator` |
| `DEFAULT_MODEL` | Default model | `qwen36-35b-heretic` |

## License

GPL v2
