# agentic-os

A local-first agent operating environment that coordinates LLM inference, memory, orchestration, coding agents, retrieval, workflow state, and tool execution into a unified developer-focused cognitive system.

## Design Philosophy

agentic-os treats LLMs as **computational resources** (planners, workers, language generators) inside a larger cognitive architecture. The orchestrator is the brain stem — responsible for memory, context construction, routing, persistence, and state management. Models are interchangeable compute units.

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

## What This Is

- **A practical local-first agentic development environment**
- **A persistent engineering cognition system** — remember failures, architecture decisions, prior context
- **A unified cognitive architecture** for software engineering workflows
- **Single-node, simple enough to understand and debug**

## What It's Not

- AGI
- A distributed cloud platform
- Kubernetes / Temporal / Kafka / NATS
- Enterprise infrastructure
- Production-grade scheduling
- Autonomous multi-agent societies

## Architecture

| Component | Role | Port |
|-----------|------|------|
| **Rust Orchestrator** | Control plane, memory coordinator, context compiler, routing layer, OpenAI-compatible frontend | 8088 |
| **PostgreSQL** | Durable structured memory — sessions, events, workflow state, checkpoints | 5432 |
| **Qdrant** | Semantic recall layer — vector memory, embeddings | 6333-6334 |
| **LiteLLM** | Model router, inference abstraction layer | 4000 |
| **OpenHands** | Coding agent frontend/runtime | 3000 |

### How components connect

1. **OpenHands** connects to `http://orchestrator:8088/v1` — it sends chat completion requests through the orchestrator
2. **Orchestrator** forwards requests to **LiteLLM** at `http://litellm:4000/v1` for model routing
3. **LiteLLM** routes to local GPU inference at `http://192.168.0.33:8000/v1`
4. **Orchestrator** persists sessions and events in **PostgreSQL**
5. **Orchestrator** indexes events in **Qdrant** for semantic search
6. All OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`) go through the orchestrator

### How memory works

- **Structured memory** (PostgreSQL): Sessions track `repo`, `task`, `actor`. Events record `event_type`, `summary`, `evidence`, `metadata` with cascading deletes.
- **Semantic memory** (Qdrant): Every event is embedded (SHA-256 hash → 384-dim vector, cosine distance) and stored for retrieval.
- **Context packing**: When OpenHands makes a request with `x-agent-repo` and `x-agent-task` headers, the orchestrator retrieves relevant events from Postgres and semantic matches from Qdrant, then injects them as system context.

## Quick Start

```bash
# Start all services
docker compose up -d

# Verify everything is healthy
curl http://localhost:8088/health
curl http://localhost:8088/health/ready
```

## Environment Variables

Copy `.env.example` and adjust as needed:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Postgres connection string | `postgresql://agent:agentpass@postgres:5432/agentstack` |
| `QDRANT_URL` | Qdrant base URL | `http://qdrant:6333` |
| `LITELLM_URL` | LiteLLM base URL + /v1 path | `http://litellm:4000/v1` |
| `LITELLM_KEY` | LiteLLM API key for proxy auth | `sk-local-agentstack` |
| `API_KEY` | API key for orchestrator endpoint auth | `sk-local-orchestrator` |
| `DEFAULT_MODEL` | Default model for completions | `qwen3.6-35b-a3b` |

## API Endpoints

All endpoints use Bearer token auth from `API_KEY` (except health checks).

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check |
| `/health/live` | GET | No | Liveness probe |
| `/health/ready` | GET | No | Readiness probe (checks all services) |
| `/v1/models` | GET | Yes | Model listing (OpenAI-compatible) |
| `/v1/chat/completions` | POST | Yes | Chat completions (streaming + non-streaming) |
| `/sessions/start` | POST | Yes | Create session |
| `/events/append` | POST | Yes | Append event |
| `/context/pack` | POST | Yes | Context pack injection |
| `/summaries/checkpoint` | POST | Yes | Checkpoint summary |
| `/search` | POST | Yes | Semantic search |

## Example curl requests

### Health check
```bash
curl http://localhost:8088/health
# {"status":"ok"}

curl http://localhost:8088/health/ready
# {"status":"ready","services":["postgres","qdrant","litellm"]}
```

### List models
```bash
curl -H "Authorization: Bearer sk-local-orchestrator" http://localhost:8088/v1/models
# {"data":[{"id":"qwen3.6-35b-a3b","object":"model","owned_by":"orchestrator"}],"object":"list"}
```

### Non-streaming chat completions
```bash
curl -H "Authorization: Bearer sk-local-orchestrator" \
     -H "Content-Type: application/json" \
     -H "x-agent-repo: my-project" \
     -H "x-agent-task: fix auth bug" \
     -d '{
  "model": "qwen3.6-35b-a3b",
  "messages": [
    {"role": "user", "content": "Help me fix the login bug"}
  ]
}' http://localhost:8088/v1/chat/completions
```

### Streaming chat completions
```bash
curl -H "Authorization: Bearer sk-local-orchestrator" \
     -H "Content-Type: application/json" \
     -H "x-agent-repo: my-project" \
     -H "x-agent-task: fix auth bug" \
     -d '{
  "model": "qwen3.6-35b-a3b",
  "messages": [
    {"role": "user", "content": "Help me fix the login bug"}
  ],
  "stream": true
}' http://localhost:8088/v1/chat/completions
```

### Start a session
```bash
curl -H "Authorization: Bearer sk-local-orchestrator" \
     -H "Content-Type: application/json" \
     -d '{
  "repo": "my-project",
  "task": "fix auth bug",
  "actor": "agent"
}' http://localhost:8088/sessions/start
# {"session_id":"uuid-here"}
```

### Append an event
```bash
curl -H "Authorization: Bearer sk-local-orchestrator" \
     -H "Content-Type: application/json" \
     -d '{
  "session_id": "session-uuid",
  "repo": "my-project",
  "actor": "agent",
  "event_type": "diagnosis",
  "summary": "Found the bug in auth middleware",
  "evidence": "line 42: missing token validation",
  "metadata": {"confidence": 0.9}
}' http://localhost:8088/events/append
```

### Context pack with semantic retrieval
```bash
curl -H "Authorization: Bearer sk-local-orchestrator" \
     -H "Content-Type: application/json" \
     -d '{
  "repo": "my-project",
  "task": "fix auth bug",
  "limit": 8
}' http://localhost:8088/context/pack
```

### Semantic search
```bash
curl -H "Authorization: Bearer sk-local-orchestrator" \
     -H "Content-Type: application/json" \
     -d '{
  "q": "auth bug",
  "limit": 5
}' http://localhost:8088/search
```

### Checkpoint
```bash
curl -H "Authorization: Bearer sk-local-orchestrator" \
     -H "Content-Type: application/json" \
     -d '{
  "session_id": "session-uuid",
  "repo": "my-project",
  "summary": "Fixed auth middleware, added token validation",
  "next_actions": ["test login flow", "add integration test"],
  "open_questions": ["Should we use JWT or session tokens?"]
}' http://localhost:8088/summaries/checkpoint
```

## Design Constraints

### Local-first
- Function entirely locally
- Avoid SaaS dependencies
- Support fully self-hosted operation
- Cloud APIs optional

### Simplicity over complexity
- Understandable systems
- Debuggable systems
- Explicit logic
- No premature optimization
- No distributed systems complexity

### Orchestrator-centric
- Clients talk to orchestrator
- Clients do NOT talk directly to PG, Qdrant, or LiteLLM
- Orchestrator owns: context, memory, routing, persistence

## MVP Scope

### Required
- Build successfully
- Run via `docker compose up`
- OpenAI-compatible chat completions
- Persistent memory
- Context pack injection
- OpenHands integration
- Local inference support
- Single-node deployment

### Not Required
- Kubernetes
- Distributed orchestration
- Enterprise auth / RBAC
- GPU scheduling
- Advanced semantic ranking
- Internet-scale coordination

## Engineering Philosophy

1. Correctness
2. Simplicity
3. Reproducibility
4. Debuggability
5. Local-first operation

DO NOT chase perfection, AGI, or distributed systems complexity.

## Validation Checklist

After starting with `docker compose up -d`:

| # | Check | Command |
|---|-------|---------|
| 1 | orchestrator builds | `docker compose build orchestrator` |
| 2 | /health works | `curl http://localhost:8088/health` |
| 3 | /health/ready works | `curl http://localhost:8088/health/ready` |
| 4 | streaming works | `curl` with `"stream": true` |
| 5 | /v1/models works | `curl -H "Authorization: Bearer sk-local-orchestrator"` |
| 6 | /v1/chat/completions works | Post request with messages array |
| 7 | sessions persist | POST /sessions/start, check /events/append |
| 8 | events persist | POST /events/append, check /context/pack |
| 9 | memory survives restart | `docker compose down && docker compose up -d` |
| 10 | OpenHands connects | Open localhost:3000 |

## License

GPL v2
