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
    ┌────────────┼─────────────┐
    ▼            ▼             ▼
 ┌──────┐   ┌────────┐   ┌─────────┐
 │PGSQL │   │Qdrant  │   │LiteLLM  │
 │state │   │memory  │   │router   │
 └──────┘   └────────┘   └────┬────┘
                               │
                       ┌───────▼──────┐
                       │ local GPU    │
                       │ inference    │
                       └──────────────┘
```

## Quick Start

```bash
cp .env.example .env
# Edit .env: set LITELLM_SALT_KEY and add your API keys to API_KEYS

# First-time only: download the embedding model (~125MB).
# This runs via Docker so no Python install is needed.
./setup-models.sh

docker compose up -d
```

```bash
# Verify health
curl localhost:8088/health/ready
# {"status":"ready","services":["postgres","qdrant","litellm"]}
```

> **Why `setup-models.sh`?** The TEI embedding container uses `hf-hub 0.3.x` which cannot follow relative URL redirects returned by the HuggingFace CDN. Pre-downloading via Python's `huggingface_hub` (which handles redirects correctly) sidesteps the crash. Run it once; the model is cached in `./models/` and reused on every subsequent `docker compose up`.

## Memory namespaces via API key

The orchestrator derives a memory namespace from the API key — no custom headers needed.

```
sk-work     → namespace: work
sk-home     → namespace: home
sk-myteam   → namespace: myteam
```

Each namespace is fully isolated. Past sessions, context packs, and semantic recall are scoped to that namespace. Different clients using different keys get completely separate memory.

Configure multiple keys in `.env`:

```
API_KEYS=sk-work,sk-home,sk-project-x
```

Clients need only two settings — base URL and API key:

```bash
# Work context
opencode --api-url http://localhost:8088/v1 --api-key sk-work

# Personal context
opencode --api-url http://localhost:8088/v1 --api-key sk-home
```

Memory is automatically separated. No headers, no configuration beyond the key.

### Explicit routing (optional)

For finer-grained control, override the namespace per-request with headers:

```bash
curl -H "Authorization: Bearer sk-work" \
     -H "x-agent-repo: payments-service" \
     -H "x-agent-task: fix auth bug" \
     -d '{"model":"qwen36-35b-heretic","messages":[...]}' \
     localhost:8088/v1/chat/completions
```

`x-agent-repo` overrides the key-derived namespace; `x-agent-task` overrides `DEFAULT_TASK`. Headers are optional — absent headers fall back to key derivation.

## Architecture

| Component | Role | Port |
|-----------|------|------|
| **Rust Orchestrator** | Control plane, memory coordinator, context compiler, OpenAI-compatible frontend | 8088 |
| **PostgreSQL** | Structured memory — sessions, events, workflow state | 5432 |
| **Qdrant** | Semantic recall — vector memory | 6333-6334 |
| **TEI (embedding)** | Text embeddings for semantic search (`BAAI/bge-small-en-v1.5`) | 8001 |
| **LiteLLM** | Model routing and inference abstraction | 4000 |

### How it works

1. Clients send requests to the orchestrator using a standard OpenAI-compatible API
2. The orchestrator derives a memory namespace from the API key and injects relevant context into the system prompt
3. Inference is forwarded to LiteLLM which routes to a local GPU
4. Every exchange is persisted to Postgres and indexed in Qdrant for future semantic recall

## API Endpoints

All endpoints use Bearer token auth (`Authorization: Bearer <key>`).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/health/ready` | GET | Readiness (postgres, qdrant, litellm) |
| `/v1/models` | GET | Model listing |
| `/v1/chat/completions` | POST | Chat completions (streaming + non-streaming) |
| `/sessions/start` | POST | Create a named session |
| `/events/append` | POST | Append an event to memory |
| `/context/pack` | POST | Fetch and format context for a namespace |
| `/summaries/checkpoint` | POST | Write a checkpoint summary |
| `/search` | POST | Semantic search over memory |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Postgres connection string | required |
| `QDRANT_URL` | Qdrant base URL | required |
| `LITELLM_URL` | LiteLLM base URL including `/v1` | required |
| `LITELLM_KEY` | LiteLLM API key | required |
| `LITELLM_SALT_KEY` | LiteLLM salt key | required |
| `EMBEDDING_URL` | TEI embedding service base URL | `http://embedding:80` |
| `API_KEYS` | Comma-separated orchestrator API keys | `sk-local-orchestrator` |
| `DEFAULT_MODEL` | Default model for completions | `qwen36-35b-heretic` |
| `DEFAULT_TASK` | Default task label when header absent | `engineering` |

## License

GPL v2
