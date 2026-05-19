# Architecture

agentic-os is a single-node Rust orchestrator that fronts LiteLLM-compatible model APIs and adds durable engineering memory. It stores structured state in Postgres, vector recall in Qdrant, and builds context packs that are injected into proxied model requests.

## Public Endpoints

All endpoints except `/health`, `/health/live`, and `/health/ready` require `Authorization: Bearer <api-key>`.

| Endpoint | Method | Behavior |
| --- | --- | --- |
| `/health` | GET | Liveness check for the HTTP process. |
| `/health/live` | GET | Liveness check alias used by container health checks. |
| `/health/ready` | GET | Checks Postgres, Qdrant, and LiteLLM reachability. |
| `/v1/models` | GET | Proxies LiteLLM `/models`; falls back to the configured default model when LiteLLM cannot return JSON. |
| `/v1/chat/completions` | POST | OpenAI-compatible chat endpoint. It derives namespace/task, injects memory context, forwards to LiteLLM, streams when requested, and persists user/assistant exchanges. |
| `/v1/messages` | POST | Anthropic-compatible messages passthrough. It keeps Anthropic format, injects memory into `system`, forwards to LiteLLM `/messages`, and persists exchanges. |
| `/sessions/start` | POST | Creates an explicit agent session for a repo/task/actor. |
| `/events/append` | POST | Inserts a memory event in Postgres and best-effort indexes it in Qdrant. |
| `/context/pack` | POST | Builds or returns a cached layered context pack for a repo/task. |
| `/cache/stats` | GET | Returns in-process context cache size and TTL. |
| `/metrics` | GET | Returns Prometheus text exposition. |
| `/metrics/json` | GET | Returns the legacy JSON metrics snapshot. |
| `/summaries/checkpoint` | POST | Persists an operator checkpoint event with next actions and open questions. |
| `/search` | POST | Runs semantic search over Qdrant memory. |

## Namespace Model

`API_KEYS` is a semicolon-delimited list of `token,namespace` entries. Authentication uses constant-time token comparison. The token selects a default memory namespace; `x-agent-repo` can override the namespace per request, and `x-agent-task` can override the task label. Namespaces isolate sessions, events, context packs, and recall.

## Memory Storage

Postgres owns durable structured memory:

- `agent_sessions` tracks work sessions by repo/task/actor.
- `agent_events` stores raw events, summaries, checkpoints, promotion state, and trajectory lineage columns (`trajectory_id`, `attempt_index`, `event_role`).
- `agent_feature_records` stores rebuildable operational feature aggregates and bounded constraint recommendations derived from `agent_events`.
- `error_index` aggregates repeated failures by repo/task/type/description.
- `token_usage` records model token accounting.

Qdrant stores vectors for events in the `agent_events` collection. Qdrant indexing is best-effort: a Postgres write can succeed even if vector indexing fails.

## Trajectory Lineage

Trajectory capture groups one user intent and its downstream context pack, model response, tools, validations, patches, remediations, failures, and final result using the existing event log. A request starts a new `trajectory_id`; follow-on events inherit it. Context packs are written before the model response they inform, and the model response stores the `context_pack_id` back-reference.

Completion writes one idempotent `trajectory_result` event with bounded statuses: `succeeded`, `abandoned`, `unresolved`, or `reverted`. `TRAJECTORY_CAPTURE_ENABLED=false` disables trajectory metadata, result emission, the idle sweep, and trajectory metrics. See [TRAJECTORIES.md](TRAJECTORIES.md).

## Retrieval Pipeline

Context construction combines semantic and lexical recall:

1. Recent and promoted memories are read from Postgres by layer.
2. Semantic hits come from Qdrant using the local ONNX embedder.
3. Lexical hits come from Postgres full-text search over summaries and evidence.
4. Semantic and FTS hits are merged with reciprocal rank fusion.
5. Scores are decayed by age so stale events are demoted but still retrievable.
6. Duplicate summaries are removed and section budgets are enforced.

## Memory Layers

Raw events start at L0. The summarizer promotes memory through L1, L2, and L3 when enough source material and promotion signals exist.

| Layer | Meaning | Typical Use |
| --- | --- | --- |
| L0 | Recent raw events | Immediate local context. |
| L1 | Session summaries | Relevant prior task context. |
| L2 | Repo-level summaries | Reusable repo conventions and decisions. |
| L3 | Project-level memory | Durable architecture truths. |

The context policy changes layer weights by task category: narrow/debug work favors fresh L0/L1 plus failures, while architecture work favors L2/L3.

Operational Constraints are assembled from deterministic feature records and placed immediately above Failure History. They are capped by count and token budget and contain only compact corrective text, never raw events or metadata. See [FEATURE_EXTRACTION.md](FEATURE_EXTRACTION.md).

## Summarizer Loop

A single background loop ticks every 60 seconds. It finds sessions with enough unsummarized messages, takes a Postgres advisory lock for each session/target level, asks LiteLLM for a concise summary, inserts a new `summary` event, marks source events summarized, and invalidates context cache entries for the repo.

## Context Cache

Context packs are cached in process. Model requests use stale-while-revalidate semantics: the orchestrator injects the newest cached pack for the repo/task/session scope, or a minimal fallback when no cache exists, then refreshes the full pack in a coalesced background task. This keeps context computation, semantic retrieval, and constraint lookup off the first-token path. The explicit `/context-pack` endpoint still builds synchronously because callers use it to fetch the pack itself.

Stored cache keys use `repo:task:event_count`; optional limit overrides are folded into the task portion. `CONTEXT_CACHE_TTL_MS` controls expiration for exact cache hits. Appending events, checkpoints, persisted exchanges, and summaries invalidate entries for the affected repo.

Request events are durably inserted into Postgres before forwarding, but their Qdrant indexing is background best-effort work so vector embedding does not block request setup.

## Startup Order

On boot the orchestrator:

1. Installs logging and metrics.
2. Creates the Postgres pool.
3. Acquires the single-writer advisory lock.
4. Runs embedded refinery migrations.
5. Initializes the Qdrant collection.
6. Loads local embedder and optional sentiment models.
7. Starts the summarizer and HTTP server.

The single-writer lock means exactly one orchestrator process may own a given Postgres database.
