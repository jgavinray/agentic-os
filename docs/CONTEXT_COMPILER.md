# Context Compiler

## Executive Summary

The context compiler is not another event collector. It is the layer that turns
existing episodes into current working knowledge.

Today Agentic OS can remember that a request happened, a tool ran, a model
failed, or a summary was written. That history is useful, but it leaves the next
model call with too much interpretation work. The model has to decide which old
facts are still true, which ones were superseded, which failures were resolved,
and which evidence belongs in the prompt.

The compiler moves that work out of the model call path. It continuously
maintains prompt-ready artifacts:

```text
event history:
  "LiteLLM returned 500."
  "Connection refused to 192.168.0.33:8000."
  "/v1/models later returned qwen36-35b-a3b."

compiled artifact:
  Current model backend is reachable at http://192.168.0.33:8000/v1.
  Prior outage on port 8000 is resolved.
  Source events: [...]
  Prompt placement: service topology + failure history.
```

That difference is where signal and fidelity improve. The raw evidence remains
available, but the prompt receives the best current representation of it.

## Purpose

Agentic OS already captures a non-trivial event stream. The context compiler adds
the missing intermediate representation between event history and prompt text:
typed, status-aware, prompt-ready artifacts.

Events answer what happened. Context artifacts answer what the model should use
now, why it is valid, where it came from, and where it belongs in the prompt.

## Problem Statement

The current context path has three practical failure modes:

1. **Meaning is inferred too late.** Event retrieval happens near the request
   path, so stale and current facts can be retrieved together. The model then
   has to resolve contradictions in the same prompt it is using to solve the
   user task.
2. **Fidelity is collapsed too early.** Many events have rich payloads, but the
   context renderer mostly emits summaries and short evidence strings. If the
   summary is weak, the model never sees the important detail.
3. **Stable prefixes are accidental.** The injected context changes with recent
   events, request task text, cache keys, and retrieval ordering. That makes
   vLLM/provider KV cache reuse less likely, even when the underlying repo facts
   are stable.

The compiler addresses these by separating three concerns:

```text
capture
  Preserve raw history and evidence.

compile
  Convert history into typed, status-aware knowledge.

assemble
  Render stable prefix bands plus dynamic tail context.
```

## Memory Ownership

```text
Agentic OS Postgres
  primary working-memory ledger for this product:
  sessions, events, summaries, execution feedback, failure signatures,
  operational constraints, trajectories, and compiled artifacts

Qdrant
  semantic index over Agentic OS events:
  fast candidate lookup, not a source of truth

Total Recall
  optional external episodic memory:
  human-readable notes and cross-agent observations outside the Agentic OS
  event ledger

context_artifacts
  typed working knowledge:
  active instructions, durable project facts, service topology, decisions,
  resolved failures, known bad approaches, and current constraints

context compiler
  promotion and consolidation layer:
  turns memory candidates into bounded working knowledge
```

This boundary matters. The compiler should not create a second event database,
and it should not treat Total Recall as required when Agentic OS already has the
relevant execution history. Agentic OS Postgres is the primary memory source for
orchestrator behavior. Total Recall is useful when memory must outlive or span
Agentic OS sessions, agents, tools, or products.

Qdrant is also not the source of truth. It is an index over event evidence.
Vector search can propose candidates, but the compiler decides whether a
candidate represents current working knowledge.

## Runtime Layering

```text
memory stores
  Agentic OS Postgres, Qdrant indexes, optional Total Recall episodes

candidate selection
  exact lookups, semantic search, recency, failure signatures, explicit markers

context compiler
  classify, extract claims, resolve status, consolidate, rank, and render

context_artifacts
  stable, typed, inspectable working knowledge with provenance and hashes

prompt assembler
  stable prefix bands plus dynamic request-tail context

model serving / KV cache
  reuses compute for byte-stable prompt prefixes
```

The KV cache is separate from memory. It only accelerates repeated token
prefixes. The context compiler decides what those prefixes should contain.

## Caching Versus Memory

Several caches can exist in this stack, but they are not interchangeable with
memory or compilation.

```text
Request/context cache
  Stores a previously assembled context pack for a short period.
  Value: avoids rebuilding the same pack repeatedly.
  Limitation: does not improve the knowledge inside the pack.

Compiled artifact cache
  Stores promoted working knowledge in `context_artifacts`.
  Value: keeps stable, typed, inspectable facts available across requests.
  Limitation: only useful if the compiler promotes the right claims.

Provider or vLLM KV cache
  Reuses model attention compute for byte-identical prompt prefixes.
  Value: lowers latency/cost for stable prefixes.
  Limitation: does not decide what should be remembered.

Episodic memory
  Stores observations and events outside the immediate context window.
  Value: preserves raw history and provenance.
  Limitation: raw recall alone still leaves interpretation to the model.

Context compiler
  Promotes memory candidates into current working knowledge.
  Value: improves signal and fidelity before the model sees the prompt.
```

Caching answers "can we reuse this?" Memory answers "what happened?" The
compiler answers "what is currently true, useful, and safe to place in the
prompt?"

## Why This Is Not RAG

Retrieval-augmented generation usually retrieves chunks and asks the model to
interpret them at answer time. That gets weaker as context grows because the
model has to sort relevance, contradiction, freshness, and authority inside the
same prompt it is using to solve the task.

The compiler should do the opposite:

```text
RAG-style path:
  retrieve N chunks -> paste chunks into prompt -> model resolves meaning

compiler path:
  collect candidates -> promote current claims -> render bounded artifacts
```

The prompt should receive artifacts, not raw memory dumps. Raw episodes and
events remain available for audit and follow-up retrieval, but ordinary model
calls should see compact statements of current working knowledge.

The effect we want is compounding signal, not compounding context size:

- Postgres contributes authoritative local execution history.
- Qdrant contributes candidate recall over that history.
- Total Recall contributes external episodic observations when explicitly
  configured.
- The compiler contributes status resolution, consolidation, provenance, and
  stable prompt placement.

## Artifact Contract

Each artifact should carry enough information for deterministic prompt assembly:

- `scope`: `global`, `repo`, `session`, `trajectory`, or `request`
- `artifact_type`: for example `service_topology`, `repo_map`,
  `durable_project_memory`, `failure_history`, `active_instruction`, or
  `tool_trace`
- `status`: `active`, `superseded`, `stale`, `resolved`, `disputed`, or
  `quarantined`
- raw/compact/rendered content
- source event IDs and source file paths
- an invalidation key
- content hash and token estimate

The rendered form is what the prompt sees. The raw and compact forms preserve
fidelity and retrieval signal.

## What An Artifact Means

An artifact is a statement of current working knowledge, not just a note.

Example event:

```text
event_type: failed_attempt
summary: LiteLLM returned 500
evidence: ConnectionRefusedError to 192.168.0.33:8000
```

Example artifacts derived from the event stream:

```text
artifact_type: failure_history
status: resolved
rendered:
  Prior failure: LiteLLM could not connect to 192.168.0.33:8000.
  Resolution: later /v1/models returned qwen36-35b-a3b.

artifact_type: service_topology
status: active
rendered:
  LiteLLM upstream: http://192.168.0.33:8000/v1
  Default model: qwen36-35b-a3b
```

This lets the model see both the current fact and the prior failure in their
correct roles. It no longer has to infer from raw chronology that the outage is
resolved.

## Prompt Bands

The compiler should eventually assemble context in this order:

```text
0 global agent contract
1 orchestrator/tool contract
2 repo map
3 service topology
4 durable project memory
5 repo decisions and failures
6 active session state
7 recent raw evidence
8 current request
```

Bands 0-5 should be byte-stable whenever possible. Bands 6-8 are expected to
change often.

## Compiler Pipeline

The compiler has a deterministic pipeline. Learned ranking can be added inside
it later, but the pipeline itself should remain inspectable:

```text
1. collect candidates
   agent_events, Total Recall observations, repo files, compose files, logs

2. classify
   instruction, repo fact, service topology, failure, remediation, tool trace

3. extract claims
   subject, claim text, evidence, source records, source files

4. resolve status
   active, superseded, stale, resolved, disputed, quarantined

5. choose representation
   raw, compact, stable rendered text, dynamic rendered text

6. assign band
   service topology, durable memory, failure history, recent evidence, etc.

7. assemble prompt
   stable prefix first, dynamic tail last

8. write ledger
   what was included, what was dropped, and why
```

XGBoost belongs in steps 4-6 after enough outcome data exists. It should score
inclusion and representation, not replace hard rules like "explicit active user
instructions must be included" or "superseded facts must not appear as active
truth."

## Candidate Sources

The compiler should read from memory systems in priority order. This avoids
treating every store as equally authoritative.

```text
1. Explicit current request and active session state
   Highest priority. Direct user instructions and current task constraints.

2. Agentic OS Postgres
   Primary orchestrator memory. Events, summaries, failures, validations,
   trajectories, operational constraints, and prior compiled artifacts.

3. Deterministic indexes and exact lookups
   Failure signatures, active errors, source file paths, compose/runtime
   configuration, and known service topology.

4. Qdrant semantic candidates
   Recall over Agentic OS event evidence. Used for discovery, not final truth.

5. Total Recall episodic candidates
   External memories and operator-written observations. Used when configured and
   relevant; never required for core Agentic OS working memory.
```

Each source can improve signal, but only after the compiler promotes its output
into an artifact. The prompt does not get a concatenation of source results.

## Promotion Rules

Promotion is the difference between memory and context. A candidate becomes a
context artifact only when the compiler can assign:

- artifact type
- scope
- status
- compact claim text
- provenance
- invalidation key
- prompt band

Examples:

```text
candidate:
  User said "be explicit" several times across separate events.

artifact:
  type: active_instruction
  status: active
  rendered: Be explicit about configuration, defaults, and assumptions.
  provenance: source event IDs
```

```text
candidate:
  A failed attempt used a stale backend URL, then a later validation succeeded.

artifact:
  type: failure_history
  status: resolved
  rendered: Prior backend connection failure is resolved; current backend route
            has been validated.
  provenance: failed event ID + validation event ID
```

The compiler should prefer fewer, stronger artifacts over more retrieved text.
Artifact count and token budget are bounded so memory improves precision instead
of increasing prompt entropy.

## Compounding Signal

The memory systems compound signal only when each one contributes a distinct
kind of evidence:

```text
Postgres says what Agentic OS observed.
Qdrant helps find related observations in that ledger.
Total Recall says what external agents or humans recorded.
Compiler says what is currently true and useful.
KV cache makes stable rendered context cheaper to reuse.
```

The compiler should explicitly record why a candidate was included, suppressed,
or superseded. That ledger is how we keep the system inspectable as learned
ranking is added.

## Diagnostics

The orchestrator exposes compiler state through:

```text
GET /context/artifacts?repo=<repo>&limit=25&ledger_limit=50
```

The response includes:

- current and recent `context_artifacts`
- recent `context_compiler_ledger` decisions
- artifact provenance through source event IDs and source file paths
- inclusion/suppression reasons for promoted candidates

This endpoint is intentionally operational. It lets an operator answer:

- which working-knowledge artifacts are active
- which memory candidates were promoted
- which candidates were suppressed or dropped
- whether the compiler is improving prompt signal or just adding context

## First Milestone

The first value-bearing slice is a `service_topology` artifact compiled from
runtime orchestrator configuration. It gives the model stable knowledge of:

- default model
- LiteLLM endpoint
- Qdrant endpoint
- summarizer endpoint and model
- known local service roles

This directly helps local-stack debugging and gives us the storage/rendering
path for richer artifacts later.

This milestone should deliver value quickly because it removes a repeated blind
spot: local model requests often fail due to routing, port, or service state.
The model should not have to rediscover the stack topology from scratch.

Expected first visible improvement:

```text
Before:
  The model sees a few recent summaries and may not know how orchestrator,
  LiteLLM, Qdrant, summarizer, and the vLLM backend relate.

After:
  Every context pack can include a stable service-topology artifact that names
  the runtime endpoints and model route.
```

## Immediate Value Criteria

The compiler should be judged by whether it changes the very next model request,
not by whether it completes a large memory roadmap.

The first useful slice must satisfy these criteria:

1. A context pack includes at least one compiled artifact that was not available
   through ordinary event summary retrieval.
2. The artifact has stable rendered text, a content hash, and an invalidation
   key.
3. The artifact carries source/provenance fields, even if the first artifact is
   derived from runtime configuration rather than source events.
4. The rendered text helps with an active operational problem in this stack.

The service-topology artifact meets that bar because local-stack failures often
come from endpoint, model, or service wiring. The model should know that wiring
before it starts interpreting logs.

The next implemented slices are:

- `repo_map`: stable map of core orchestrator modules and their roles.
- `active_instruction`: recent explicit user instructions from Agentic OS
  Postgres promoted into bounded working context.
- `failure_history`: resolved failure/remediation pairs promoted from
  execution-feedback events.
- `durable_project_memory`: optional Total Recall notes promoted through the
  explicit HTTP API when `TOTAL_RECALL_URL` is configured.
- `repo_decisions`: recent decision/checkpoint candidates promoted from the
  Agentic OS event ledger.
- `session_state`: recent active-session events promoted when a session ID is
  available.
- `context_compiler_ledger`: inclusion and suppression records for compiler
  decisions.
- `stable_prefix_hash` and `dynamic_tail_hash`: separate context-pack hashes
  for inspecting cacheable compiler prefix stability.

## Total Recall Deployment

Total Recall is not required for the first service-topology slice. That slice is
compiled from Agentic OS runtime configuration and works without an external
episodic memory service.

Total Recall is an optional episodic source. It becomes useful when the compiler
needs durable observations outside Agentic OS' own event ledger: cross-agent
notes, operator-written memories, or project context captured by a different
tool. It should run as an HTTP MCP service on the Compose network so the
orchestrator can query it without shelling out or depending on a desktop MCP
client.

This repo provides an optional Compose overlay:

```bash
docker compose -f compose.yaml -f compose.total-recall.yaml up -d total-recall
```

Agentic OS consumes that service only when the orchestrator has an explicit URL:

```bash
TOTAL_RECALL_URL=http://total-recall:8811
```

The compiler reads `GET /api/recent` and task-specific `POST /api/search` from
Total Recall, deduplicates note IDs, and promotes bounded notes into
`durable_project_memory`. It does not paste raw note dumps into the prompt.

The overlay mounts explicit container paths through `total-recall-config.yaml`.
That matters because Total Recall defaults to `~/.total-recall`; inside a
container that would otherwise be container-local state unless configured.

Total Recall should use the same embedding model as Agentic OS. The Compose
overlay mounts Agentic OS' local Snowflake `snowflake-arctic-embed-l-v2.0`
model directory at `/data/models/embed`, and `total-recall-config.yaml`
declares that model path with a 1024-dimensional embedding size.

That model alignment matters. If Total Recall and Agentic OS index memories in
different semantic spaces, cross-store recall will be noisy and difficult to
reason about. The first integration requirement is therefore:

```text
Total Recall must load the configured Snowflake embedding model, and its vector
table dimension must match the configured model.
```

The local Total Recall configuration is explicit:

```yaml
embedding:
  model: Snowflake/snowflake-arctic-embed-l-v2.0
  model_path: /data/models/embed
  dimension: 1024
  cache_dir: /data/models/embed
```

Agentic OS should still compile from its own Postgres memory first. Total Recall
should add episodic signal only when it has relevant external memory that the
Agentic OS ledger does not already contain.

## Future Milestones

1. Add learned ranking after enough compiler-ledger and outcome data exists.
2. Add stronger learned suppression for noisy decision/session candidates.
