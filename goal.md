# agentic-os

## Mission

agentic-os is a local-first agent operating environment designed to coordinate:

- LLM inference
- memory
- orchestration
- coding agents
- retrieval
- workflow state
- tool execution

into a unified developer-focused cognitive system.

The goal is NOT to build:
- AGI
- a distributed cloud platform
- a Kubernetes replacement
- enterprise infrastructure

The goal IS to build:
- a practical local-first agentic development environment
- capable of persistent memory and orchestration
- optimized for software engineering workflows
- capable of leveraging multiple local inference systems simultaneously

This project is designed primarily for:
- local inference
- GPU workstations
- DGX Spark systems
- Mac Minis
- small homelab environments

---

# Core Design Philosophy

Traditional AI systems are typically:

text user -> chatbot -> model 

agentic-os instead treats models as computational resources inside a larger cognitive architecture.

The orchestrator is the actual "brain stem."

The models are:
- reasoning engines
- planners
- workers
- language generators

The orchestrator is responsible for:
- memory retrieval
- context construction
- workflow coordination
- routing
- persistence
- state management

This architecture treats:
- cognition as infrastructure
- orchestration as the core primitive
- models as interchangeable compute units

---

# Primary Objectives

The MVP must:

1. run entirely locally
2. support local inference
3. support persistent memory
4. support multiple models
5. support orchestration
6. support coding workflows
7. support OpenAI-compatible APIs
8. support OpenHands/opencode
9. survive restarts
10. be simple enough to understand and debug

The MVP is NOT attempting to solve:
- distributed cognition
- autonomous societies of agents
- enterprise multi-tenancy
- production-grade scheduling
- internet-scale orchestration

---

# Intended User Experience

The user experience should feel like:

text persistent engineering cognition 

The user should be able to:
- stop and restart the system
- resume work later
- retrieve prior engineering context
- coordinate multiple agents
- leverage multiple GPUs/models
- continuously improve engineering throughput

The system should:
- remember prior work
- remember failures
- remember architecture decisions
- provide relevant context automatically
- reduce repeated prompting

The user should interact primarily with:
- OpenHands
- opencode
- Claude Code
- OpenAI-compatible clients

Those systems should communicate with the orchestrator.

The orchestrator should coordinate:
- memory
- retrieval
- model routing
- persistence

---

# System Architecture

## High-level Architecture

text                 ┌────────────────────┐                 │ OpenHands          │                 │ opencode           │                 │ Claude Code        │                 └─────────┬──────────┘                           │                           ▼                 ┌────────────────────┐                 │ Rust Orchestrator  │                 │ "brain stem"       │                 └─────────┬──────────┘                           │        ┌──────────────────┼──────────────────┐        │                  │                  │        ▼                  ▼                  ▼ ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │ Postgres    │   │ Qdrant      │   │ LiteLLM     │ │ state       │   │ memory      │   │ router      │ └─────────────┘   └─────────────┘   └──────┬──────┘                                             │                                             ▼                              ┌─────────────────────────┐                              │ vLLM / Ollama / Spark   │                              │ local inference         │                              └─────────────────────────┘ 

---

# Component Responsibilities

## OpenHands / opencode / Claude Code

These are:
- interfaces
- coding runtimes
- agent frontends

They SHOULD NOT:
- own memory
- own orchestration
- own routing

They SHOULD:
- communicate with the orchestrator
- receive context from the orchestrator
- execute coding workflows

---

## Rust Orchestrator

The orchestrator is:
- the control plane
- the memory coordinator
- the context compiler
- the routing layer
- the OpenAI-compatible frontend

The orchestrator is responsible for:
- building context packs
- retrieving memory
- routing requests
- persisting events
- coordinating workflows
- exposing OpenAI-compatible APIs

The orchestrator is NOT:
- the inference engine
- the vector database
- the coding runtime

---

## LiteLLM

LiteLLM is:
- the model router
- the inference abstraction layer

LiteLLM is responsible for:
- routing model requests
- supporting multiple backends
- normalizing APIs

LiteLLM SHOULD NOT:
- own orchestration
- own memory
- own workflow logic

---

## Postgres

Postgres stores:
- sessions
- events
- workflow state
- checkpoints
- summaries
- structured memory

Postgres is:
- the durable structured memory layer

---

## Qdrant

Qdrant stores:
- vector memory
- semantic retrieval data
- memory embeddings

Qdrant is:
- the semantic recall layer

---

## Local Inference

Inference backends may include:
- vLLM
- Ollama
- llama.cpp
- DGX Spark
- RTX 5090 systems

Models are:
- computational resources
- interchangeable workers

The architecture should support:
- multiple models
- multiple GPUs
- multiple runtimes

---

# Design Constraints

## Local-first

The system MUST:
- function locally
- avoid SaaS dependencies
- support fully self-hosted operation

Cloud APIs are optional.

---

## Simplicity over complexity

The MVP should prefer:
- understandable systems
- debuggable systems
- explicit logic

DO NOT:
- overengineer
- prematurely optimize
- add distributed systems complexity

---

## Orchestrator-centric design

Clients MUST talk to:
- the orchestrator

Clients MUST NOT talk directly to:
- LiteLLM
- Qdrant
- Postgres

The orchestrator owns:
- context
- memory
- routing
- persistence

---

# MVP Scope

## Required

The MVP MUST:
- build successfully
- run via docker compose up
- support OpenAI-compatible chat completions
- support persistent memory
- support context pack injection
- support OpenHands
- support local inference
- support a single-node deployment

---

## Explicit Non-goals

DO NOT IMPLEMENT:
- Kubernetes
- Temporal
- Kafka
- NATS
- LangGraph
- distributed orchestration
- enterprise auth
- RBAC
- cloud-native architectures
- GPU scheduling
- autonomous multi-agent societies
- advanced semantic ranking
- internet-scale coordination

This is NOT a production platform.

This is a:
- local-first engineering cognition system

---

# MVP Technical Requirements

## Required Endpoints

The orchestrator MUST support:

text GET  /health GET  /health/live GET  /health/ready GET  /v1/models  POST /v1/chat/completions  POST /sessions/start POST /events/append POST /context/pack POST /summaries/checkpoint 

---

# Required Features

## OpenAI Compatibility

The orchestrator MUST:
- expose OpenAI-compatible APIs
- support streaming responses
- support model listing

This is required for:
- OpenHands
- opencode
- Claude Code compatibility

---

## Persistence

The system MUST:
- persist sessions
- persist events
- survive restarts

Persistence is more important than advanced intelligence.

---

## Context Injection

The orchestrator MUST:
- retrieve prior memory
- construct context packs
- inject memory automatically

The goal is:
- persistent engineering cognition

---

## Local Inference Support

The system MUST support:
- vLLM
- Ollama
- local inference backends

The architecture SHOULD support:
- multiple concurrent models

---

# Current Known Gaps

## Rust

### Missing chrono feature

Fix:

toml tokio-postgres = { version = "0.7.17", features = ["with-serde_json-1", "with-uuid-1", "with-chrono-0_4"] } 

---

### Missing Cargo.lock

Generate:

bash cargo generate-lockfile 

Commit:
- Cargo.lock

---

## Dockerfile

Use:

dockerfile FROM cgr.dev/chainguard/rust:latest-dev AS builder  WORKDIR /app  COPY Cargo.toml Cargo.lock ./ COPY src ./src  RUN cargo build --release  FROM cgr.dev/chainguard/wolfi-base:latest  WORKDIR /app  COPY --from=builder /app/target/release/orchestrator /usr/local/bin/orchestrator  EXPOSE 8088  ENTRYPOINT ["/usr/local/bin/orchestrator"] 

DO NOT:
- use crt-static
- use musl hacks

---

## Orchestrator Gaps

The orchestrator currently lacks:
- /v1/models
- streaming support
- auth enforcement
- postgres connection pooling
- proper semantic retrieval
- health checks
- structured logging

These MUST be added.

---

## Compose Gaps

The compose file currently lacks:
- healthchecks
- image digest pinning
- proper service readiness checks
- environment file support

These should be improved.

---

# Success Criteria

The MVP is successful if:

1. docker compose up works
2. OpenHands launches
3. OpenHands connects successfully
4. orchestrator proxies requests successfully
5. LiteLLM routes requests successfully
6. local models respond successfully
7. sessions persist
8. events persist
9. memory survives restarts
10. coding workflows become easier

---

# Engineering Philosophy

Prioritize:
1. correctness
2. simplicity
3. reproducibility
4. debuggability
5. local-first operation

DO NOT:
- chase perfection
- chase AGI
- chase distributed systems complexity

The objective is:
- practical engineering cognition infrastructure
- for local-first software development workflows
