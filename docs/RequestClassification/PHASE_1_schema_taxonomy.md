# Phase 1: Schema, Taxonomy, And Structs

## Goal

Create the request classification storage and bounded Rust taxonomy. Do not
extract features, backfill rows, or change live routing in this phase.

## Scope

Implement:

- SQL migration for `agent_request_classifications`.
- Rust enum definitions for intent, domain, artifact type, complexity, risk,
  route, and response contract.
- Schema and policy version constants.
- Rust structs mirroring the table shape.
- Tests for enum closure and schema privacy.

Do not implement:

- Feature extraction.
- Backfill jobs.
- Boot-time classification.
- CLI commands.
- Route enforcement.
- LLM/model selection changes.

## Table

Create `agent_request_classifications`:

```text
event_id text references agent_events(id)
repo text not null
session_id text not null
trajectory_id uuid null
event_created_at timestamptz not null
classified_at timestamptz not null

classification_schema_version integer not null
routing_policy_version text not null
classifier_source text not null

intent text not null
domain text not null
secondary_domains text[] not null default '{}'
artifact_type text not null
risk text[] not null default '{}'
complexity text not null
recommended_route text not null
response_contract text not null

features jsonb not null
```

Primary key:

```text
(event_id, classification_schema_version, routing_policy_version)
```

Indexes:

```text
(repo, event_created_at desc)
(repo, recommended_route, event_created_at desc)
(repo, intent, domain)
(repo, complexity)
gin(risk)
```

## Enum Requirements

Every enum must include `unknown`.

Unknown input maps to `unknown` and increments a future metric in later phases.
Phase 1 only needs the bounded conversion functions.

## Privacy Allowlist

Unbounded strings are allowed only for:

- `event_id`
- `repo`
- `session_id`
- `routing_policy_version`
- `classifier_source`

`features` is JSONB, but Phase 2 must define a closed feature-key set.

No raw request body, command, path, URL, auth value, or tool output is stored in
this table.

## Required Tests

1. Table migration compiles and applies cleanly.
2. Every enum has an `unknown` variant.
3. SQL check constraints reject values outside the bounded taxonomy.
4. Privacy test rejects unbounded text columns outside the allowlist.
5. Structs compile without importing handler or model-routing internals.

## Stop Condition

Stop when the migration runs cleanly, enums and structs compile, and Phase 1
tests pass. Do not write feature extraction code.
