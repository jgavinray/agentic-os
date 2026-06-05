# Orchestration Policy

agentic-os derives an orchestration policy for each classified request. The
policy is the deterministic bridge between "what is this request?" and "what
operational surface should the model or client be allowed to use?"

The policy layer is intentionally not a learned classifier. Request
classification produces bounded labels; orchestration policy turns those labels
into bounded operational constraints. This gives the orchestrator an auditable
control plane for context sources, tool visibility, tool authorization, edit
scope, validation expectations, git behavior, runtime actions, prompt
refinement, and risk handling.

## Why This Exists

Small local models need more explicit operating structure than frontier models.
A frontier coding agent can often infer the right tool, edit scope, and safety
posture from a long prompt. A local harness should not rely on that inference.
It should make the operating envelope explicit before the model sees the tool
menu or attempts a tool call.

The orchestration policy layer provides that envelope:

- It makes intent-to-tool behavior deterministic.
- It keeps high-risk or destructive requests from silently inheriting broad
  tool access.
- It records what policy would have governed a request, even while the system is
  still mostly proxy-compatible.
- It gives the context compiler and telemetry systems a structured explanation
  of why context and tools were selected.
- It creates a durable data set for later classifier or routing improvements
  without changing the runtime model contract first.

## Position In The Pipeline

The live request path is:

```text
client request
  -> authentication and namespace selection
  -> request classification
  -> orchestration policy derivation
  -> tool-menu shaping and request metadata enrichment
  -> context packing
  -> LiteLLM forwarding
  -> response and event persistence
```

For pre-tool hooks, the path is:

```text
client pending tool call
  -> /tools/authorize
  -> bounded tool-authorization classification text
  -> request classification
  -> orchestration policy derivation
  -> policy-aware tool-call authorization
  -> tool_authorization_decision event
  -> agent_orchestration_policies row
```

The policy is derived before the model request is forwarded. It is persisted
after a durable event exists, so the policy row can reference the event id.

## Source Files

| File | Responsibility |
| --- | --- |
| `orchestrator/src/orchestration_policy.rs` | Defines policy enums, derives policies from classifications, builds compact metadata, and persists policy rows. |
| `orchestrator/src/handlers.rs` | Calls policy derivation on request and tool-authorization paths, attaches compact metadata, and persists policy decisions. |
| `orchestrator/src/tool_mediation.rs` | Applies policy to model tool menus and pending tool calls. |
| `orchestrator/migrations/V16__orchestration_policies.sql` | Creates the append-only policy ledger table. |
| `docs/TOOL_MEDIATION.md` | Documents client-facing tool shaping and authorization behavior. |
| `docs/RequestClassification/ARCHITECTURE.md` | Documents the upstream labels consumed by policy derivation. |

## Policy Shape

`OrchestrationPolicy` contains:

- `context_sources`: context systems that may contribute evidence, such as Total
  Recall, Postgres events, Qdrant semantic recall, compiled summaries, the
  context ledger, or raw-capture-derived features.
- `allowed_tools`: tool capabilities visible or executable under the policy.
- `required_tools`: tool capabilities that must be available for the request to
  be answered correctly.
- `blocked_tools`: capabilities that must not be offered or authorized.
- `edit_policy`: whether editing is forbidden, file-limited, scoped, or broader.
- `validation_policy`: what validation posture is expected.
- `git_policy`: whether commits or pushes are allowed or required.
- `runtime_policy`: whether restarts or remote-host operations are allowed.
- `scope_policy`: additive scope invariants such as `no_scp`.
- `prompt_refinement_policy`: whether the request is a prompt/spec review flow.
- `risk_policy`: deterministic risk overlays applied to the request.

Every policy enum is closed, serializable, and has an `unknown` label. `None` is
a semantic variant only where "no policy applies" is a meaningful state, such as
`ValidationPolicy::None` and `PromptRefinementPolicy::None`.

## Derivation Rules

The derivation function is:

```rust
derive_orchestration_policy(classification, request_text, raw_capture_enabled)
```

It is deterministic and has no network or database I/O. It consumes:

- the request intent,
- the artifact type,
- request risk labels,
- the original request text for prompt/spec review detection,
- whether raw capture is enabled.

Intent selects the base policy. Risk overlays are then applied. Finally,
blocked tools are treated as authoritative and removed from both
`allowed_tools` and `required_tools`.

### Intent Defaults

The policy layer handles every current `RequestIntent`:

- `Explain`
- `Debug`
- `Implement`
- `ModifyConfig`
- `GenerateConfig`
- `OperateTool`
- `Plan`
- `Summarize`
- `Classify`
- `Search`
- `Unknown`

The `Unknown` fallback is deliberately minimal: no allowed tools, read-only,
no git changes, no restarts, no validation, and `no_scp`.

`Implement` is intentionally narrower than a general coding-agent mode. It
allows repository read/search/list, file read, file edit/create, and git read.
It blocks generic shell, shell mutation, docker mutation, deployment, service
restart, git write/publishing, and remote host access. The policy carries a
targeted-test validation posture as an expectation, but generic shell validation
is not exposed during the implementation request itself; a later
validation-specific phase should make that tool surface explicit.

### Risk Overlays

Risk overlays are applied after the intent default:

- `ExternalCurrentInfoRequired` adds `web_search` to required tools and records
  `external_web_required`.
- `HighStakes` records `high_stakes_guardrail` and blocks mutation-oriented
  capabilities.
- `DestructiveCommand` records `destructive_requires_confirmation` and blocks
  shell mutation, docker mutation, deployment, and service restart.
- `SecretPresent` records `secret_capture_allowed` only when raw capture is
  enabled.
- Raw capture being enabled records `raw_capture_enabled` and enables
  `raw_capture_features` as a context source.

The policy derives what is allowed, not what is executed. Existing live-policy
guardrails may still stop requests before model forwarding.

### Prompt And Spec Review Overlay

Prompt/spec review is modeled as a policy overlay, not as a separate request
intent. This keeps the request taxonomy stable while still allowing the
orchestrator to treat prompt review as a distinct workflow.

The overlay is triggered when the request text contains a prompt/spec/task word
and a feedback/review/refinement word. When triggered:

- `prompt_refinement_policy` becomes `multi_pass_review`.
- edit policy becomes read-only.
- git policy becomes no git changes.
- runtime policy becomes no restart.
- allowed tools are narrowed to `repo_read`.
- `web_search` is kept only when external current information is required.

This lets a client ask for prompt review without accidentally converting the
review into implementation.

## Tool Mediation Integration

Tool mediation has a local capability taxonomy because it maps client tool names
and shell commands into practical execution categories. The orchestration policy
has a broader capability taxonomy because it describes the operating envelope
for the whole request.

`tool_mediation.rs` bridges the two with a mapping function. Important rules:

- `TextSearch` and `FileList` map to policy `RepoRead`.
- `FileRead` can be allowed by either policy `FileRead` or `RepoRead`.
- `Validation` maps to policy `ShellRead`.
- `Publishing` maps to policy `GitWrite`.
- generic shell maps to policy `ShellRead`.
- unrecognized shell commands map to `ShellMutation`.

The shell default is intentionally conservative. If a command is not recognized
as read, search, or list, it is treated as mutation. That prevents a request
with read-only shell access from running arbitrary shell commands.

Policy enforcement happens in two places:

- Model request shaping removes client-provided tools that are not allowed or
  are explicitly blocked.
- `/tools/authorize` denies pending tool calls that violate the policy.

Menu shaping is subtractive. agentic-os never injects tools that the client did
not advertise, because proxy-mode clients still own tool execution.

## Tool Authorization Classification Text

Tool authorization needs enough request text to classify the attempted action,
but it must not serialize arbitrary tool arguments into the classifier input.

`tool_authorization_classification_text` therefore builds bounded text from:

- non-empty `user_intent`, when present,
- `tool:<tool_name>`,
- string values for the bounded keys `command`, `cmd`, `script`, `query`,
  `path`, and `file_path`.

Unknown argument keys are ignored. This preserves enough signal for destructive
commands such as `rm -rf /tmp/example` while avoiding accidental inclusion of
unbounded or sensitive argument fields.

## Persistence

Every persisted policy row goes into `agent_orchestration_policies`.

The table is append-only:

- no update path,
- no `ON CONFLICT`,
- one row per persisted policy decision.

The table records:

- event and session identity,
- classification schema version,
- routing policy version,
- policy schema version,
- intent and recommended route,
- JSONB arrays for context/tool/scope/risk labels,
- scalar edit/validation/git/runtime/prompt-refinement labels,
- source.

JSONB arrays store `as_str()` labels, for example:

```json
["repo_read", "web_search"]
```

There are no JSONB `CHECK` constraints. Rust owns label validation through the
closed enums and `as_str()` methods.

Compact policy metadata is also attached to request and tool-authorization
events. That event metadata is useful for timeline inspection. The policy table
is the normalized append-only ledger used for policy analytics.

## Relationship To Context Compiler

The context compiler decides what context should be built and injected. The
orchestration policy tells it which context sources are eligible and why.

For example:

- `Explain` uses lightweight Postgres/Qdrant context by default and does not
  automatically pull durable memories or compiled summaries.
- `Debug` can additionally enable the context ledger for failure history.
- `Implement` uses repository-scoped Postgres/Qdrant context while avoiding
  Total Recall, compiled summaries, and stale session artifacts by default.
- raw capture enables raw-capture-derived features as a context source.
- prompt/spec review narrows tools but can still allow repo context.

The policy is not a replacement for retrieval or scoring. It is the control
surface that lets retrieval and context packing operate inside an explicit
request envelope.

## Relationship To Prompt Cache And KV Cache

The orchestration policy does not implement provider prompt caching or runtime
KV caching. Those are separate layers:

- provider prompt cache is affected by stable system/context prefixes,
- vLLM KV/prefix cache is runtime state inside the inference backend,
- orchestration policy records why a request was shaped a certain way and which
  context/tool surfaces were eligible.

Policy data can explain cache behavior, but it is not the cache itself.

## Operational Notes

- The feature is active when request classification and tool mediation are
  enabled in the existing orchestrator flow.
- Tool shaping affects only tools the client included in the request.
- Runtime tool enforcement requires clients to call `/tools/authorize`.
- Policy persistence requires a durable request or tool event. If no session is
  provided for tool authorization, the authorization response can still be
  returned, but there is no session-scoped event to attach a policy row to.
- The current implementation is deterministic rules only. It does not train or
  invoke a classifier model.
- Composite/decomposition labels are classification features only today. They
  are not yet used to schedule subtasks or derive separate per-sub-intent
  policies.

## Next Steps

The next policy layer should consume `sub_intents` from request classification
and derive child envelopes without widening the parent request:

- parent `blocked_tools` and risk overlays remain authoritative,
- each sub-intent gets the smallest tool/context set that can satisfy it,
- implementation sub-intents keep read/search/edit/write tools only,
- validation sub-intents can explicitly expose validation tools,
- publishing, deployment, restart, and destructive operations require explicit
  sub-intents and policy overlays,
- concurrency planning should operate on those child envelopes after dependency
  relationships are represented.

## Testing Coverage

The current tests cover:

- every intent family,
- implementation tool/context narrowing,
- composite/decomposition labels,
- unknown fallback behavior,
- high-stakes and destructive risk overlays,
- external web requirement handling,
- raw-capture policy state,
- prompt/spec review overlay,
- snake_case serialization,
- migration structure,
- compact metadata shape,
- blocked-tools authority,
- OpenAI and Anthropic tool menu shaping,
- `/tools/authorize` policy enforcement,
- shell read versus shell mutation distinction,
- bounded tool-authorization classification text.

The database insert path is exercised structurally by unit tests. Full database
round-trip coverage depends on the existing disposable Postgres test setup used
by the repository's integration-style database tests.
