# Tool Mediation

agentic-os mediates tool use deterministically. The current implementation works in proxy mode: clients still provide and execute tools, while the orchestrator shapes the tool menu before model dispatch and exposes a pre-execution authorization endpoint for client hooks.

This is a bridge toward full agent orchestration. It does not add learned routing, scoring, autonomous retries, reinforcement learning, or an agent-owned tool loop.

## Model

The policy split is:

- policy rules decide which capability should satisfy an intent,
- the orchestrator enforces those rules by shaping model requests, authorizing pending calls, recording decisions, and emitting bounded metrics.

Tools are mapped into bounded capabilities:

- `file_read`
- `text_search`
- `file_list`
- `file_edit`
- `validation`
- `publishing`
- `shell`
- `unknown`

The first rule family is canonical-tool preference: when a narrower canonical tool exists, shell fallbacks are hidden or denied for that capability.

## Orchestrator Configuration

Tool mediation is enabled on the orchestrator with:

```bash
TOOL_MEDIATION_ENABLED=true
```

This is the default. The endpoint is mounted by the orchestrator at `POST /tools/authorize`; there is no separate service to run.

The endpoint uses the same authentication model as other non-health endpoints:

```http
Authorization: Bearer <orchestrator-api-key>
```

With the flag enabled, two paths are active:

- Proxy shaping: clients that send model traffic through `/v1/chat/completions` or `/v1/messages` get automatic, subtractive tool-menu shaping before the request is forwarded to LiteLLM.
- Tool authorization: clients that support pre-tool hooks can call `/tools/authorize` before executing a pending tool call.

Proxy shaping alone changes what the model can see in that request. Runtime enforcement requires the client or adapter to call `/tools/authorize` before execution.

## Proxy Menu Shaping

When `TOOL_MEDIATION_ENABLED=true`, `/v1/chat/completions` and `/v1/messages` inspect the incoming client-provided `tools` array before forwarding to LiteLLM.

For file-read intent:

```text
offered tools: Read, Bash
decision: shape
allowed tools: Read
hidden tools: Bash
reason: prefer_canonical_tool
```

The same shaping path supports OpenAI-style tools and Anthropic-style tools. If no canonical tool is available, shell is left visible because hiding it would remove the only available capability.

Menu shaping is subtractive. agentic-os does not inject tools the client did not advertise, because the client may not be able to execute them.

## Tool Authorization

`POST /tools/authorize` lets a client ask agentic-os before executing a pending tool call. This is the enforcement path for clients with pre-tool hooks, such as Claude Code `PreToolUse`.

Minimum request fields:

- `tool_name`
- `arguments`
- `available_tools`

Recommended lineage fields:

- `session_id`, so the decision is persisted as an event
- `repo` and `task`, so the event lands in the intended namespace
- `trajectory_id`, `attempt_index`, and `parent_event_id`, when the client already has trajectory context

Request:

```json
{
  "session_id": "session-1",
  "repo": "agentic-os",
  "task": "engineering",
  "trajectory_id": "00000000-0000-0000-0000-000000000000",
  "attempt_index": 1,
  "tool_name": "Bash",
  "arguments": {
    "command": "cat README.md"
  },
  "available_tools": ["Read", "Bash"]
}
```

Response:

```json
{
  "policy_version": "deterministic-v1",
  "decision": "deny",
  "reason": "prefer_canonical_tool",
  "capability": "file_read",
  "attempted_tool": "Bash",
  "preferred_tool": "Read",
  "replacement": {
    "tool_name": "Read",
    "arguments": {
      "path": "README.md"
    }
  },
  "message": "Use the Read tool for file_read instead of shell fallback commands."
}
```

If no canonical replacement is available, the call is allowed.

An allow response means the client should continue with the original tool call. A deny response means the client should block the original call and surface `message` plus any `replacement` guidance back into its tool loop. The orchestrator does not post-process the client state in proxy mode.

Example with curl:

```bash
curl -sS http://localhost:8088/tools/authorize \
  -H 'Authorization: Bearer sk-local-orchestrator' \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "session-1",
    "repo": "agentic-os",
    "task": "engineering",
    "tool_name": "Bash",
    "arguments": {"command": "cat README.md"},
    "available_tools": ["Read", "Bash"]
  }'
```

Expected response:

```json
{
  "policy_version": "deterministic-v1",
  "decision": "deny",
  "reason": "prefer_canonical_tool",
  "capability": "file_read",
  "attempted_tool": "Bash",
  "preferred_tool": "Read",
  "replacement": {
    "tool_name": "Read",
    "arguments": {
      "path": "README.md"
    }
  },
  "message": "Use the Read tool for file_read instead of shell fallback commands."
}
```

## Client Hook Adapters

Clients usually do not speak the `/tools/authorize` schema directly. A thin adapter should translate the client's hook payload into the generic authorizer request and translate the authorizer response back into the client's hook result format.

The adapter should:

- collect the attempted tool name and arguments,
- include the complete set of tools the client can execute,
- call `/tools/authorize` with the orchestrator bearer token,
- allow the original tool call when `decision=allow`,
- deny the original tool call when `decision=deny`,
- return the denial message and replacement guidance in whatever format the client exposes to the agent.

This keeps client-specific hook formats outside the core orchestrator while preserving one deterministic authorization policy.

## Trajectory Recording

Menu shaping metadata is attached to request and model-response metadata for captured trajectories. Authorization decisions are recorded as `tool_authorization_decision` events when the client supplies `session_id`. If trajectory fields are supplied, the event uses:

- `trajectory_id`
- `attempt_index`
- `event_role=tool_call`

This keeps decisions queryable without introducing a new storage backend or changing the trajectory event-role enum.

## Metrics

Tool mediation metrics are bounded:

- `tool_mediation_decisions_total{decision,reason}`
- `tool_mediation_tools_total{action,capability}`

Allowed `decision` values are `allow`, `deny`, `shape`, `pass`, and `unknown`.

Allowed `reason` values are `prefer_canonical_tool`, `no_tools`, `not_applicable`, `disabled`, and `unknown`.

Allowed `action` values are `offered`, `allowed`, `hidden`, `denied`, and `unknown`.

Allowed `capability` values are the bounded capability enum above.

Metric labels never include tool names, commands, filenames, trajectory IDs, or user-controlled text.

## Feature Flag

`TOOL_MEDIATION_ENABLED=true` by default.

When false:

- model request tool menus are not shaped,
- `/tools/authorize` returns allow with `reason=disabled`,
- authorization decisions are not persisted as events,
- tool mediation metrics remain at zero except explicit disabled authorization calls.

## Future Modes

The current mode is proxy-compatible. Future modes can reuse the same primitives:

- broker mode, where clients route all pending tool calls through agentic-os before execution,
- orchestrator-owned tools, where agentic-os executes or delegates tools itself,
- full agent loop ownership, where model call, tool call, validation, remediation, and finalization all occur inside one active trajectory.

Those modes require explicit client/runtime integration. The current implementation does not claim to stop tools executed entirely outside agentic-os.
