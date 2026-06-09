# Tool Mediation Narrows Claude CLI Implementer To Read-Only

Date observed: 2026-06-08

## Summary

When using Claude CLI as the implementer behind the orchestrator, the client
offers implementation-capable tools, but the orchestrator forwards only `Read`
to the routed vLLM model. This makes the model appear to be operating in an
implementation workflow while the actual forwarded tool menu is read-only.

This is not a case where the orchestrator ignores tool mediation. The evidence
shows the orchestrator is actively mediating the tool list. The problem is that
the current policy/tool-shaping path removes the tools required for the Claude
CLI implementer workflow.

## Expected Behavior

For an implementation task driven through Claude CLI:

- Claude CLI may expose a client-side local tool inventory.
- The orchestrator should remain the enforcement point for what the model sees.
- The forwarded model tool menu should match the derived task intent and risk.
- Implementation tasks should allow a bounded implementation-capable tool set
  when the task is explicitly scoped for implementation.
- Validation-capable tools should be available when the task requires tests or
  build checks, subject to policy.
- The audit trail should clearly record offered tools, forwarded tools, hidden
  tools, and the reason each tool was hidden.

For a safe implementation workflow, a reasonable forwarded menu would usually
include read and edit capabilities, with shell mutation, deploy, restart, git
write, and remote access still blocked unless explicitly authorized.

## Actual Behavior

The latest captured Anthropic messages show Claude CLI offering three tools:

- `Bash`
- `Edit`
- `Read`

The orchestrator forwarded only one tool:

- `Read`

Observed raw/forwarded comparison:

```text
exchange_id                              received_at                  raw_tools  fwd_tools  raw_tool_names   fwd_tool_names
293edf61-015a-4280-927d-6f241fa8d45b     2026-06-08 13:12:00 UTC      3          1          Bash, Edit, Read  Read
9b20fd2f-452b-4cc6-892c-ebab9b1a4dad     2026-06-08 13:11:42 UTC      3          1          Bash, Edit, Read  Read
69b73efa-d667-4c66-a25f-57cb35f01018     2026-06-08 13:11:38 UTC      3          1          Bash, Edit, Read  Read
ba416fcb-196b-47f8-bd36-7e33dd8a3b0c     2026-06-08 13:11:32 UTC      3          1          Bash, Edit, Read  Read
5f5318d3-f320-4fc7-92ff-11a9b9bd1262     2026-06-08 13:11:24 UTC      3          1          Bash, Edit, Read  Read
```

The forwarded request metadata still describes a broad engineering path:

```text
task: engineering
requested_model: claude-opus-4-8
routed_model: qwen3.6-27b
selected_route: claude_opus_tier
policy_version: claude-tier-routing-v1
baseline_arm: orchestrator_policy_enabled
reasoning_policy: medium
cache_policy_enabled: false
cache_bypass_reason: streaming
```

The metadata does not make the tool mediation result obvious. The raw and
forwarded request bodies prove the behavior, but the forwarded metadata does
not include a compact explanation such as:

- offered tools
- allowed tools
- hidden tools
- hidden capabilities
- policy reason
- canonical-tool reason
- request classification used by the policy

## Why This Matters

This creates a mismatch between the intended workflow and the model execution
surface.

The active workflow is:

1. ChatGPT acts as the orchestrator.
2. Claude CLI acts as the implementer.
3. The orchestrator routes Claude CLI requests through local/open-weight models.
4. The implementer is expected to read files, edit files, and run bounded
   validation when instructed.

With only `Read` forwarded to vLLM, the routed model cannot actually perform
the implementation role. It can inspect state, but it cannot edit or validate
through the model-visible tool protocol.

This can produce misleading symptoms:

- The user believes the harness is being exercised as an implementation system.
- The model appears unproductive or hesitant.
- The model may generate prose instead of edits because it cannot see edit
  tools.
- The model may ask for tools that were present client-side but hidden
  upstream.
- Implementation progress becomes dependent on accidental behavior outside the
  intended orchestrator-mediated path.
- Evaluation data becomes contaminated because the run is not really testing
  an implementation-capable local model path.

## Evidence Trail

The evidence was reconstructed from `raw_http_exchanges` in the capture
database by comparing `raw_request_body` and `forwarded_request_body` for recent
`messages` endpoint requests.

The relevant behavior is visible only by comparing raw and forwarded payloads.
The ordinary runtime logs observed during this investigation showed routing and
token clamping, but did not clearly emit the tool shaping result.

The latest rows had:

- raw request tool count: `3`
- forwarded request tool count: `1`
- raw tool names: `Bash, Edit, Read`
- forwarded tool names: `Read`

The same pattern repeated across multiple adjacent exchanges, so this was not a
single transient request.

## Relevant Code Path

The Anthropic messages handler prepares an Anthropic-formatted request, derives
request policy, and then applies policy-aware tool shaping before forwarding:

```text
handlers_anthropic_messages.rs
  prepare_anthropic_litellm_request(...)
  classify_and_derive_request_policy(...)
  shape_anthropic_request_with_policy(...)
```

The tool-shaping path:

```text
tool_mediation_shaping.rs
  shape_anthropic_request_with_policy(...)
  shape_request(...)

tool_mediation_shaping_policy.rs
  policy_shaping_for_tools(...)
  policy_hides_tool(...)

tool_mediation_policy.rs
  policy_allows_tool_capability(...)
  policy_blocks_tool_capability(...)
```

The implementation policy allows implementation capabilities at the policy
level:

```text
orchestration_policy_intents_mutating.rs
  implement_policy()
    allowed:
      RepoRead
      FileRead
      FileEdit
      GitRead
    blocked:
      ShellMutation
      ShellRead
      DockerMutation
      Deploy
      RestartService
      GitWrite
      RemoteHostAccess
```

However, the observed forwarded request did not include `Edit`, even though
`FileEdit` is allowed by the implementation policy. That means either:

- the request was not classified into the implementation policy shape,
- the tool capability mapping classified the client `Edit` tool in a way the
  policy hid,
- a risk overlay removed edit capability,
- canonical tool shaping hid more than intended,
- or the metadata task label `engineering` is too coarse to diagnose the actual
  policy path.

The current audit trail does not make this determination directly visible.

## Suspected Failure Modes

### 1. Coarse Intent Label

The forwarded metadata shows `task: engineering`, which is too broad to reason
about tool authorization. It does not say whether the request was classified as
implementation, debugging, prompt repair, documentation, validation, or
something else.

If the derived classification was not `implement`, then hiding `Edit` may be
obedient behavior, but the system still failed to align the task classification
with the requested implementer workflow.

### 2. Tool Capability Mapping Mismatch

The client tool names are Claude CLI tool names, not generic orchestrator
capability names. The policy layer reasons over capabilities such as:

- `FileRead`
- `FileEdit`
- `ShellRead`
- `ShellMutation`
- `GitWrite`

If the `Edit` tool is not mapped cleanly to `FileEdit`, or if the Anthropic tool
payload shape causes name/capability extraction to fail, the policy may hide it.

### 3. Shell Tool Overloading

`Bash` is currently treated as shell capability. In the implementation workflow,
`Bash` can represent several very different operations:

- safe read-only inspection
- test execution
- build validation
- file mutation through shell redirection
- destructive commands
- deploy/restart operations

The current policy blocks shell for implementation, which is reasonable for
mutation safety, but it also prevents validation commands from being available
through the same client tool. This creates a gap for bounded validation.

The system needs a more precise distinction between:

- shell read
- shell validation
- shell mutation
- destructive shell
- deploy/restart

Without that distinction, implementation tasks either become too permissive or
too weak to validate.

### 4. Missing Tool-Mediation Audit Event

The raw/forwarded exchange bodies contain the proof, but there is no obvious
first-class event row that says:

```text
offered_tools = [Bash, Edit, Read]
forwarded_tools = [Read]
hidden_tools = [Bash, Edit]
reason = policy_filtered
policy_intent = ?
risk_overlays = ?
```

This makes diagnosis slow and makes trend analysis harder.

## Impact On Evaluation

This materially affects the current model-exercise goal.

If the local model path is being evaluated as an implementer while it only sees
`Read`, then the resulting traces cannot fairly answer whether the local model
can perform implementation work under orchestration. They instead measure a
read-only constrained model being asked to participate in an implementation
workflow.

That can distort labels such as:

- model capability failure
- prompt failure
- routing failure
- tool failure
- classifier failure
- human steering required
- accepted output
- retry count

The right label for these runs may be closer to:

```text
tool_policy_mismatch
implementation_tool_underexposure
classification_or_policy_failure
```

depending on the final root cause.

## Recommended Fix Direction

Do not simply bypass tool mediation. The orchestrator should remain the
enforcement point.

Recommended next steps:

1. Persist a structured tool-mediation decision event for every request with a
   tool menu.
2. Include request classification, policy version, risk overlays, offered
   tools, allowed tools, hidden tools, and hide reasons.
3. Add tests for Anthropic Claude CLI tool payloads specifically:
   - `Read` maps to file/repo read.
   - `Edit` maps to file edit.
   - `Bash` does not collapse all shell use into one undifferentiated
     capability.
4. Add an implementation-intent fixture where `Read` and `Edit` survive, while
   deploy/restart/git-write/destructive shell remain hidden.
5. Add a validation-intent fixture that allows bounded validation without
   allowing destructive shell mutation.
6. Make the forwarded metadata or associated event expose the tool decision in
   bounded form.
7. Re-run the Claude CLI implementer harness and confirm the raw/forwarded
   request comparison shows an implementation-capable but still governed tool
   menu.

## Acceptance Criteria For Fix

This bug is fixed when all of the following are true:

- An explicit implementation task through Claude CLI forwards `Read` and `Edit`
  to the routed model.
- The same task does not forward deploy, restart, remote host access, git write,
  or destructive shell capabilities unless explicitly authorized.
- Validation tasks can run bounded validation without granting broad destructive
  shell authority.
- The capture database contains a structured, queryable tool-mediation event or
  equivalent bounded metadata for every shaped request.
- The event records the classification/policy basis for the decision.
- Tests cover Anthropic Claude CLI tool payloads, not only generic OpenAI-style
  tool payloads.
- Evaluation labels can distinguish model failure from tool-policy mismatch.

## Open Questions

- What exact request classification was derived for the observed exchanges?
- Was `Edit` hidden because the intent was not classified as implementation, or
  because the Anthropic `Edit` tool payload failed capability mapping?
- Should bounded validation be represented as a separate capability from
  `ShellRead`?
- Should Claude CLI harness runs include an explicit task/intention metadata
  override instead of relying on broad `engineering` defaults?
- Should tool mediation outcomes be stored in the same raw capture database or
  in a derived telemetry table linked by raw exchange ID?

