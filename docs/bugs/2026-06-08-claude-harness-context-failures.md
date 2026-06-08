# Claude Harness Context Failure Modes

Date observed: 2026-06-08

This note captures failure modes observed while implementing
`docs/TELEMETRY_PROMPT_INTERVENTIONS.md` through the Claude CLI harness routed
through the orchestrator.

The purpose is to preserve evidence for later remediation. This is not a fix
plan and does not mark the prompt-intervention implementation complete.

## Summary

The Claude CLI harness repeatedly produced large multi-turn tool transcripts
that exceeded the routed model context window. The orchestrator had enough raw
capture and ledger data to reconstruct most of the failure, but some important
upstream retry details were only visible in logs.

The key failure pattern was:

- Claude CLI requested `claude-opus-4-8`.
- The orchestrator routed the request to `qwen3.6-27b`.
- Claude CLI requested `max_tokens=64000`.
- The orchestrator clamped output tokens to `32768`.
- LiteLLM/vLLM rejected the request because input tokens plus requested output
  tokens exceeded the `65536` context window.
- The orchestrator retried with reduced output tokens, but the retry still
  exceeded the context window.

## Evidence

Representative raw exchange:

- raw exchange ID: `2dba6bee-a02b-41d6-b520-e2ac2c56166e`
- endpoint: `messages`
- namespace: `agent-os-clean-002`
- requested model: `claude-opus-4-8`
- routed model: `qwen3.6-27b`
- raw `max_tokens`: `64000`
- forwarded `max_tokens`: `32768`
- message count: `37`
- raw request JSON size: about `127306` chars
- forwarded request JSON size: about `126274` chars

LiteLLM reported these context-window failures:

- first attempt: `32769` input tokens plus `32768` output tokens equals
  `65537`, exceeding the `65536` limit
- retry attempt: `33026` input tokens plus `32511` output tokens equals
  `65537`, still exceeding the `65536` limit

The largest contributors in the captured request were:

- tool-result blocks: about `71956` chars
- assistant-message content: about `41401` chars
- tool-use payloads: about `28054` chars
- ordinary text blocks: about `12938` chars

## Failure Modes

### 1. Output Budget Is Clamped Without Input-Aware Budgeting

The orchestrator clamps `max_tokens` from `64000` to `32768`, but that clamp is
not sufficient when the input is already near half of the model context window.

Impact:

- A request can pass local max-token validation and still be rejected upstream.
- The failure depends on total prompt size, not only requested output size.

Needed follow-up:

- Compute or estimate input budget before forwarding.
- Cap output tokens from `context_window - input_tokens - safety_margin`.
- Use a larger safety margin for models that report prompt tokens as "at
  least".

### 2. Context-Window Retry Has Too Little Safety Margin

The retry calculation reduced output tokens after the first upstream failure,
but the second request still exceeded the context window by one token.

Observed arithmetic:

- first error reported `32769` input tokens
- retry selected `32511` output tokens
- second error reported `33026` input tokens
- `33026 + 32511 = 65537`

Likely contributing factor:

- the retry path mutates the request by adding fresh `agentic_os` metadata,
  including a new attempt ID, which can alter tokenization and input token
  counts between attempts

Needed follow-up:

- Recompute the budget using the retried request shape, or use a much larger
  retry margin.
- Avoid adding bulky or changing metadata to retried model-visible payloads
  when possible.
- Persist retry-budget inputs and outputs for audit.

### 3. Claude CLI Tool Transcript Bloat

The implementer prompt was narrow, but Claude carried a full tool transcript
inside the request:

- file reads
- source listings
- test output
- tool errors
- patch attempts
- assistant reasoning/debug text

This caused the request to grow over repeated tool turns. A small repair task
became a large prompt because every intermediate tool interaction remained in
the request history.

Needed follow-up:

- Bound tool-result content before it is sent back through the model path.
- Prefer targeted excerpts over full file reads in weak/local routes.
- Consider turn limits for local implementer sessions.
- Add a preflight check that detects when tool transcript growth approaches
  the context window.

### 4. Tool Schema and Tool Availability Drift

The routed model attempted several invalid or unavailable tools:

- `exec_bash`, which was unavailable
- `Bash` with `cmd` instead of the required `command` field
- `Write`, which was not enabled in that context

The model then used shell commands to write a temporary patch script. This is a
model/tool-use reliability signal, not an implementation success signal.

Needed follow-up:

- Capture denied, malformed, and unavailable tool attempts as first-class
  telemetry.
- Include tool schema failure rates in routing decisions.
- Prefer smaller prompts or stronger models for tasks requiring precise tool
  use.
- Make tool availability explicit in prompts and verify the model obeys it.

### 5. Implementer Claims Are Not Verification

One Claude run returned that the edit was done and that the test "should" pass,
but local validation still failed:

- `cargo test --manifest-path orchestrator/Cargo.toml prompt_intervention_extract`
  failed `normalization_replaces_timestamp`

This reinforces the rule that model output is an allegation until verified.

Needed follow-up:

- Treat model-reported test success as unverified unless command output is
  captured and checked independently.
- Require checkpoint commits to be gated by local validation run by the
  orchestrator.

### 6. Runtime Source Drift

The running orchestrator container logged:

`failed to assemble prompt intervention records: evidence_excerpt contains unredacted secrets`

That behavior was not present in the current checked-out source at the logged
line after the revert. This indicates the deployed container was stale relative
to the current `refactor` worktree.

Impact:

- Runtime telemetry may reflect code that is no longer present locally.
- Raw captures from this window can be misleading if interpreted as evidence
  for the current source tree.

Needed follow-up:

- Include build SHA or source revision in orchestrator logs and health output.
- Record service build identity alongside raw captures.
- Rebuild/redeploy after revert commits before using runtime behavior as
  evidence.

### 7. Raw Capture and Ledger Gaps

The raw capture database stored client request/response exchanges, and the
LiteLLM call ledger stored upstream attempts, but the most useful context-window
details were not fully normalized.

Observed:

- `raw_http_exchanges` captured the client-visible request and response.
- `litellm_call_ledger` captured failed attempts as `http_error` / `4xx`.
- `litellm_call_ledger.error_message` only stored
  `upstream returned non-success status`.
- Exact context-window math was visible in LiteLLM/orchestrator logs, not in a
  structured ledger field.
- `litellm_callback_payloads` had no rows for the inspected window.

Needed follow-up:

- Persist upstream error bodies or parsed error details for failed attempts.
- Add structured fields for context limit, input tokens, requested output
  tokens, retry output tokens, and retry reason.
- Link raw exchange IDs to upstream attempt IDs where possible.

## Current Implementation State

At the time this note was written, the worktree had an uncommitted partial
extraction checkpoint:

- `orchestrator/src/lib.rs` modified
- `orchestrator/src/prompt_intervention_extract.rs` untracked

The focused extraction test was not passing, so this checkpoint was not ready
to commit.

## Suggested Remediation Themes

- Add input-aware output budgeting before LiteLLM forwarding.
- Increase retry safety margin and record retry-token calculations.
- Bound or summarize tool-result history in Claude CLI implementer runs.
- Capture malformed and unavailable tool attempts as model-behavior telemetry.
- Add runtime build identity to logs, health, and raw captures.
- Normalize context-window failure details into the LiteLLM call ledger.
- Continue requiring independent validation before accepting model-produced
  implementation work.
