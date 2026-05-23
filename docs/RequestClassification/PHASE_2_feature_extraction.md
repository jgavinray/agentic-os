# Phase 2: Deterministic Feature Extraction

## Goal

Extract deterministic request-level features and labels from `agent_events`
without writing database rows or changing routing behavior.

## Scope

Implement a pure module entry point:

```text
classify_request_event(event) -> RequestClassification
```

The function consumes already-loaded event data. It does not query Postgres,
Qdrant, LiteLLM, the filesystem, or the network.

## Input

Use:

- `agent_events.event_type`
- `agent_events.summary`
- `agent_events.evidence`
- safe structured values from `agent_events.metadata`
- event lineage identifiers

Do not inspect raw files or external systems.

## Feature Keys

The feature object must use a closed set of keys:

```text
char_count
line_count
estimated_tokens
has_code_block
has_yaml
has_json
has_stack_trace
has_logs
has_shell_command
has_url
has_file_path
has_secret_candidate
contains_error_words
contains_destructive_verbs
asks_for_latest
asks_for_file_generation
detected_domain_terms
has_kubernetes_terms
has_docker_terms
has_llm_terms
has_networking_terms
has_security_terms
has_config_shape
has_diff_or_patch
has_test_failure
has_auth_header_shape
```

Counts are integers. Flags are booleans. Term lists are bounded arrays of known
terms or labels.

## Label Rules

Labels are deterministic rule outputs:

- Intent from action verbs and request shape.
- Domain from bounded domain term dictionaries.
- Artifact type from syntax and metadata shape.
- Risk from high-confidence keyword and structure checks.
- Complexity from feature counts, risk, and tool requirement.
- Recommended route from the routing policy version.
- Response contract from intent, route, and artifact type.

## Route Policy v1

Suggested deterministic policy:

- `refuse_or_guardrail` for `unsafe_security` or high-risk destructive command.
- `web_required` for `external_current_info_required`.
- `ask_clarification` for operate/modify requests missing target context.
- `tool_required` for explicit tool operation.
- `deterministic_template` for L0 known templates.
- `small_local_model` for L1/L2 low-risk explain/summarize/classify.
- `strong_local_model` for L3 complex but not high-risk.
- `unknown` only when no rule applies.

Phase 2 only computes this recommendation. It does not enforce it.

## Required Tests

1. Feature extraction is deterministic for identical input.
2. Feature keys are closed and stable.
3. Secret candidates set `secret_present`.
4. Destructive commands set `destructive_command`.
5. Latest/current phrasing sets `external_current_info_required`.
6. Medical/legal/finance requests set `high_stakes`.
7. Kubernetes/Docker/LLM/networking/security terms map to the expected domains.
8. Unknown or empty events produce bounded `unknown`/safe defaults.
9. No raw text is copied into the features object.

## Stop Condition

Stop when pure feature extraction and label tests pass. Do not write database
backfill, boot integration, CLI, metrics, or route enforcement.
