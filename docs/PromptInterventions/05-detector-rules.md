# Detector Rules

The MVP detector is deterministic and phrase/rule based. It must be versioned.

MVP detector version: `prompt-intervention-detector-v1`.

MVP taxonomy version: `prompt-interventions-v1`.

## Required Rule Mappings

- Text containing “I don't want you to do it” or “do not do it” maps to
  `stop_and_redirect` when it also includes an alternate requested behavior.
  Signal family: `steering`. Burden type: `human_stop_control`. Failure
  relation: `prevention`.
- Text containing “stop” maps to `stop_and_redirect` only when it clearly
  instructs the agent to stop, pause, not continue, or change task direction.
- Text containing “do not implement”, “do not edit”, “do not write”, “do not
  run”, “do not deploy”, or “do not mutate” maps to
  `implementation_block`. Signal family: `steering`. Burden type:
  `human_stop_control`. Failure relation: `prevention`.
- Text containing “review and refine”, “rewrite this prompt”, “make the prompt
  executable”, or “repair this prompt” maps to `prompt_repair`. Signal family:
  `steering`. Burden type: `human_prompt_repair`. Failure relation:
  `quality_control`.
- Text containing “implementation-breaking”, “not ready to hand off”, “major
  issue”, or “blocking issue” maps to `quality_gate`. Signal family:
  `steering`. Burden type: `human_prompt_repair`. Failure relation:
  `quality_control`.
- Text containing “edit only”, “only edit”, “do not read other files”, “do not
  create files”, “limit scope”, or “narrow the scope” maps to
  `scope_narrowing`. Signal family: `steering`. Burden type:
  `human_scope_control`. Failure relation: `prevention`.
- Text containing “acceptance tests required”, “run tests”, “validate”, “dry
  run”, “check before proceeding”, or “verification required” maps to
  `validation_requirement`. Signal family: `validation_pressure`. Burden type:
  `human_validation_control`. Failure relation: `quality_control`.
- Text containing “unsafe”, “production”, “destructive”, “secret”, “rollback”,
  “migration”, “data loss”, or “IAM” maps to `risk_warning` when used as a
  caution, blocker, or safety concern. Signal family: `risk_control`. Burden
  type: `human_risk_control`. Failure relation: `risk_control`.
- Text asking for missing namespace, deployment, file, repo, command output, or
  prior decision context maps to `clarification_request`. Signal family:
  `context_pressure`. Burden type: `context_recovery`. Failure relation:
  `recovery`.
- Text containing “hallucinated”, “wrong file”, “wrong command”, “fake”,
  “malformed”, “scope drift”, or “that is not in the repo” maps to
  `model_failure_correction`. Signal family: `failure_correction`. Burden
  type: `model_error_recovery`. Failure relation: `correction`.

## Confidence Rules

- Exact explicit stop/redirect, implementation block, or scope narrowing:
  confidence at least 0.9.
- Explicit validation requirement or prompt repair: confidence at least 0.85.
- Risk warning or model-failure correction with clear evidence: confidence at
  least 0.8.
- Ambiguous or context-dependent matches: confidence below 0.8.

Headline metrics count only non-superseded labels with confidence at least 0.8.

## Negative Rules

The detector must avoid obvious false positives.

- The word “stop” inside unrelated prose, code, logs, or quoted examples should
  not automatically create `stop_and_redirect`.
- “Run tests” in a generated checklist should not create a new intervention
  unless it is an instruction steering the current model path.
- File paths alone do not create `scope_narrowing`.
- A production keyword alone does not create `risk_warning` unless it is framed
  as risk, caution, validation, or blocking context.
- A prompt that asks the model to implement a feature with normal acceptance
  criteria is not automatically `prompt_repair`.
- A raw tool result containing an error is not automatically
  `model_failure_correction`; it must be framed as correcting prior model
  behavior or steering the next attempt.
