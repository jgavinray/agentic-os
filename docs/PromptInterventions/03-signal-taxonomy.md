# Signal Taxonomy

The implementation must generate explicit telemetry signals. A signal is a
bounded classification of evidence found in captured prompt/message history.

Every signal must answer:

- what happened?
- where is the immutable evidence?
- what kind of burden or failure does it represent?
- how confident is the detector?

## Signal Dimensions

Every intervention record must include:

- `signal_family`
- `signal_type`
- `signal_strength`
- `burden_type`
- `failure_relation`

## Signal Families

The MVP signal families are closed and bounded.

- `steering`: the prompt redirects, narrows, blocks, or repairs model behavior.
- `failure_correction`: the prompt corrects a model or harness mistake.
- `risk_control`: the prompt introduces a safety, production, data, secret, or
  destructive-action constraint.
- `validation_pressure`: the prompt requires tests, dry-runs, checks, or other
  verification before acceptance.
- `context_pressure`: the prompt asks for missing context, prior decisions,
  identifiers, files, logs, namespace, deployment, or repository facts.
- `no_signal`: no intervention signal was detected.

## Intervention Types

The MVP intervention taxonomy is closed and bounded.

- `stop_and_redirect`: blocks the current action and replaces it with another
  task.
- `scope_narrowing`: reduces files, tools, task breadth, context breadth, or
  allowed behavior.
- `prompt_repair`: rewrites or repairs a prompt/spec before handoff.
- `quality_gate`: blocks handoff until a concrete issue is fixed.
- `risk_warning`: identifies safety, production, data, secret, destructive, or
  rollback risk.
- `clarification_request`: asks for missing context required to proceed.
- `implementation_block`: explicitly says not to implement, edit, write, run,
  deploy, or mutate state.
- `validation_requirement`: adds tests, commands, dry-runs, checks, or
  acceptance gates required before completion.
- `model_failure_correction`: corrects hallucination, wrong files, wrong
  commands, malformed output, or scope drift.
- `other`: bounded fallback for early backfill; must be rare and should not be
  used for clear cases covered by the other labels.

## Family Mapping

Each intervention type maps to one primary signal family:

- `stop_and_redirect` maps to `steering`
- `scope_narrowing` maps to `steering`
- `prompt_repair` maps to `steering`
- `quality_gate` maps to `steering`
- `implementation_block` maps to `steering`
- `model_failure_correction` maps to `failure_correction`
- `risk_warning` maps to `risk_control`
- `validation_requirement` maps to `validation_pressure`
- `clarification_request` maps to `context_pressure`
- `other` maps to the closest family only when bounded evidence exists;
  otherwise no record should be emitted

## Signal Strength

The MVP signal-strength taxonomy is closed and bounded.

- `explicit`: the prompt directly states the steering or failure correction.
- `implicit`: the prompt implies steering from constrained instructions.
- `ambiguous`: there is a weak match, and the label should not count in
  headline metrics unless confidence is high enough.

## Burden Types

The MVP burden taxonomy is closed and bounded.

- `human_prompt_repair`: a user or prior assistant rewrote the prompt so the
  next model attempt could succeed.
- `human_scope_control`: a user or prior assistant narrowed files, tools,
  behavior, or task breadth.
- `human_stop_control`: a user or prior assistant stopped or redirected an
  undesired path.
- `human_validation_control`: a user or prior assistant added required
  verification gates.
- `human_risk_control`: a user or prior assistant added safety or operational
  constraints.
- `context_recovery`: the prompt requests missing context needed before safe
  progress.
- `model_error_recovery`: the prompt corrects hallucination, wrong file,
  malformed output, or scope drift.
- `unknown_burden`: used only when a valid signal exists but the burden cannot
  be classified more specifically.

## Failure Relations

The MVP failure-relation taxonomy is closed and bounded.

- `prevention`: steering was applied before a bad action occurred.
- `correction`: steering corrected an observed bad or inadequate action.
- `recovery`: steering helped resume after failure or missing context.
- `quality_control`: steering raised the quality bar before handoff.
- `risk_control`: steering reduced safety or operational risk.
- `unknown_relation`: used only when the relationship is unclear.

## Authorship Limits

Available telemetry does not guarantee whether steering text was written by
ChatGPT, a human, or another assistant.

Correct signal wording:

- `external steering was present`
- `prompt contained implementation block`
- `prompt corrected a hallucinated file claim`
- `prompt required validation before acceptance`

Incorrect signal wording without provenance:

- `ChatGPT corrected the model`
- `human rejected the model`
- `frontier model steered the local model`
