# Examples

These examples define the expected labels for representative prompt snippets.

## Stop And Redirect

Input prompt excerpt:

> No I don't want you to do it - I am explicitly asking you to develop me a
> prompt and report it here.

Expected output behavior:

- emit one intervention record
- intervention type: `stop_and_redirect`
- signal family: `steering`
- signal strength: `explicit`
- burden type: `human_stop_control`
- failure relation: `prevention`
- source kind: `user_message`
- blocked behavior: implementation
- replacement behavior: develop prompt and report it
- confidence: at least 0.9
- evidence excerpt contains the quoted steering language, redacted if needed

## Prompt Repair

Input prompt excerpt:

> Review and refine the orchestration-policy implementation prompt below. Do
> not implement anything. Your job is to make the prompt executable by a coding
> agent without ambiguity.

Expected output behavior:

- emit at least one `prompt_repair` intervention record
- emit an `implementation_block` record if the detector supports multiple
  labels per exchange
- `prompt_repair` signal family: `steering`
- `prompt_repair` burden type: `human_prompt_repair`
- `prompt_repair` failure relation: `quality_control`
- `implementation_block` signal family: `steering`
- `implementation_block` burden type: `human_stop_control`
- `implementation_block` failure relation: `prevention`
- source kind: `user_message`
- confidence for both explicit labels: at least 0.85

## Scope Narrowing

Input prompt excerpt:

> Edit only /archive/omniscient/cmd/omniscient/sync.go. Do not read other
> files. Do not create files.

Expected output behavior:

- emit one `scope_narrowing` intervention record
- signal family: `steering`
- signal strength: `explicit`
- burden type: `human_scope_control`
- failure relation: `prevention`
- source kind: `user_message`
- target behavior: single-file edit
- blocked behavior: broad repo edit or additional file access
- confidence: at least 0.9
- evidence excerpt must not expose the full prompt

## Validation Requirement

Input prompt excerpt:

> Acceptance tests required: already-published transcripts are skipped; dry-run
> does not publish or mark published; extraction failure records failure.

Expected output behavior:

- emit one `validation_requirement` intervention record
- signal family: `validation_pressure`
- signal strength: `explicit`
- burden type: `human_validation_control`
- failure relation: `quality_control`
- source kind: `user_message`
- target behavior: testable implementation
- confidence: at least 0.85

## Model Failure Correction

Input prompt excerpt:

> That file does not exist in the repo. You hallucinated the path. Use the
> actual migration file and do not invent repo structure.

Expected output behavior:

- emit one `model_failure_correction` intervention record
- may also emit `scope_narrowing` if the text constrains future file access
- `model_failure_correction` signal family: `failure_correction`
- `model_failure_correction` burden type: `model_error_recovery`
- `model_failure_correction` failure relation: `correction`
- confidence: at least 0.8

## Missing Context Recovery

Input prompt excerpt:

> What namespace and deployment are you targeting? I cannot give production
> rollout commands without those identifiers and the current failure symptoms.

Expected output behavior:

- emit one `clarification_request` intervention record
- signal family: `context_pressure`
- signal strength: `explicit`
- burden type: `context_recovery`
- failure relation: `recovery`
- source kind: `user_message` or `assistant_message`, depending on where the
  text appeared in captured messages
- confidence: at least 0.8

## Negative Example

Input prompt excerpt:

> The service failed because the stop channel was closed during shutdown.

Expected output behavior:

- do not emit `stop_and_redirect`
- no intervention record unless another explicit steering phrase is present
