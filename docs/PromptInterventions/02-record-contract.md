# Record Contract

For each detected intervention, the system must emit one append-only derived
record.

## Required Fields

Each intervention record must include:

- unique intervention record ID
- exact raw exchange ID that produced the evidence
- optional trajectory ID
- optional request event ID
- optional attempt ID
- optional requested model
- optional routed model
- optional baseline arm
- optional selected route
- optional routing policy version
- exact prompt hash
- normalized prompt hash
- prompt fingerprint version
- source kind
- intervention type
- signal family
- signal type
- signal strength
- burden type
- failure relation
- optional target behavior
- optional blocked behavior
- optional replacement behavior
- bounded evidence excerpt
- evidence hash
- labeler type
- confidence
- taxonomy version
- optional superseded-record pointer
- creation timestamp

## Semantics

The raw exchange ID means:

> This intervention label was derived from this exact captured request/response
> exchange.

The exact prompt hash means:

> This exact extracted prompt/message text appeared here.

The normalized prompt hash means:

> This prompt has the same normalized shape as other prompts after deterministic
> removal of volatile values.

The evidence excerpt means:

> This short redacted excerpt explains why the intervention was labeled.

The signal fields mean:

> This is the measurable type of steering, failure correction, risk control,
> validation pressure, or context pressure represented by the evidence.

The intervention record is a derived interpretation. The raw exchange remains
the source of truth.

## Source Kinds

The MVP source-kind taxonomy is closed and bounded.

- `raw_prompt`: evidence came from the whole extracted raw prompt body.
- `user_message`: evidence came from a user-role message.
- `assistant_message`: evidence came from an assistant-role message present in
  the request history.
- `tool_result`: evidence came from a tool-result message or equivalent tool
  output message.
- `posthoc_review`: evidence was added by a later operator or review pass.

The implementation must not infer author identity from source kind.

## Labeler Types

The MVP labeler-type taxonomy is closed and bounded.

- `rule`: deterministic detector produced the label.
- `posthoc`: operator or backfill review produced the label.
- `human`: human-facing correction workflow produced the label.
- `local_model`: reserved for later offline experiments.
- `frontier_model`: reserved for later offline experiments.

Runtime MVP labeling must use deterministic rules only. `local_model` and
`frontier_model` are allowed values only so future offline labeling can be
versioned without changing the taxonomy.

## Superseding

Intervention records are append-only. They are never updated or deleted.

Corrections are represented by writing a new record that points at the older
record it supersedes.

Default reports and headline metrics must exclude records superseded by a newer
record. Audit views must still allow operators to inspect both the original and
replacement records.
