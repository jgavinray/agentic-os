# Extraction And Fingerprints

The system must extract prompt/message text deterministically from captured
request payloads and compute repeatable fingerprints.

## Extraction Order

The detector must inspect sources in this order:

1. latest user-role message
2. assistant-role messages present in request history
3. tool-result messages present in request history
4. full raw request body only when structured message extraction fails

## Message Content Handling

- Plain string content is extracted as-is.
- Array content is scanned in order.
- Text blocks contribute their text.
- Tool-result blocks contribute bounded tool-result text and are marked as
  tool-result source candidates.
- Non-text blocks are ignored unless they contain explicit tool-result text.
- Missing or malformed message arrays fall back to raw-prompt extraction.

The detector may emit multiple intervention records for one exchange when the
same prompt contains distinct interventions, such as both `scope_narrowing` and
`validation_requirement`.

## Prompt Hashes

The system must compute two hashes.

Exact prompt hash:

- hash of the exact extracted prompt/message text used by the detector
- changes when any character in the extracted text changes
- used for exact-repeat analysis

Normalized prompt hash:

- hash of deterministic normalized text
- used for prompt-family trend analysis
- never replaces the exact prompt hash for audit

MVP normalization version: `prompt-fingerprint-v1`.

## Normalization Behavior

Normalization must:

- lowercase all text
- collapse repeated whitespace
- replace UUID-like values with a stable placeholder
- replace absolute file paths with a stable placeholder
- replace date and timestamp-like values with a stable placeholder
- replace large tool-result blocks with a stable placeholder
- replace system-reminder blocks with a stable placeholder
- preserve imperative steering phrases such as `do not implement`, `edit only`,
  `review and refine`, and `acceptance tests required`

The implementation must record the prompt fingerprint version on every output
record.

## Evidence Excerpts

Evidence excerpts must be short, bounded, and sufficient to justify the label.

MVP excerpt behavior:

- maximum 500 characters
- taken from the smallest text span that supports the label
- redacted before storage
- never copied into metrics labels
- never expanded to include full raw prompt or full tool output

Required redaction behavior:

- redact bearer tokens
- redact API keys
- redact values beginning with common secret prefixes such as `sk-`
- redact authorization header values
- redact cookie header values
- redact obvious password or token assignments

The evidence hash must be computed from the redacted evidence excerpt.
