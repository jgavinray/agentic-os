# Overview

Prompt intervention telemetry records evidence that a model path required
external steering before it produced useful work.

The system should answer:

> Did this model or orchestration path succeed directly, or did it require
> stop/redirect instructions, scope narrowing, prompt repair, quality gates,
> validation demands, context recovery, or model-failure correction before the
> work became usable?

## Purpose

Accepted output alone is not enough. If a local model eventually succeeds only
after repeated steering, stop instructions, prompt repair, and human attention,
that burden must be visible in telemetry.

The telemetry must help answer:

- Which model paths produce accepted work without intervention?
- Which model paths require repeated prompt repair?
- Which prompt families require repeated scope narrowing?
- Which policy changes reduce intervention burden?
- Which local model paths are false economy because they increase human
  steering?

## Ownership

This belongs to telemetry and feedback measurement.

It may be consumed by adversarial validation later, but it is not part of the
adversarial finding lifecycle. It records operational evidence about steering
burden, not whether a model finding is true.

## Existing Inputs

The implementation must use telemetry the system already captures.

Primary input:

- raw HTTP exchange records containing request body, parsed request body,
  forwarded request body when available, response body, endpoint, timestamps,
  and exchange ID

Optional correlated inputs:

- model-attempt metadata from forwarded request metadata
- trajectory ID
- request event ID
- attempt ID
- requested model
- routed model
- baseline arm
- selected route
- routing policy version
- outcome feedback records

The implementation must tolerate missing optional metadata. A record with an
exchange ID and detectable intervention evidence is valid even when attempt,
trajectory, or routing metadata is unavailable.

## Non-Goals

- Do not mutate raw prompt or response captures.
- Do not store full raw prompts in intervention records.
- Do not treat an intervention as proof that the final answer was good or bad.
- Do not infer that ChatGPT, a human, or another assistant authored a prompt
  unless provenance was explicitly captured.
- Do not use raw prompt text, file paths, commands, auth values, or user text as
  metrics labels.
- Do not commit workspace-local prompt bodies, response bodies, labels, or
  backfilled datasets into the open-source repo.
- Do not add model-assisted labeling to the MVP runtime path.
