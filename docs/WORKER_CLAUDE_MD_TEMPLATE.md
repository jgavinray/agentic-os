# Worker CLAUDE.md Template

Copy the block below into the `CLAUDE.md` of any repository a small local
model works on through the orchestrator. Small dense models follow short,
imperative, front-loaded rules far better than prose; keep the whole file
under ~60 lines and put repo specifics (build commands, layout) immediately
after this block.

The `SCOPE-CHECK:` and `STUCK:` markers are deterministic strings: the
orchestrator (or an operator scanning transcripts) can detect them in
responses and escalate, which converts "the model gave up silently" into an
observable event. They pair with the orchestrator-side guardrails: tool menu
narrowing, per-command shell authorization, the injected Operating Envelope,
and edit-scope enforcement.

---

```markdown
# Working Rules (read first)

## Scope
- Do exactly what the task asks. Nothing else.
- Do not refactor, rename, reformat, or "improve" code you were not asked to change.
- Do not create new files unless the task requires one.
- Do not edit files you have not read first.
- If a change outside the stated scope seems necessary, STOP and write:
  `SCOPE-CHECK: <file or change> — <why it seems needed>`
  then end your turn. Do not make the change.

## Validation
- After every edit, run the build check (e.g. `cargo check`).
- Before finishing, run the narrowest tests that cover your change
  (e.g. `cargo test <module>`).
- Never claim success without showing the passing command output.
- Do not run or "fix" unrelated test suites.

## Forbidden
- `git commit`, `git push`, branching, tagging.
- Deleting files or directories.
- Docker, service restarts, deployments.
- Installing packages or changing dependency manifests unless asked.

## When stuck
- After 2 failed attempts at the same fix, stop and write:
  `STUCK: <what you tried> — <what failed>`
  then end your turn. Do not try a third variation.
- If the task references a file or symbol that does not exist, say so
  instead of guessing.

## Reporting
- End with: files changed, validation command + result, anything left undone.
```
