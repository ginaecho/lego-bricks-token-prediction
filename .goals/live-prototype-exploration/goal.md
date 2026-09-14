# Goal: Run and assess the Token Yield prototype

## User Request

With the prototype, save it somewhere so it is not revised by the following exploration. Use the existing methods and model to apply the prototype to really run it, then find what should be improved in the LEGO brick pre-simulations and model building.

## Refined Goal

Preserve an immutable snapshot of the current HTML prototype, then execute the demonstrated sustainability-report request through the repository's existing Token Yield methods using the configured Azure Foundry environment. Produce auditable quote, live-run, and invoice evidence where telemetry permits, and write a separate prioritized report identifying improvements needed in brick pre-simulations, measurement contracts, model fitting, and product readiness. Existing modeling code must remain unchanged.

## Acceptance Criteria

- [ ] A frozen, clearly labeled copy of the prototype exists outside the active prototype path and includes a checksum or immutable reference.
- [ ] The demo request is run through existing repository methods with the configured Azure credential; credentials and endpoint secrets are not written to files or committed.
- [ ] Evidence artifacts distinguish live feasibility results from replay, synthetic, and pipeline-only data; quote, actual usage, reconciliation, and any failure/exclusion reason are recorded.
- [ ] A prioritized improvement report covers pre-simulation design, brick vocabulary/counting, telemetry and cost measurement, model forms/baselines, validation/gates, and product UX implications.
- [ ] Existing modeling source files are unchanged; only isolated exploration artifacts and process records are added.
- [ ] Repository tests and the smallest relevant quality gates are run and their results are recorded.

## Scope Boundaries

**In scope:**
- Snapshotting the current prototype.
- Running one live demo request through existing methods and configured Azure Foundry access.
- Recording evidence and limitations.
- Inspecting existing experiments and methods to produce a prioritized improvement report.
- Running existing tests/quality gates.

**Out of scope:**
- Editing existing modeling code, brick definitions, or production pipeline behavior.
- Implementing the recommended model or pre-simulation improvements.
- Storing API keys, tokens, or other credentials in the repository.
- Treating a failed or incomplete live run as measured success.

## Applicable Project Conventions

**Quality gate command:**
- `python -m pytest -q`
- `make prove` when feasible; live external calls are not required for offline gates.

**Commit convention:**
- Conventional commits with goal role markers: Builder uses `type(scope): [B] description`; Inspector uses `chore(scope): [I] description`.
- Assisted-by trailer required: `Assisted-by: Claude:Sonnet-4.6`

**Guidelines:**
- `.agents/skills/experiment-design/SKILL.md`
- Repository Python and Markdown instructions supplied by the workspace.

**Rules:**
- Preserve user changes and avoid unrelated modifications.
- Keep credentials in environment/configuration only; never write them to files.
- Label pipeline-only, replay, synthetic, and feasibility evidence explicitly.
