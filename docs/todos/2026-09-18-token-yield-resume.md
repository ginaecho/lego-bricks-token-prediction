# Token Yield resume plan (2026-09-18)

Snapshot of where the project stands and what to pick up next. The core
new-brick feedback loop is complete and verified; the items below are follow-ups.

## Current state

* `origin/main` and local `main` are synced at `5f2fd93` (the assay merge).
* `gc/model-improvement` is fully contained in `main`.
* Pull request #4 (assay falsification harness) is merged. No open pull requests remain.
* The full agent loop ran live and published a model that includes a new,
  human-approved brick:
  * Published model version `26b8a14c`, run `5d80e7e2`, new brick `novel_e154b3027f28`.
  * Holdout MAE input 120.94 / output 118.50 versus baseline 177.75 / 196.06 (about 32% and 40% better).
  * The live catalog pointer references this model.
* Paid campaign accounting: 444 calls, rated USD 3.54, safety-accounted USD 42.96,
  USD 0 reserved, cap USD 100, operational stop USD 96, about USD 53 remaining to the stop.

## How to run the app

See [run-the-app.md](../run-the-app.md). Offline mock mode makes no paid calls;
live Foundry mode requires Azure credentials and an explicit budget.

## Follow-up backlog

### 1. Fix pre-existing feedback test failures

* 21 tests in `tests/test_marketplace_feedback.py` fail with
  `RuntimeError: human establishment approval callback required`.
* These fail on pristine `79de7b0` as well, so they predate the assay merge.
  Older new-function tests do not supply the human-approval callback that the
  establishment gate now requires.
* Fix: update those tests to pass an establishment-approval callback (approve or
  reject as the assertion intends), matching the pattern already used in
  `tests/test_marketplace_agents.py`. Verify the full suite, then have the change
  independently reviewed.

### 2. Harden Tab 03 saved-build loading

* Deferred robustness work on `marketplace-sales-demo.html` saved-build snapshots.
* Reject malformed snapshots atomically: boolean or array values must not coerce
  to numbers, negative commercial rates must be rejected, and unknown assumption
  keys must not partially apply. A rejected load must leave the active build,
  rates, and assumptions unchanged, with no default-rate quote leak.
* Add regression coverage for these malformed cases across desktop and mobile.

### 3. Re-enable the measurement-selection policy and versioned scope

* The reward-based measurement-selection policy (bandit) and the versioned
  Archive Manager v2 paid scope are implemented and independently reviewed, but
  intentionally left disabled.
* Resuming either one needs a separate, explicit paid-run authorization bound to
  the original canonical funding store, then a bounded live run and verification.

### 4. Optional new-brick live demo polish

* An optional live walkthrough of the new-brick loop for presentation. The
  recorded demo already exists at
  [docs/media/token-yield-live-loop-demo.mp4](../media/token-yield-live-loop-demo.mp4).

## Known limitations to preserve

* Measured multi-brick combinations are unavailable; project forecasts are
  additive sums of individual brick forecasts, not jointly measured workflows.
* Published pilots are small-sample, MAE-only, reference-context, and uncertified
  for production. Staffing, labor rates, and ROI remain planning assumptions.
* New-brick establishment always requires an explicit human approval. Do not
  fake agreement, invent usage, or bypass the acceptance gate to force a publish.

## Guardrails for any resumed paid work

* Preserve the canonical funding store and its lock; never fork, copy, or move it.
* Preserve the original USD 25 pilot ledger and all historical run records.
* Stop honestly on genuine dissent, provider or accounting ambiguity, unknown
  telemetry, or the operational stop. No automatic retries beyond the bounded
  policy.
