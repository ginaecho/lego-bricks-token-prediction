# Marketplace model implementation plan

Date: 2026-09-16.
Inputs: [requirements](spec.md), [constraints](constitution.md),
[architecture](architecture-summary.md), [structure](project-structure.md),
and [technology stack](tech-stack.md).

## Summary and TODO roadmap

Build and verify an offline shared token-prediction baseline first. Then add
versioned marketplace contracts, pricing, broader measurements, composition,
calibration, and evidence-gated specialization.

1. Document requirements and architecture.
2. Build input/output token channels and freeze an offline baseline.
3. Define the four marketplace service contracts.
4. Wire explicit selections into estimates and separate price calculation.
5. Acquire permitted workloads and measure broader execution conditions.
6. Fit workflow adjustments and calibrate ranges on independent projects.
7. Evaluate specialist models and introduce controlled feedback/versioning.

## Technical context and constraint check

Reuse Python, JSON artifacts, the existing ridge/constant estimators, and pytest.
No new dependency, deployment, source download, or paid execution is needed for
the first milestone. Existing APIs and recorded runs remain unchanged.

The design satisfies all [constraints](constitution.md): measured labels,
project separation, explicit failures, provenance, no unauthorized spending,
and no production or interval claim. These constraints must be rechecked at
each subsequent milestone.

## Applied guidelines

Use existing Python module conventions, typed public functions, pathlib,
strict finite-number validation, and deterministic JSON persistence. Retain
Python 3.9 compatibility. Use focused tests from the existing pytest suite.
No repository-local migration guideline directory was found; this is a model
extension, not a technology migration.

## Phase 1: Requirements and design

Plan 1.1 addresses REQ-001 through REQ-010.

- [x] T001 [Plan:1.1] Save the specification, constraints, architecture, structure, and stack under `docs/marketplace-models/`; include requirement and task checkpoints.

## Phase 2: Offline shared-model foundation

Plan 2.1 addresses REQ-001, REQ-002, REQ-003, REQ-004, and REQ-010.

- [x] T002 [Plan:2.1] Add shared call-level input/output fitting, support-aware forecasting, and disjoint holdout evaluation in `token_yield/customer_models.py`; retain existing cost/project schemas.
- [x] T003 [Plan:2.1] Add targeted checks in `tests/test_customer_models.py` for grouped selection, token units, serialization, invalid selections, runtime mismatch, and backward compatibility.

Plan 2.2 addresses REQ-003, REQ-004, REQ-005, REQ-009, and REQ-010.

- [x] T004 [Plan:2.2] Add audited offline training in `token_yield/customer_pilot.py` and `examples/marketplace_token_models.py`; verify raw measurements and write only a new run directory.
- [x] T005 [Plan:2.2] Train on `runs/20260914_customer_scoping_v2/` training calls; save reloadable artifacts and retrospective holdout analysis in `runs/20260916_marketplace_tokens_v1/`, then document results in `docs/marketplace-models/plan.md`.

Dependencies: T002 follows T001; T003 and T004 follow T002; T005 follows both.
Exit: two reproducible channels, saved parameters, explicit limitations,
historical holdout scores, and passing targeted regression checks. Winning
training-CV error is not an unbiased generalization estimate.

## Phase 3: US2 service contracts and quoting

Plan 3.1 addresses REQ-001, REQ-004, and REQ-006.

- [x] T006 [US2] [Plan:3.1] Create `token_yield/marketplace.py` with four versioned service contracts derived from `experiments/customer_requests/template.json`, preserving the actual scoping semantics and acceptance checks.
- [x] T007 [US2] [Plan:3.1] Add explicit-selection validation and quote feature generation in `token_yield/marketplace.py`; reject unknown versions, invalid quantities, incompatible inputs, and unsupported execution settings.

Plan 3.2 addresses REQ-007 and REQ-010.

- [x] T008 [US2] [Plan:3.2] Reuse `token_yield/economics.py` from `token_yield/marketplace.py` to rate token forecasts; persist rate-card identity and separate API estimates from tool/review fees and selling-price margin.

Dependencies: T006 follows T005; T007 follows T006; T008 follows T007.
Exit: selecting a supported service produces a conditional research estimate;
unsupported selections yield reasons rather than an apparently valid price.

## Phase 4: US3 broader measurements and composition

Plan 4.1 addresses REQ-005, REQ-008, and REQ-009.

- [x] T009 [US3] [Plan:4.1] Create `experiments/marketplace/manifest.json` with source permissions, deduplication groups, train/calibration/test assignments, input-size/quantity matrix, quality criteria, and a separately approved execution budget.
- [ ] T010 [US3] [Plan:4.1] Extend `token_yield/marketplace.py` and an explicitly gated `examples/marketplace_measurements.py` to record versioned standalone, batched, sequential, and failed/retried executions without treating unknown usage as zero.
- [ ] T011 [US3] [Plan:4.1] Add and compare measured workflow adjustments in `token_yield/marketplace_models.py`; never reuse the historical batching percentage as a universal discount.

Dependencies: T009 follows T008; T010 follows T009 and approval; T011 follows
completed measurements. Explicitly distinguish metered incomplete executions
from complete accepted deliverables.

## Phase 5: US3 calibration and controlled specialization

Plan 5.1 addresses REQ-009 and REQ-010.

- [ ] T012 [US3] [Plan:5.1] Preregister production error, coverage, interval-width, quality, latency, and sample-size gates in `docs/marketplace-models/promotion.md` before inspecting new test outcomes.
- [ ] T013 [US3] [Plan:5.1] Implement independent calibration and untouched-test reporting in `token_yield/marketplace_models.py`; expose no calibrated range if evidence is insufficient.
- [ ] T014 [US3] [Plan:5.1] Compare shared, family, and dedicated predictors using grouped validation in `token_yield/marketplace_models.py`; promote specialists only after the frozen decision rule passes on independent evidence.
- [ ] T015 [US3] [Plan:5.1] Add consented telemetry, drift reporting, versioned offline retraining, and explicit promotion/rollback metadata in `token_yield/marketplace.py`; never silently replace the quoted model.

Dependencies: T012 follows T009 and precedes new test analysis; T013 follows
T011 and T012; T014 follows T013; T015 follows T014.
Exit: demonstrated evidence against the preregistered gates, or an explicit
research-only status. No automatic promise of specialist superiority.

## Validation and release checks

Run focused customer model/pilot regression tests. Check that the saved artifact
reloads without fitting and reproduces its predictions, source hashes match,
and no holdout project enters training. Inspect the final diff and preserve all
unrelated changes. New acquisition, cloud execution, and production promotion
remain separate gates.

## Requirement mapping

| Requirement | Plan items | Implementation evidence |
|---|---|---|
| REQ-001 | 1.1, 2.1, 3.1 | Four-operation schema and versioned registry |
| REQ-002 | 1.1, 2.1 | Two token channels and four candidate forms |
| REQ-003 | 1.1, 2.1, 2.2 | Training folds and disjoint historical evaluation |
| REQ-004 | 1.1, 2.1, 2.2, 3.1 | Runtime checks, source hashes, reloadable artifacts |
| REQ-005 | 1.1, 2.2, 4.1 | Offline run and approved provenance manifest |
| REQ-006 | 1.1, 3.1 | Explicit marketplace selection contract |
| REQ-007 | 1.1, 3.2 | Versioned price calculation separate from tokens |
| REQ-008 | 1.1, 4.1 | Measured composition comparisons |
| REQ-009 | 1.1, 2.2, 4.1, 5.1 | Quality ledger, independent calibration and promotion gates |
| REQ-010 | 1.1, 2.1, 2.2, 3.2, 5.1 | Regression checks and specialization decision record |

Mapping coverage is planning coverage, not completed implementation.

## First milestone results

Completed T001-T009. T010 is partially implemented; T011's offline implementation
is in progress, but measured fitting is blocked by the incomplete campaign.
T012-T015 remain pending.

The [saved run](../../runs/20260916_marketplace_tokens_v1) contains
[models](../../runs/20260916_marketplace_tokens_v1/models.json),
[predictions](../../runs/20260916_marketplace_tokens_v1/predictions.json), and
[analysis](../../runs/20260916_marketplace_tokens_v1/analysis.json).

Training used 40 measured calls from four projects. Historical evaluation used
20 calls from two different projects. Both token channels selected LEGO using
training-only leave-one-project-out validation.

| Historical holdout MAE, tokens per call | Input | Output |
|---|---:|---:|
| Constant baseline | 60.49 | 228.15 |
| Size baseline | 23.25 | 123.28 |
| Size+units baseline | 29.66 | 88.27 |
| Selected LEGO predictor | 32.50 | 35.24 |

The size baseline beat the selected LEGO model on historical input-token error.
Do not switch based on these holdout labels. This is evidence for an independent
future comparison, not proof that LEGO is uniformly better.

Validation passed 181 targeted customer model/pilot tests. All 60 persisted
forecasts reproduced after JSON reload with fitting disabled. Historical
evaluation reproduced exactly; 122 source-file hashes and all five training
code hashes matched. No new API calls were made; additional API spend was USD 0.
Intervals and upper budget bounds remain null. These are call-level scoping
estimates, not whole-project delivery forecasts or production price guarantees.

### Reproduce offline

From the repository root, choose a new output directory:

```powershell
.\.venv\Scripts\python.exe -m examples.marketplace_token_models `
  --source-run runs\20260914_customer_scoping_v2 `
  --run-dir runs\marketplace_tokens_reproduction
```

The command refuses to overwrite an existing output directory or write inside
the frozen source run. The corresponding
[VS Code task](../../.vscode/tasks.json) records the first run's command; change
its output directory before rerunning it.

## Service selection and pricing milestone

T006-T008 are complete. See the [quoting guide](quoting.md) for API contracts,
supported modes, rate-card format, commercial assumptions, and a runnable
offline example.

The implementation preserves the original four scoping services. Explicit
selections are versioned by template hash and validated against the supplied
requirements. Unsupported batches, runtime changes, and out-of-range inputs
receive no marketplace total or price. Separate execution sums fresh,
independent calls; it does not model sequential handoffs.

Pricing uses a runtime-bound, versioned rate card with no assumed cache saving.
Unknown tools/review fees and gross margin remain null. The content-hashed
quote and rated quote preserve model, template, selection, and pricing identity.

Validation passed 166 marketplace, customer-model, and targeted pricing tests,
plus the earlier 40 marketplace/contract tests. The
[saved offline example](../../runs/20260916_marketplace_quote_v1) selects Extract
and Report for the Paradigm brief in separate calls. It estimates 1,409.91 total
tokens and USD 0.007798 API cost under the recorded runtime and September 14
rates. Selling price remains unknown because no commercial fees or margin were
supplied. This example spent USD 0 and is not a current provider quote.

## Measurement authorization

On 2026-09-16, the user approved an additional API budget up to USD 50 and chose
to reuse only the six previously approved paraphrased briefs for execution
variants. No new web retrieval, private input, customer contact, or deployment
is authorized. The new cap is separate from the historical pilot accounting;
historical budget files must not be changed.

The same six briefs cannot supply a new independent calibration/test set.
Retain their historical group identities and label repeated measurements as
execution-variant research, not newly unseen projects.

T009 is complete: the frozen manifest verifies 120 planned calls across eight
independent batching layouts and six briefs. Preparation and accounting checks
passed 59 marketplace, measurement, and budget tests. The explicitly gated
runner executed the approved campaign and halted on the 36th call after a
contract failure.

See the [measurement guide](measurements.md) for preview, execution, accounting,
and failure handling. T010 remains unchecked because independent batching is
implemented but separately versioned sequential handoffs and retries are not.

### Measurement outcome

The [saved measurement run](../../runs/20260916_marketplace_measurements_v1)
contains 36 attempted calls, of which 35 passed the original acceptance checks.
The final Report response supplied report fields directly instead of nesting
them under the required `report` key. The response remains a failed attempt;
it was not repaired after execution or counted as an accepted deliverable.

All 36 raw request/response hashes, measured usage channels, recorded costs,
and acceptance checks were independently replayed successfully. Historical-rate
calculated spending is USD 0.21738, including USD 0.00379 for the failed call.
Conservative safety accounting settled USD 2.62992. There are no unknown charges
or active reservations. These figures are not reconciled provider invoices.

No retry or resume occurred. Only the first two training briefs were reached,
so the run cannot satisfy the full measurement matrix or support T011 fitting.
The approval claim and failed evidence remain preserved. Another paid attempt
requires reviewed authorization and cumulative accounting, not deleting the
claim or resetting the original cap. Existing trained baseline artifacts remain
unchanged.
