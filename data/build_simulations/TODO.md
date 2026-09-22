# Construction-token model: remaining work

Planning snapshot: 2026-09-22. This is a plan, not a running campaign or
authorization to start paid provider calls. No construction training is running
on a VM. The existing Studio predictor measures workload execution, not the
software-construction target described here.

## Saved baseline

- [x] Independently construct all 16 standalone types and one integrated build
  at each size two, three and four: 19 measured builds in total.
- [x] Save actual implementations, tests, examples, manifests, artifact hashes
  and per-request subscription-agent usage traces.
- [x] Record 2,921,703 total construction tokens without adding cache or
  reasoning counters twice.
- [x] Publish a 38-field feature dictionary, staffing/duration assumptions,
  counterfactual Azure price scenarios and explicitly pseudo 0-5 outcome scores.
- [x] Separate 14 real-use-case references from independently designed builds.
- [x] Enumerate 1,455 memberships and mark unmeasured entries explicitly.

Current split: 14 training records, zero validation records and five test
records. There is no trained or promoted construction-token model. The 19
bounded Python CLI builds are not evidence of complete enterprise delivery.

## 1. Freeze the next measurement protocol

- [ ] Select an initial 100-200 additional independent builds, balanced across
  types, pairs, triples and quadruples, with repeat trials to estimate variance.
  This is a sampling target, not a guarantee of adequate accuracy.
- [ ] Preserve existing membership-group split assignments. Keep all repeats
  and execution-order variants of a membership in its original partition.
  Populate validation and reserve fresh unseen-combination test memberships
  before collecting or inspecting their labels.
- [ ] Freeze comparable implementation scope, acceptance requirements, builder
  instructions, allowed tools and model/version metadata before dispatch.
  Record actual instructions/specification fingerprints for each new trial.
- [ ] Add unique trial IDs and immutable wave manifests so repeat measurements
  cannot overwrite earlier records. Re-importing an agent is not another trial.
- [ ] Version any richer features for order, reuse, handoffs and scope. Do not
  treat assumed staffing or months as experimentally observed token drivers.
- [ ] Freeze baseline comparisons, acceptance thresholds, uncertainty-coverage
  targets and stopping rules using training/validation evidence only.

Done when: the selected memberships, trial IDs, split groups, specifications
and evaluation rules are saved before new builders start.

## 2. Collect and preserve additional build evidence

- [ ] Use isolated subscription subagents first; measure each integrated build
  directly rather than copying standalone source or summing standalone labels.
- [ ] Start with bounded concurrency (planning assumption: 3-4 agents), subject
  to subscription allowance. Record quota limits, failures and elapsed time.
- [ ] Capture completed-agent input/output counters, cache details, model IDs,
  timestamps, test/repair consumption and the declared measurement boundary.
  Keep unsuccessful trials as failed evidence rather than silently dropping
  their costs or substituting predicted labels.
- [ ] Run static validation and independent runtime verification of each
  accepted artifact; save outputs and hashes with the trial.
- [ ] Keep parent orchestration separate from builder usage. Never export
  credentials, the raw session database or unrelated conversations.
- [ ] Do not use direct paid API fallback without a separately approved budget.
  Save each completed wave before local grading and update coverage afterward.

Done when: each accepted label has unique authoritative events and reproducible
artifacts, validation is populated, and failed/incomplete trials remain visible.

## 3. Train and evaluate the construction predictor

- [ ] Fit input and output construction tokens separately; derive total usage
  and cost afterward. Never pool workload-execution labels into this dataset.
- [ ] Exclude post-build sizes, actual test counts, observed usage, price labels
  and pseudo outcomes from pre-build input features.
- [ ] Compare a simple training-mean baseline with regularized models using
  grouped cross-validation; fit preprocessing inside each training fold.
- [ ] Tune only with training/validation groups. Report learning curves,
  train/validation gaps, MAE, relative errors, residuals by composition size
  and type, and repeat-trial variability.
- [ ] Evaluate the frozen candidate once on untouched test groups. Publish
  uncertainty intervals, coverage, sample counts and limitations alongside
  metrics. Failed acceptance must not silently redefine the same test set.
- [ ] Save immutable model artifacts, feature/protocol versions, split manifests
  and per-iteration reports. Promote only if the frozen gates pass; otherwise
  retain the current model and document the next measurement requirement.

Done when: a reproducible evaluation supports the declared inference domain.
Neither absence of overfitting nor universal composition accuracy is assumed.

## 4. Connect the accepted model to Studio

- [ ] Provide a separate read-only construction forecast endpoint and registry;
  preserve the existing workload-execution predictor.
- [ ] Wire both marketplace selections and custom project inputs to the
  complete selected composition, not a sum of standalone predictions.
- [ ] Show construction input/output/total tokens, uncertainty, model version
  and reference-priced cost immediately through local inference. Define and
  verify the latency target before release; selection must not start training.
- [ ] Clearly distinguish build cost from recurring execution cost, declared
  staffing and pseudo ROI. Warn or abstain outside supported scope.
- [ ] Perform browser runtime verification for all 16 types, unseen accepted
  combinations, custom inputs, edits, persistence and explicit error states.
- [ ] Update the walkthrough only after the construction path is actually live.

## 5. Evaluate feedback and recommendation learning separately

- [ ] Keep `build-outcomes-0to5-v1` distinct from older 1-5 feedback schemas.
- [ ] Implement a clearly labeled synthetic reward/re-ranking experiment, with
  held-out scenarios and a fixed-policy baseline; do not claim real ROI gains.
- [ ] For future real learning, collect reviewed client outcomes, action and
  policy versions, available choices and logged propensities prospectively.
- [ ] Promote a real recommendation policy only after suitable held-out outcome
  evaluation. Random pseudo scores alone cannot establish policy improvement.

## Planning estimates, not a running-job promise

For 100-200 additional builds: roughly 4-10 hours of collection with 3-4
concurrent agents, plus 5-30 minutes of local fitting/evaluation compute and
2-4 hours of Studio integration. Allow roughly 1-2 working days for the first
iteration, with additional time if quality gates fail.

The current wave averaged about 154,000 consumed tokens per build. At the saved
GPT-5.4 no-cache reference rates, holding those counts constant gives about
$0.52 per build, or $52-$103 for 100-200 builds; a $75-$200 allowance includes
possible repairs and larger scopes if API fallback is separately approved.
These are counterfactual reference-price estimates, not a live Azure quote,
Copilot charge or spending authorization. Actual subscription charges remain
unknown. Local regression fitting does not require provider API calls.
