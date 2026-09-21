# How Token Yield is tested

Token Yield is tested as a complete learning system, not only as a prediction
formula. The tests follow the same path as a real project:

```text
Understand -> Decompose -> Measure -> Build features
    -> Train -> Evaluate -> Predict -> Observe -> Learn
```

## 1. Understand existing and new functionality

The first test is whether the system understands what the requested agent
capability does.

For an existing function, the catalog must return the matching brick or
composition. For a proposed new function, specialized agents independently
describe its behavior, constraints, evidence, and acceptance criteria.

The contract layer checks:

* Valid agent roles and message types
* Strict JSON structure
* Approved feature atoms
* Capability version and identity
* Required evidence and citations
* Disagreement between agents

Ambiguity or unresolved dissent blocks establishment. A fluent proposal does
not automatically become a catalog brick.

## 2. Decompose work into measurable bricks

Plain-English projects are mapped to named task counts. Tests cover:

* Known requests with expected brick combinations
* Multiple units of the same brick
* Previously unseen combinations
* New-function proposals
* Requests the current vocabulary cannot represent
* Heuristic fallback behavior when an agent encoder is unavailable

The decomposition becomes the shared structure for training, quoting, and
later chargeback.

## 3. Pre-simulate representative workloads

Each approved brick is exercised across relevant sizes and combinations before
it is trusted for prediction.

The measurement contract requires:

* Frozen workload identity
* Runtime and model identity
* Input, output, cached, reasoning, and total tokens
* Latency, retry, tool, and failure evidence
* Contract and quality results
* Source provenance
* Budget reservation and settlement

Incomplete and rejected attempts remain visible because they still consume
tokens and time.

Offline tests use explicit synthetic measurements. Foundry campaigns use
metered provider responses and an immutable campaign budget.

## 4. Build leakage-safe features

Feature engineering is tested independently from model fitting.

Only quote-time information can enter the predictor:

* Brick counts
* Prompt and context size
* Planned output allowance
* Call layout
* Document and requirement counts

Realized output size, retries, final quality, and actual cost are targets or
post-run diagnostics. Tests reject them when they appear as pre-run features.

## 5. Train competing models

The trainer compares multiple candidate forms:

* Constant
* Size
* Size plus units
* LEGO ridge

The implementation tests:

* Deterministic fitting
* Numerical guards and finite outputs
* Candidate selection
* Grouped cross-validation
* Stable JSON serialization
* Reload and prediction without refitting

Repeated calls from one project remain in the same fold. This prevents the
model from seeing a near-copy of its evaluation project during training.

## 6. Evaluate on unseen work

The strongest model test is not how closely it fits its training rows. It is
how well it predicts work it did not see.

Evaluation includes:

* Projects excluded from fitting
* Brick combinations excluded from fitting
* Plain-English requests decomposed before execution
* Runtime and template compatibility
* Requests outside observed feature ranges
* Comparison with simpler baselines

Current measured results include:

| Evaluation | Result |
| --- | ---: |
| Leave-one-out error on 35 measured agent runs | 2.55% average total-token error |
| Four held-out brick combinations | 2.2% average error |
| Three plain-English requests | 0% to 3.5% error |
| Marketplace model on 20 calls from two held-out projects | 32.5 input-token and 35.2 output-token average error per call |

An unsupported request returns an explanation and no production-shaped quote.

## 7. Test agent coordination and human control

The marketplace is tested as a stateful workflow.

Server and agent tests verify:

* A page view never starts paid work
* A stage runs only after its current approval
* Duplicate and stale approvals fail
* Cancellation stops before the next safe boundary
* Final artifact publication is atomic
* Only one paid run can be active
* Run and call limits are enforced
* Budget is reserved before dispatch
* Failed and unknown calls remain charged or reserved
* Restart cannot reset campaign spend
* Saved artifacts reproduce the same predictions

Browser tests drive the sales, operations, provenance, feedback, and terminal
views. They also confirm that narrow and desktop layouts expose the same run
state.

## 8. Test the feedback and reward loop

Reinforcement requires an explicit reward, not vague positive feedback.

The [local delivered-project feedback prototype](customer-outcomes-prototype.md)
has executable coverage in `tests/test_customer_outcomes.py`,
`tests/test_delivery_feedback.py`, and `tests/test_brick_learning.py`. These
tests exercise reward extraction, outcome regression, pending evidence,
real/synthetic separation, duplicate-credit prevention, protected holdouts,
human-gated promotion, persistence, HTTP boundaries, and changed recommendations.
They demonstrate local learning with synthetic evidence, not production
customer-value improvements.

Run these alongside the existing model and measurement-policy tests:

```powershell
python -m pytest -q tests/test_customer_outcomes.py tests/test_delivery_feedback.py tests/test_brick_learning.py tests/test_customer_models.py tests/test_marketplace_models.py tests/test_measurement_policy.py
```

The existing measurement policy rewards calibration improvement per reference
dollar and is gated to offline mocks. The verified business-impact reward
described below remains a proposed validation plan, not live ROI-learning
results already demonstrated by the recorded campaign.

Token Yield separates several signals:

| Signal | Example metric | Why it matters |
| --- | --- | --- |
| Forecast accuracy | Absolute token error | Improves cost prediction |
| Bias | Mean signed relative error | Detects systematic underpricing or overpricing |
| Acceptance | Human-approved outcome | Prevents optimizing for attempts rather than useful work |
| Quality | Contract and reviewer result | Prevents cheap but unusable delivery |
| Efficiency | Cost per accepted outcome | Balances cost and successful completion |
| Human effort | Review and correction time | Captures work shifted from agents to people |
| Client value | Adoption, time saved, cost avoided, revenue supported | Builds the future outcome and ROI model |

The proposed outcome funnel and reward are defined in the
[learning architecture](architecture.md#8-learn-from-delivery):

```text
AES = 100 * independent_completion * retention * verified_customer_usefulness
net_benefit = verified_attributable_benefit - total_delivery_and_operating_cost
reward = w_A * (AES / 100)
       + w_B * clip(net_benefit / reference_benefit, -1, 1)
```

The weights sum to one; the reference benefit is positive. Freeze the cohort,
work-unit definitions, weights, evidence threshold, and observation window
before a trial. Safety, acceptance, and budget gates cannot be overridden by
reward. Financial costs already included in net benefit are not subtracted
twice. Client-reported value remains distinct from verified value.

Before enabling a production business-impact extension, validate these cases
against delivery evidence; local synthetic tests are not sufficient:

| Proposed test | Required outcome |
| --- | --- |
| 100 assigned units, 80 independently completed, 60 kept, 30 verified useful | A = 0.8, K = 0.75, U = 0.5, AES = 30 |
| Required human approval with no corrective intervention | Approval does not reduce independent completion |
| Substantive human correction or agent retries | Rescue is recorded; all resource and review costs remain charged |
| No assigned units or invalid funnel counts | No score; report the invalid scope or inconsistent evidence |
| Proven zero completions or zero retained units | Zero useful autonomous yield; downstream rates are not applicable |
| Missing customer outcome evidence | Provisional funnel only; final reward remains pending |
| Duplicate or delayed feedback | Credit the original work once; retain timestamp and evidence revisions |
| Claimed time savings without a monetization baseline | Do not turn hours into verified financial benefit |
| Verified benefit below full delivery cost | Preserve negative net benefit and ROI |
| Zero or unknown total cost | ROI remains unavailable |
| High reward with failed safety or budget checks | Block policy promotion and execution |
| Policy learned on calibration feedback | Final holdouts remain untouched by reward tuning |

New records are scored against the model that existed before they arrived.
Only then are they eligible for the training store. This preserves drift and
surprise instead of allowing immediate refitting to hide them.

## 9. Run the test suites

Install the development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Run the complete Python suite:

```powershell
python -m pytest -q
```

Run the measured prediction demonstration:

```powershell
python -m examples.composition_demo
```

Run the offline marketplace:

```powershell
python -m examples.marketplace_demo_server --port 8765
```

Then open:

<http://127.0.0.1:8765/marketplace-sales-demo.html?mode=custom>

Paid Foundry evaluation is not part of the default test suite. It requires an
explicit campaign approval, budget, approved deployment, and configured
credentials.

## 10. Evidence produced by a run

A complete run preserves:

* The original request
* The chosen bricks and feature vector
* Agent proposals and human approvals
* Measurement rows
* Candidate and selected model details
* Evaluation metrics
* Token and cost forecasts
* Actual token usage and reconciliation
* Feedback and reward signals
* Model, source, and artifact hashes

This evidence makes the result reproducible and gives the next learning cycle a
traceable starting point.
