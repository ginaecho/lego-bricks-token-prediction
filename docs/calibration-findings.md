# Calibration: where the data comes from, and what it said

The first version of Token Yield could not answer its own foundational
question. `CalibrationStore` accepted numbers you handed it — a data-entry API,
not a calibration layer — and the model those numbers fed was frozen: `1×/2×/4×`
complexity multipliers and a `+15%` interaction surcharge, all hardcoded. No
quantity of data could move them. "Tune the model from real runs" was impossible
by construction.

This document records how that was fixed, what was measured, and what the
measurements falsified.

## 1. Is agent token usage measurable at all?

Yes, directly. A dispatched subagent returns a usage block alongside its result:

```
subagent_tokens: 40540
tool_uses: 1
duration_ms: 4804
```

That is the measurement channel the whole layer rests on. The very first probe
— *read one file, answer one question in two sentences* — cost **40,540
tokens**, which immediately suggested the finding that reshaped everything: a
trivial task is not cheap, because most of its cost is not the task.

## 2. Probe design

A probe is a self-contained task at a declared **kind** and **scope**
(`token_yield/probes.py`). Two design constraints matter:

- **Graded scope within a kind.** A slope cannot be estimated from one point
  per kind. Scope was varied 1 → 3 → 5 → 8.
- **Replicates with byte-identical prompts.** Without repeats there is no way to
  separate model error from run-to-run variance.

Each probe runs in a fresh, memoryless subagent, so runs are independent.

| kind | scope unit | scopes measured | runs |
|---|---|---|---|
| `comprehension` | files read | 1, 1, 3, 3, 3, 5, 8 | 7 |
| `code_write` | functions written | 1, 3, 3, 8 | 4 |

## 3. What was falsified

### 3.1 Cost does not scale with work

| scope | comprehension tokens |
|---|---|
| 1 | 41,651 |
| 3 | 46,359 / 42,402 / 42,402 |
| 5 | 47,236 |
| 8 | 57,732 |

**8× the scope moved tokens 1.39×.** The multiplicative model predicted 8×.

Fitting `tokens = fixed + marginal × scope` gives `37,571 + 2,305 × scope` — a
large per-invocation constant plus a small marginal. For `code_write` the
marginal is smaller still (~245/function), and cross-validation preferred a
plain constant.

The fixed part is the **agent boot cost**: system prompt, tool schemas,
scaffolding — paid before any of your work begins. It is nearly identical
across kinds (37.6k vs 38.9k), which is what you would expect of a cost that
has nothing to do with the task.

### 3.2 The interaction surcharge had the wrong sign

The asserted model added **+15%** for mixing task types. Measured directly, by
running one agent that did both a 3-file comprehension task and a 3-function
write task:

| | tokens |
|---|---|
| two separate agents | 81,697 |
| one batched agent | 43,491 |
| old model's prediction (+15%) | 93,952 |

Batching cost **53% of separate — a 47% saving**. The old constant was 2.2× the
truth and pointed the wrong way. The affine model explains it exactly: the boot
cost is paid once instead of twice.

### 3.3 There is a noise floor, and it is not small

Identical prompts, fresh agents, repeated:

| kind | scope | n | mean | sd | CV |
|---|---|---|---|---|---|
| comprehension | 1 | 2 | 41,096 | 556 | 1.4% |
| comprehension | 3 | 3 | 43,721 | 1,865 | 4.3% |
| code_write | 3 | 2 | 37,976 | 1,773 | 4.7% |

Pooled: **~5%**. No model can predict a kind more precisely than the same task
repeated predicts itself.

## 4. Method: the shape is fitted, not chosen

`token_yield/costmodel.py` fits four forms and selects by leave-one-out
cross-validation, breaking ties toward fewer parameters:

| form | equation | LOO MAPE, comprehension | LOO MAPE, code_write |
|---|---|---|---|
| **affine** | `a + b·scope` | **5.8%** | 8.1% |
| **constant** | `c` | 10.5% | **4.7%** |
| power | `a·scope^b` | 7.9% | 7.8% |
| proportional | `b·scope` | 49.3% | 90.0% |

`proportional` *is* the old `1×/2×/4×` rule. It finishes last in both.

Cross-validation rather than in-sample fit matters here: a more flexible form
must not win merely by having more parameters to bend. And the selection is
genuinely two-sided — given through-origin data it picks `proportional`, given
flat data it picks `constant`. Tests pin all three directions.

## 5. Validation

**Skill ratio.** `backtest.py` reports cross-validated error as a multiple of
the noise floor. `comprehension` sits at 1.29×, `code_write` at 0.71× — both at
or near the limit the process allows. The actionable reading: more measurement
of these kinds will not help; reducing run-to-run variance would.

**Out-of-sample.** The strongest claim. Per-kind models were fitted *only* on
single-kind runs; the batched runs were held out entirely. Predicting them:

| | predicted | measured | error |
|---|---|---|---|
| two separate agents | 83,369 | 81,697 | 2.0% |
| one batched agent | 45,142 | 43,491 | 3.8% |

Both inside the 5% noise floor, from a structure that was never fitted to them.
That is the evidence that the fixed/marginal split is physically real rather
than a curve-fitting artefact. Pinned by
`test_fitted_models_predict_the_unseen_composition_experiment`.

## 6. The loop

`learn.py` scores each new record **against the standing model before absorbing
it**. The ordering is the entire discipline — once a record is folded into a fit
it can no longer surprise it, and a model that silently absorbs contradicting
data looks healthy forever.

A `DriftReport` fires on systematic one-sided error, on error above threshold,
or on scope outside the fitted range, and it names the direction: a budget 50%
low and one 50% high call for opposite actions.

## 7. Limits

- **Eleven runs, two kinds, scope 1–8, one harness, one model, one repository.**
  Every fitted model records the range it was fitted over, and `plan.py` flags
  extrapolation beyond it rather than silently answering.
- **`code_write` selecting `constant` means scope carried no signal over the
  measured range** — not that writing code has no scale cost. Its slope is real
  but small enough to hide under the boot cost at these sizes.
- **Duration is far noisier than tokens** and does not track them. Wall-clock is
  reported but is the least trustworthy output.
- **The batching result is validated for two kinds at scope 3.** A long batched
  context has costs this linear picture does not model; pushing batching far is
  itself an extrapolation.
- **The boot cost is harness-specific.** It will differ with a different system
  prompt or tool surface, which is precisely why it is fitted rather than
  hardcoded.

## 8. Extending it

```bash
python -m examples.calibration_demo     # the findings and the loop, end to end
```

To calibrate your own work: register kinds with a scope unit that means
something for your tasks, run graded probes with replicates, and feed finished
production runs back in as `ScopedRecord(..., provenance=Provenance.PRODUCTION)`.
The store refits, the form can change, and drift is reported rather than
absorbed.
