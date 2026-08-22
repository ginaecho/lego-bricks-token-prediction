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

| kind | scope unit | repos | runs |
|---|---|---|---|
| `comprehension` | files read (and bytes) | 3 | 15 |
| `code_write` | functions written | 1 | 4 |
| `test_write` | tests written | 1 | 3 |
| `code_review` | functions reviewed | 1 | 3 |
| `docs` | functions documented | 1 | 3 |

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

### 3.4 The scope unit itself was wrong

The three findings above came from probing **one** repository. Repeating the
identical graded task against `psf/requests` and `pallets/click` broke the
model — and then fixed it.

At scope 8 (eight files read) the three repositories cost 57,732 · 59,851 ·
**149,655** tokens. Same file count, 2.6× the cost, because click's eight files
are 272 KB against requests' 60 KB.

Fitting each repository separately:

| scope unit | harness-dose | requests | click | one pooled model |
|---|---|---|---|---|
| **files** | 2,305 /file | 3,265 /file | 15,794 /file | **19.8% MAPE** |
| **bytes** | 0.418 /byte | 0.389 /byte | 0.419 /byte | **2.8% MAPE** |

Measured in files the slope disagrees 7× between repositories, so a model
fitted on one cannot price another. Measured in bytes the slopes agree to
within 7% and the intercepts to within 2%, and a **single model fits all three
repositories at 2.8% — inside the noise floor**:

```
tokens = 36,044 + 0.4174 × bytes          (≈ 2.4 bytes per token)
```

This is the finding that most justifies the whole approach. A hardcoded scope
unit would have hidden it exactly as the hardcoded multipliers hid the first
three. So the *signal* is now selected the same way the shape is: every record
carries both candidates, and cross-validation picks. On the shipped data the
selector chooses `affine@bytes` at 3.6% over `affine@scope` at 24.0% — it was
never told to prefer bytes.

## 4. Method: the shape and the signal are fitted, not chosen

`token_yield/costmodel.py` fits four forms and selects by leave-one-out
cross-validation, breaking ties toward fewer parameters:

Both the model form and the explanatory signal are chosen by leave-one-out
cross-validation over every (form, signal) pair.

| form | LOO MAPE, comprehension (on `bytes`) | LOO MAPE, code_write |
|---|---|---|
| **affine** | **3.6%** | 8.1% |
| **constant** | 25.6% | **4.7%** |
| power | 14.9% | 7.8% |
| proportional | 81.2% | 90.0% |

`proportional` *is* the old `1×/2×/4×` rule. It finishes last in both.

The selector's output on the shipped data — note that each kind picked a
different explanatory signal, unprompted:

```
comprehension : tokens = 37,326 + 0.4131 × bytes    (n=15, 3 repos)
code_write    : tokens = 38,883                     (scope carried no signal)
test_write    : tokens = 43,424 + 1,161  × scope     (output-driven)
docs          : tokens = 43,319 +   496  × scope     (output-driven)
code_review   : tokens = 40,767 +     5  × scope     (effectively flat)
```

`comprehension` is driven by what it *reads*; `test_write` and `docs` by what
they *write*; `code_review` by neither, because it reads a fixed file and emits
three sentences regardless.

Cross-validation rather than in-sample fit matters here: a more flexible form
must not win merely by having more parameters to bend. And the selection is
genuinely two-sided — given through-origin data it picks `proportional`, given
flat data it picks `constant`. Tests pin all three directions.

## 5. Validation

**Skill ratio.** `backtest.py` reports cross-validated error as a multiple of
the noise floor. `comprehension` sits at 0.79× over 15 runs and three
repositories, `code_write` at 0.71× — both at the limit the process allows. The actionable reading: more measurement
of these kinds will not help; reducing run-to-run variance would.

**Out-of-sample.** The strongest claim. Per-kind models were fitted *only* on
single-kind runs; the batched runs were held out entirely. Predicting them:

| | predicted | measured | error |
|---|---|---|---|
| two separate agents | 82,495 | 81,697 | 1.0% |
| one batched agent | 43,612 | 43,491 | **0.3%** |

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

## 6b. Mining: what work a project is actually made of

Probes price a unit of work. They cannot say what work a project *contains*.
`mine.py` reads a repository's history, classifies each commit into a kind
through transparent ordered rules — each carrying the evidence that fired it —
and reports the distribution.

Across 419 commits from the three repositories:

| kind | share | median lines | mean confidence |
|---|---|---|---|
| docs | 24% | 7 | 0.94 |
| code_change | 23% | 29 | 0.30 |
| refactor | 21% | 6 | 0.46 |
| feature | 14% | 47 | 0.66 |
| bug_fix | 11% | 34 | 0.75 |
| test_write | 8% | 31 | 0.94 |

Commits matching no strong rule are labelled `code_change` at low confidence
rather than forced into a category — 23% of real commits do not announce what
they are, and pretending otherwise would put a made-up label under a budget.

Crossing that distribution against the kinds actually measured produces a
**measurement backlog**:

> Coverage: 8% of mined work is a kind we have measured
> Backlog: probe `docs` → +24%, `code_change` → +23%, `refactor` → +21% …

Acting on it moved coverage from **8% to 32%** in three probes. That is the
loop closing: mining says what to measure, probing measures it, and coverage
records the gap that remains. Mining supplies the *what and how much*; only
probes and production runs supply the *how expensive*.

## 7. Limits

- **28 runs, five kinds, three repositories, one harness, one model.** Every
  fitted model records the range it was fitted over, and `plan.py` flags
  extrapolation beyond it rather than silently answering.
- **Only 32% of mined real work is a kind that has been measured.** The other
  68% cannot be priced, and the framework refuses to price it rather than
  guessing. `bug_fix`, `feature` and `refactor` — the kinds most people care
  about budgeting — are all still unmeasured.
- **Three kinds rest on three points each** (`docs`, `test_write`,
  `code_review`) with no replicates, so their skill against the noise floor is
  unknown and reported as such. Their intervals are floored at the measured
  noise level rather than trusting a near-perfect fit to three points.
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
python -m examples.mining_demo          # mine repos, classify, see the gap
```

To calibrate your own work: register kinds with a scope unit that means
something for your tasks, run graded probes with replicates, and feed finished
production runs back in as `ScopedRecord(..., provenance=Provenance.PRODUCTION)`.
The store refits, the form can change, and drift is reported rather than
absorbed.
