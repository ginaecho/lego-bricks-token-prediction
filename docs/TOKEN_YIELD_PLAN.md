# Token Yield — Final Plan

> **Supersedes** `TOKEN_YIELD_BUILD_SPEC.md` and `poc-plan.md` for all engineering and evidence
> decisions. Those two remain in the repo as sources; where any of them disagrees with this file,
> **this file wins**.
>
> **What this is.** One document that builds the system *and* decides whether the system's central
> claim is true. The build spec knew how to construct Token Yield but had no way to fail. The POC
> plan knew how to fail honestly but produced only a verdict, not a product. This merges them:
> the spec's structure, artefacts and CLI, with the POC plan's baselines, seals and Stop branches.
>
> **Audit pin:** `ginaecho/lego-bricks-token-prediction` `main` @ `c63702d`. Branch `gc/prototype`
> is synthetic and excluded. The pinned repo is *evidence*, not the build target.
>
> Version 1.0 · 4 September 2026

---

## 0. How to use this document

1. Read it once before writing code. §4 (contract), §5 (data), §7 (metrics and gates) are
   normative. §9 is the execution order.
2. Work **one phase at a time**. A phase does not start until the previous phase's Definition of
   Done is ticked **and** its gate returns `pass` or `narrow`. A gate returning `stop` ends the
   project at that phase — that is a legitimate, planned outcome, not a failure of execution.
3. Every phase ends with runnable commands and passing tests. Where a live stack is unavailable,
   use `MockAdapter` (§6.2) — but everything it produces is labelled `pipeline_only` and can never
   move a gate.
4. **Never invent measurements.** Placeholders are marked `TODO(measure)` and fail CI if they reach
   a report.
5. Every fitted number is written beside the campaign id and the `data_sha256` that produced it.
6. Core package: standard library only. Extras behind optional installs.

### The one-paragraph summary

Token Yield claims that agent cost decomposes into a fixed start-up toll, a linear context term,
and a per-brick marginal — and that a plain-English request can be encoded into bricks well enough
to quote it before it runs. The start-up toll is roughly 89% of a median task. **That means a
predictor that ignores bricks entirely will look excellent on total error.** Everything in this
plan is arranged around that single fact: we measure twice, once on the total and once on the
variable portion with the toll removed, and only the second number is allowed to decide anything.

---

## 1. What we are testing

> Given one pinned model, one owned harness, and one frozen corpus: does brick decomposition
> predict **billable cost** on sealed, unseen work better than the simple baselines — with usable
> intervals — and does batching save money **without producing worse answers**?

Five claims, scored independently so failure is diagnosable:

| Claim | Question | Failure means |
| --- | --- | --- |
| **G0 — runtime** | Is the pinned deployment served, with per-run usage, no cache, no hidden reasoning tokens? | Measurement is impossible. Nothing downstream is interpretable. |
| **G1 — decoder** | Given human-verified brick counts, does the fitted model beat B0–B3 on the *variable* portion? | The composition arithmetic does not hold here. |
| **G2 — encoder** | Can an LLM map unseen request text onto the vocabulary accurately enough to quote? | Quoting is not automatable; bricks may still be a manual scoping aid. |
| **G3 — end-to-end** | Encoder counts → quote, scored against sealed actuals? | The pipeline is not shippable even if both halves work alone. |
| **G4 — batching** | Is one combined request cheaper than *N* separate ones **at equal answer quality**? | Batching is a discount on worse output, not a saving. |

### Three pre-registered outcomes, all of them successes

| Verdict | Meaning | What ships |
| --- | --- | --- |
| **Feasible** | Bricks add real predictive power over baselines, intervals are usable, batching saves at equal quality. | The full `ty` product surface: quote, run, invoice, refit. Expand to a second model. |
| **Narrow** | Start-up, context and total units are predictable, but per-brick structure adds little — or intervals are unusable, or batching is unproven. | A **harness-overhead estimator**. Per-brick prices are hidden. Do not sell project budgeting. |
| **Stop** | No skill over baselines, unusable encoder, drift, integrity failure, or batching that wrecks quality. | The negative result, published. Do not expand model or domain. |

A clean, well-attributed **Stop** is a finished project. It tells you exactly what to fix before
spending more.

---

## 2. Background: what the reference showed, and what it can carry

Source: `docs/composition-findings.md`, MIT, v0.1.0 — 39 runs, fresh memoryless subagent on
`claude-haiku-4-5`, 33 SEC filings.

| Finding | Measured | Can it be a prior for us? |
| --- | --- | --- |
| Null task = 29,821 tokens; three nulls within 0.06% | strong | **No.** That is Claude Code `subagent_tokens` on a runtime we do not own. Our boot will differ by an order of magnitude. It is a *shape* claim (a fixed toll exists), not a *value* claim. |
| Context linear, 0.366–0.41 tokens/byte | strong | **Shape only.** Re-measure the slope. |
| Retrieve 5,384/unit, ~10× any other brick | strong | **Hypothesis to re-test**, and the most interesting one. |
| Review, Classify, Draft, Remediate below noise | strong | **Pre-registered replication check** (§7.4). |
| Sub-additive composition, 64 / 65 / 71% saved | **model-derived counterfactual** | **No.** Never measured as a live batched-vs-separate comparison. Withdrawn as a prior. |
| LOO MAPE 2.55% best vs 6.93% constant | in-harness reconstruction | **No.** Optimistically biased; total-error only; see §2.1. |
| Held-out 2.2% mean, worst 4.7% | 4 compositions, author-selected | **No.** Not a sealed set. |
| Encoder accuracy | **never measured** (3 cases) | **No.** This is the largest open risk in the whole idea. |

Verified directly against the pinned repo while writing this plan:

- `experiments/train_runs.jsonl` line 1 is the null probe at **29,821 tokens**, `context_bytes: 0`.
- Every row carries a **single scalar `tokens`** field with no input/output split. Billable dollars
  cannot be recovered from these rows at all, because output tokens cost roughly 5× input.
- The published form fits **nine** brick coefficients against 35 rows, one of them negative
  (`draft = −137`).

### 2.1 The fixed-cost illusion — the control that shapes this whole plan

With a ~30k intercept and a variable portion of a few hundred tokens, `return 30969` scores an
excellent MAPE. So does almost anything. **Any gate phrased as "mean APE < 10%" is passable by a
constant** and therefore tests nothing.

Every accuracy claim here is scored twice:

- **Total MAPE** on $y$ — the headline number, quoted for realism.
- **Variable-portion WAPE** on $y_{var} = y - \hat{b}$ — the number that decides.

$$\text{vWAPE} = \frac{\sum_i \lvert \hat{y}_{var,i} - y_{var,i} \rvert}{\sum_i \lvert y_{var,i} \rvert}$$

$\hat{b}$ is the median of our own null probes, frozen in `frozen/runtime.json` before any fitting.
$y_{var} \le 0$ stays in the table as a measurement failure; it is **never clamped**.

**Consequence:** if total MAPE passes only because the toll dominates, the vWAPE gate forces
**Narrow**. That is intentional and is the point of the design.

### 2.2 Quarantined priors (CI-enforced)

`64%`, `65%`, `71%`, `2.55%`, `2.2%`, and any encoder-accuracy or interval claim from the reference
campaign may appear **only** in `docs/evidence/excluded-priors.md`. A grep gate fails the build if
they appear in any generated report, README, or deck as a Token Yield property.

---

## 3. Goals and non-goals

### Goals
- **G-A Audit** the reference honestly: reproduce its arithmetic *and* rescore it on the variable
  portion, so we know what we are actually inheriting.
- **G-B Rate card** for our own stack and corpus: noise floor, tokens/byte, per-brick marginals,
  each with an interval and each compared against a baseline.
- **G-C Quote** plain-English requests before they run, with a scored encoder and a calibrated band.
- **G-D Verdict** — Feasible / Narrow / Stop, computed mechanically from a frozen policy, on a
  sealed blind set that nobody has looked at.
- **G-E Demo** — request → bricks → quote → real run → actual → error → per-brick invoice, in five
  minutes, no manual steps, plus the verdict with every gate shown.

### Non-goals (v1.0)
Wall-clock duration; output quality as a *prediction* target (it is measured only as a batching
guard); pricing human approval or wait time; finance-system integration; multi-tenancy, auth,
production hardening; multi-model transfer (Phase 7, conditional); a UI beyond the CLI and one
optional static dashboard.

---

## 4. The locked contract

Frozen at the end of Phase 0, hashed into `frozen/contract.json`. Changing any row invalidates the
campaign and starts a new `campaign_id`.

| Item | Lock |
| --- | --- |
| **Model** | One literal deployment/snapshot id. No aliases, no fallback. Every response asserts the echoed model matches. Snapshot withdrawn mid-campaign → **Stop**, do not retarget. |
| **Sampling** | `temperature=0`, `top_p=1`, fixed for the entire campaign. |
| **Harness** | Fresh, memoryless conversation per run. One frozen system prompt. One frozen tool set: file read + search, scoped to `TY_CORPUS_DIR` only. No shell, no network, no nested subagents. |
| **Cache** | Never enabled. Any non-zero cache token count excludes the row. |
| **Tokens** | Provider `usage` record only. No `len/4`, no local tokeniser, no estimate. Missing usage is a **failed measurement**, never a zero. |
| **Target** | $y = T_{in} + (p_{out}/p_{in}) \cdot T_{out}$ — *billable units*. Dollars $= y \cdot p_{in} \cdot 10^{-6}$. Rates from freeze-day `pricing.json`. **Never hardcode the ratio.** |
| **Vocabulary** | The nine reference bricks, names unchanged. Four **primary** bricks carry the gates (§7.4). Extension only under `allow_extension: true` in Phase 7. |
| **Corpus** | 30+ real documents, 1 KB–40 KB, hashed and manifested before any run. Split into `fit` and `blind` **by source document** before the campaign. |
| **Blind set** | Sealed at Phase 1. Never opened, never labelled, never scored for encoder accuracy, until Phase 5. |
| **Fit data** | New runs only. The reference's 39 rows are never mixed into any fit. |
| **Volume** | ≤ **200** accepted dispatches (§8). Encoder calls on a separate ≤ 250 ledger. |
| **Quality** | Frozen answer keys and a 100-point rubric, graded by humans. **No LLM-as-judge.** |
| **Python** | ≥ 3.11. Core package dependency-free. |

---

## 5. Data contracts (normative)

UTF-8 throughout. JSONL = one object per line, append-only, never rewritten, never deleted.

### 5.1 Evidence class — on every row and every report

| Label | Meaning |
| --- | --- |
| `replay` | Recomputed from the reference's committed JSONL. Reference only. **Can never move a gate.** |
| `pipeline_only` | Fixture or `MockAdapter`. Proves the code path; proves nothing about the model. **Can never move a gate.** |
| `feasibility` | Real dispatch under the locked contract. The only class that counts. |

Enforced in CI:
1. A report mixing classes must state the split per number, or it fails validation.
2. A partial or interrupted run **must not print PASS** and must exit non-zero.
3. Any mandatory gate failure exits non-zero.
4. Every headline number carries a provenance class: `measured` / `derived` / `synthetic` / `missing`.

### 5.2 `vocabulary/bricks.yaml`

```yaml
version: 1
allow_extension: false
primary: [Retrieve, Reconcile, Validate, Extract]   # these carry the gates
bricks:
  - name: Review
    category: cross-cutting          # corrective | adaptive | perfective | preventive | cross-cutting
    bought_as: "Contract and compliance review"
    definition: "Read one or more documents and answer a judgement question about them."
    unit: "One judgement question answered over the supplied documents."
    nearest_neighbour: "Answering a question about a named value is Extract, not Review."
    examples:
      - "Read the filing and state whether revenue guidance was raised, held or cut."
      - "Review this SOW and say whether the acceptance criteria are testable."
  # Extract, Classify, Retrieve, Reconcile, Draft, Remediate, Validate, Report — same shape
```

Validation (`vocabulary.py`): exactly the nine names unless `allow_extension`; every brick has a
non-empty `definition`, `unit`, `nearest_neighbour` and ≥ 2 `examples`; `primary` is a subset of
`bricks`; names unique and case-sensitive. **`nearest_neighbour` is required** — it is the field
that makes inter-rater agreement achievable, and the reference has no equivalent.

Counting rule, normative: **count requested targets** — not documents, not rows, not tool calls.
Reading a document in service of another brick does not add a brick. `out_of_scope` and
`needs_clarification` are first-class all-zero outcomes.

### 5.3 `runs.jsonl`

```json
{
  "run_id": "c001-base-retrieve-s2-r1",
  "campaign_id": "campaign_001",
  "evidence_class": "feasibility",
  "split": "fit | blind | batching | probe",
  "tier": "null | ladder | base | composite | blind | batching | live",
  "timestamp": "2026-09-15T10:32:11+05:30",
  "probe_id": "base-retrieve-s2",
  "replicate": 1,
  "instruction_sha256": "…",
  "instruction": "Find which of the attached filings reports a 26% decline …",
  "units": {"Review": 0, "Extract": 0, "Classify": 0, "Retrieve": 1, "Reconcile": 0,
            "Draft": 0, "Remediate": 0, "Validate": 0, "Report": 0},
  "context_files": ["corpus/acme-10k-2025.txt"],
  "context_bytes": 20315,
  "rendered_prompt_bytes": 21044,
  "adapter": "azure_openai",
  "model_sent": "<deployment>", "model_echoed": "<deployment>",
  "sampling": {"temperature": 0, "top_p": 1, "seed": null},
  "prompt_tokens": 38102, "completion_tokens": 389,
  "cache_read_tokens": 0, "cache_write_tokens": 0, "reasoning_tokens": 0,
  "billable_units": 40047, "cost_usd": 0.040047,
  "elapsed_s": 41.7,
  "tool_calls": 3, "tool_result_bytes": 18422,
  "status": "ok | failed | excluded",
  "exclusion_code": null,
  "accepted": true,
  "output_sha256": "…",
  "runtime_hash": "…", "pricing_hash": "…", "git_sha": "…",
  "notes": ""
}
```

**Invariants.** `billable_units == prompt_tokens + ratio × completion_tokens`, recomputed on load,
never trusted from the file. `cost_usd` computed from `pricing.json`, never taken from the provider
response. Failed runs are kept with `status != ok` and excluded from fitting, **never deleted**.
`units` always carries all nine keys.

**Exclusion codes:** `model_mismatch`, `cache_nonzero`, `reasoning_nonzero`, `truncated`,
`max_turns`, `transport_error`, `path_violation`, `schema_invalid`, `usage_missing`. Transport
errors get a fresh `run_id`; the failed attempt is retained but not fitted.
**Non-transport exclusions above 10% → Stop.**

**Idempotency:** `run_id = sha256(probe_id | context_file_hashes | runtime_hash | pricing_hash | replicate)`.
A killed campaign resumes without duplicates.

**Quote-time features only.** Permitted in any model: `units.*`, total units, `context_bytes`,
`rendered_prompt_bytes`. **Forbidden:** `tool_calls`, `completion_tokens`, `elapsed_s`, anything
observable only after the run.

### 5.4 `coefficients.json`

```json
{
  "campaign_id": "campaign_001",
  "fitted_at": "2026-09-20T18:00:00+05:30",
  "data_sha256": "<sha256 of the exact runs.jsonl lines fitted>",
  "n_runs_fitted": 66,
  "boot_hat": 4812.0,
  "form": "bytes_per_brick_nnls",
  "intercept": 4790.0,
  "bytes_coef": 0.3661,
  "marginals": {"Retrieve": 5384.0, "Reconcile": 1770.0, "…": 0.0},
  "marginals_ols_unclamped": {"Draft": -137.0, "…": 0.0},
  "selection": {
    "method": "grouped_loo_cv",
    "metric": "vWAPE",
    "scores_total_mape":  {"constant": 6.93, "…": 2.55},
    "scores_vwape":       {"constant": 100.0, "…": "TODO(measure)"},
    "lift_vs_best_baseline_pct": "TODO(measure)",
    "bootstrap_lb": "TODO(measure)"
  },
  "conformal": {"level": 0.90, "nonconformity": "relative", "half_width": "…"},
  "noise_floor_cv_pct": 0.6,
  "identifiability": {"condition_number": 12.4, "max_vif": 2.1, "spearman_units_bytes": 0.08},
  "evidence_class": "feasibility"
}
```

### 5.5 Model forms — the baseline ladder **is** the form ladder

This is the key integration. The build spec's six nested forms already *are* a baseline ladder; the
only change is what counts as "better".

| Key | Features | Role |
| --- | --- | --- |
| `constant` | intercept only | **B0** — the fixed-cost illusion, made explicit |
| `units` | intercept + Σ all units | **B1** |
| `bytes` | intercept + context_bytes | **B2** |
| `bytes_units` | intercept + bytes + Σ units | **B3 — the real competitor** |
| `per_brick` | intercept + one coefficient per brick | decoder candidate |
| `bytes_per_brick` | intercept + bytes + per-brick | decoder candidate |
| `bytes_per_brick_nnls` | as above, non-negative least squares | decoder candidate |
| `cell_mean` | mean of each (brick, size) cell | **B5** — diagnostic ceiling, excluded from the gate |

**Selection rule (replaces "strictly beats").** A richer form is chosen only if it beats the best
simpler form by **≥ 15% relative vWAPE** under grouped leave-one-document-out CV **and** the
bootstrap lower bound on that lift exceeds zero. Ties and near-ties go to the simpler form.
`per_brick` beating `bytes_units` by 0.3 pp of total MAPE is not evidence of anything.

**Negative coefficients are reported, never silently clamped.** `bytes_per_brick_nnls` is the
principled way to impose non-negativity; the unclamped OLS values are recorded alongside so the
sign pattern stays visible.

### 5.6 `gold_requests.jsonl`

```json
{
  "request_id": "g017",
  "request": "read the filing, pull three fields, write two auditor checks",
  "gold_units": {"Review": 1, "Extract": 3, "Classify": 0, "Retrieve": 0, "Reconcile": 0,
                 "Draft": 0, "Remediate": 0, "Validate": 2, "Report": 0},
  "context_files": ["corpus/acme-10k-2025.txt"],
  "split": "fit",
  "labellers": ["VB", "AS"], "adjudicator": "RK", "agreed": true,
  "paraphrase_of": null,
  "notes": "'checks' could be Validate or Draft; resolved to Validate by the unit rule"
}
```

### 5.7 Quote and Invoice

```json
// Quote — committed to disk BEFORE dispatch
{"request_id": "blind-003", "quoted_at": "…", "decomposition": {…},
 "context_bytes": 20315,
 "predicted_units": 34804, "predicted_usd": 0.0348,
 "interval": {"level": 0.90, "lo": 31200, "hi": 39900, "method": "split_conformal"},
 "baselines": {"constant": 30969, "bytes_units": 34120},
 "outlier_flags": ["Retrieve: 5,384/unit is 10x other bricks; narrow the search space"],
 "coefficients_ref": "campaign_001@a1b2c3d4", "evidence_class": "feasibility"}

// Invoice — after the run
{"run_id": "…", "actual_units": 33723, "predicted_units": 34804,
 "error_pct": 3.2, "variable_error_pct": 41.7, "interval_hit": true,
 "lines": [
   {"item": "start-up",       "detail": "b̂, measured",                "units": 30969},
   {"item": "context",        "detail": "20,315 bytes × 0.3661",       "units": 7437},
   {"item": "Review × 1",     "units": 52},
   {"item": "Extract × 3",    "units": 1254},
   {"item": "Validate × 2",   "units": 2076},
   {"item": "reconciliation", "detail": "actual − Σ model lines",      "units": -8065}
 ],
 "sum_check": 33723}
```

**Rules.** The invoice always carries a final `reconciliation` line so `Σ lines == actual_units`
**exactly**. The model lines show where the model *thought* the cost was; the reconciliation line
is the residual, and a large one is the most honest signal in the system.

`variable_error_pct` is mandatory and sits beside `error_pct`. In the example above, a 3.2% total
error hides a 41.7% error on the part that bricks are supposed to explain — **that is exactly the
number a reader must not be allowed to miss.**

---

## 6. Components

Layout, module boundaries and conventions follow the build spec; the additions are marked **NEW**.

```
token-yield/
├── vocabulary/bricks.yaml
├── frozen/                        # NEW — hashed, immutable after their phase
│   ├── contract.json  runtime.json  pricing.json  budget.json
│   ├── corpus-manifest.json  split-seal.json      verdict-policy.json
├── token_yield/
│   ├── vocabulary.py  schemas.py  features.py
│   ├── harness/  adapter.py  mock_adapter.py  azure_openai_adapter.py
│   │             probes.py  campaign.py  runner.py
│   ├── costmodel.py  select.py
│   ├── baselines.py               # NEW — B0–B3, B5, implemented before any decoder
│   ├── conformal.py               # NEW — split conformal intervals
│   ├── metrics.py                 # NEW — MAPE, vWAPE, lift, coverage, bootstrap
│   ├── decompose/  prompt.py  parser.py  decomposer.py  keyword.py  evaluate.py
│   ├── quote.py  invoice.py  refit.py  report.py
│   ├── blind.py                   # NEW — sealed-set dispatch and scoring
│   ├── batching.py                # NEW — matched arms, saving decomposition
│   ├── grading.py                 # NEW — blinded rubric scoring, ΔQ
│   ├── verdict.py                 # NEW — mechanical Feasible/Narrow/Stop
│   └── cli.py
├── experiments/  reference/  campaign_001/  decompose_eval/
├── evidence/                      # NEW — the reproducible package
├── corpus/  tests/  demo/  docs/
```

### 6.1 Adapter

```python
class RunResult(TypedDict):
    prompt_tokens: int; completion_tokens: int
    cache_read_tokens: int; cache_write_tokens: int; reasoning_tokens: int
    model_echoed: str; elapsed_s: float; output_text: str; raw: dict

class AgentAdapter(Protocol):
    name: str
    model: str
    def run(self, instruction: str, context_files: list[Path], *,
            temperature: float = 0.0) -> RunResult: ...
```

Fresh conversation every call, identical system prompt, identical tool set. Documents attached by
reading files into the prompt in v1.0 — record the mechanism in `notes`, because it *is* the thing
being measured. `AzureOpenAIAdapter` reads `usage` from the response, retries once on transient
errors, raises `AdapterError` otherwise, and **never estimates tokens**.

### 6.2 MockAdapter — with a negative control **(NEW)**

Deterministic, seeded by `TY_SEED`. Two modes:

| Mode | Generative model | Purpose |
| --- | --- | --- |
| `--mock-mode brick` | `y = b + slope·bytes + Σ marginal·units`, × `(1 + N(0, σ))` | The pipeline must recover planted coefficients within noise. |
| `--mock-mode null` **(NEW)** | `y = b + slope·bytes`, **no brick term at all** | The pipeline must report **no brick skill** — `select_form` must choose `bytes`, the lift gate must fail, and `ty verdict` must print **Narrow**. |

The `null` mode is the most important test in the suite. Without it, a green CI proves only that
the estimator can find a signal that was planted for it. With it, CI proves the system can also
correctly say *"there is nothing here."* A build in which `--mock-mode null` yields Feasible is
broken, regardless of what the other tests say.

Everything either mode produces is stamped `pipeline_only`.

### 6.3 Probes and campaign design

| Tier | What | n |
| --- | --- | --- |
| `null` | "Reply with the single word DONE.", no context, no tools | 6 (3 opening, 3 closing — drift bracket) |
| `ladder` | one instruction over pads at ~1 / 4 / 9 / 20 / 35 KB | 5 |
| `ladder` | **units ⟂ bytes probe (NEW)** — fixed ~16 KB delivered as 1 / 2 / 4 documents | 6 |
| `base` | each of 9 bricks alone × 3 context sizes × 2 replicates | 54 |
| `composite` | 2–4 bricks per instruction, every brick appearing ≥ 2×, ≥ 1 "same brick twice" | 12 |
| `blind` | **sealed** (§9 P1), from blind-corpus documents, arities 1–5 | 24 |
| `batching` | 8 two-task bundles × 3 dispatches + 3 four-task × 5 | 39 |

The base tier makes units and bytes **orthogonal by construction** — that is a genuine strength of
the original design and it is kept. The units⟂bytes probe *verifies* it rather than assuming it: it
fits `y ~ a + b·bytes + c·n_docs` and reports whether document count carries cost independently of
size. If it does, the composite tier must be rebalanced. **Report it; do not smooth it.**

### 6.4 Metrics **(NEW)**

`metrics.py` is pure and fixture-tested against hand-computed values:
`mape`, `vwape`, `relative_lift`, `bias`, `p90_ape`, `wilson_ci`, `paired_bootstrap`,
`coverage`, `noise_floor_cv`, `krippendorff_alpha`.

### 6.5 Conformal intervals **(NEW)**

Heteroscedasticity is checked on training CV residuals first; that check chooses absolute vs
relative nonconformity **before** any held-out data is opened. Split conformal on a calibration
fold gives the 90% band. Jackknife+ over the fit set is a cross-check. `band_pct` derived from
held-out MAPE — the build spec's heuristic — is **retired**; it is not a calibrated interval and
must not be presented as one.

### 6.6 CLI

| Command | Does | Phase |
| --- | --- | --- |
| `ty audit` | Reference arithmetic check **and** variable-portion rescore, side by side, `replay` | 0 |
| `ty vocab validate` / `agreement` | Lint `bricks.yaml`; inter-rater agreement + Krippendorff α | 1 |
| `ty corpus index` / `split --seal` | Manifest with hashes; seal the fit/blind split | 1 |
| `ty design-campaign` / `run-campaign` | Build and execute the probe list | 2 |
| `ty g0` **(NEW)** | Runtime gate: model echo, usage present, cache 0, reasoning 0, cost reconciles | 2 |
| `ty fit` | Baselines first, then decoder candidates; grouped CV on vWAPE; conformal | 3 |
| `ty report` | Rate card, form table with **both** metrics, predicted-vs-actual | 3 |
| `ty decompose` / `eval-decomposer` | Encoder output; score against gold | 4 |
| `ty quote` | Decompose + price + band + baselines | 4 |
| `ty blind` **(NEW)** | Quote all 24 sealed cases, commit quotes, then dispatch, then score | 5 |
| `ty batching` **(NEW)** | Matched separate vs batched arms; saving decomposition; ΔQ | 5 |
| `ty verdict` **(NEW)** | Mechanical Feasible / Narrow / Stop from `frozen/verdict-policy.json` | 6 |
| `ty run` / `refit` | Live quote → dispatch → invoice; refit and version | 6 |
| `ty demo` | The five-minute script | 6 |

All commands take `--adapter mock|azure_openai`, `--campaign <id>`, `--json`. Every command prints
its evidence class. Live dispatch refuses without `--confirm-cost` and a positive budget.

---

## 7. Metrics and gates (frozen before any data is seen)

### 7.1 Acceptance metrics

| ID | Metric | Reference | **PASS** | **NARROW** | **STOP** |
| --- | --- | --- | --- | --- | --- |
| M1 | Noise floor (CV over replicates) | 0.29% | < 1% | 1–3%: widen gates by 3×CV | > 3% |
| M2 | Start-up constant $\hat{b}$ | 29,821 | measured, 3 nulls within 2% | — | drift > 5% start-to-end → campaign invalid |
| M3 | Tokens/byte, R² on the ladder | 0.366 | R² > 0.9 | 0.7–0.9 | < 0.7 |
| M4 | Replicates behind each primary brick | thin | ≥ 2, coefficient interval excludes 0 | interval includes 0 | — |
| M5a | LOO **total** MAPE, best form | 2.55% | ≤ 15% | 15–30% | > 30% |
| M5b | LOO **vWAPE** lift vs best of B0–B3 | not measured | ≥ 20% and bootstrap LB > 0 | positive, < 20% | ≤ 0 |
| M6 | **Blind** total MAPE (24 sealed) | — | ≤ 15% | 15–30% | > 30% |
| M6b | **Blind** vWAPE lift vs best baseline | — | ≥ 15%, LB > 0 | positive | ≤ 0 |
| M7 | Encoder exact match (9-vector) | not measured | ≥ 85% | 70–85% | < 70% |
| M8 | Encoder F1, primary bricks | not measured | ≥ 0.90 Retrieve, ≥ 0.80 each other primary | one primary 0.70–0.80 | any primary < 0.70 |
| M9 | End-to-end quote MAPE | 3 cases | ≤ 20% | 20–35% | > 35% |
| M9b | Interval coverage on blind | — | ≥ 20/24 at 90%, median rel. half-width ≤ 25% | 18–19/24, or width fails → ranges unusable | < 18/24 |
| M10 | Invoice reconciliation | — | **exact**, always | — | any mismatch = bug, not a verdict |
| M11 | Batching saving $s$ | 64–71% *modelled* | ≥ 40% point, 95% LB ≥ 20%, $s>0$ on ≥ 10/12 bundles | positive, below bar | CI includes 0 |
| M12 | Batching quality $\Delta Q$ **(NEW)** | never measured | 90% LB on $\Delta Q > -5$ pts; batched critical failures ≤ separate + 1 | 1 brick fails → drop it, recompute | ≥ 2 bricks fail |
| M13 | Inter-rater agreement | — | exact-vector ≥ 85%, α ≥ 0.80 macro, no brick < 0.70 | — | fail → do not run the campaign |
| M14 | Non-transport exclusions | — | ≤ 10% | — | > 10% |
| M15 | Identifiability | — | cond < 30, max VIF < 5, $\lvert\rho(\text{units,bytes})\rvert \le 0.20$ | collinearity unbroken → **caps verdict at Narrow** | rank-deficient |
| M16 | Demo reliability | — | 3 consecutive clean rehearsals | — | — |

M5a/M5b, M6/M6b and M9 are quoted **as pairs, always, in every artefact**. A report showing total
error without its variable-portion twin fails CI.

### 7.2 Composition — how the verdict is computed

`ty verdict` reads only `frozen/verdict-policy.json` and emits exactly one outcome.

| Prediction | Batching | Verdict |
| --- | --- | --- |
| All point gates pass; intervals usable | ≥ 40% saving, quality non-inferior | **Feasible** |
| Point gates pass | cheaper but quality fails (M12) | **Narrow** — quote; do not recommend batching |
| Point gates pass | untested, α fail, or < 8 bundles | **Narrow** — quoting feasible; batching unproven |
| M6 passes but M6b lift fails; `bytes_units` still meets M6/bias | any | **Narrow** — ship a runtime estimator; **hide per-brick prices** |
| Point gates pass; M9b intervals unusable | any | **Narrow** — point quotes only |
| M6, M6b, or bias fails materially | not run | **Stop** |
| M15 collinearity unbroken | any | **Narrow (cap)** |
| Seal, leakage, drift, or budget integrity fails | any | **Stop — invalid** |

### 7.3 Where Narrow still pays

A **Narrow** verdict is not a consolation. It means start-up cost, context size and total workload
are predictable on this runtime — which is enough to forecast agent spend for capacity planning,
to price a batching lever if M11/M12 hold, and to tell an engineer before they hit Enter roughly
what a run will cost. It is not enough to invoice per brick or to budget a project by decomposing
it. Shipping the smaller true thing is the point.

### 7.4 Pre-registered replication hypotheses **(NEW)**

Written down before measurement, so that confirming *or* refuting them is informative:

- **H1** A fixed start-up toll exists and dominates small tasks. *(shape, expect confirm)*
- **H2** Context cost is linear in bytes with R² > 0.9. *(expect confirm)*
- **H3** Retrieve's marginal is ≥ 3× the next-largest brick. *(the headline claim)*
- **H4** Review, Classify, Draft and Remediate have marginals indistinguishable from noise. *(if
  refuted on our runtime, that is a genuine finding, not an error)*
- **H5** Batching a bundle saves ≥ $(N-1)\hat{b}$ in units. *(the mechanism; if the residual
  saving beyond this is ≈ 0, sub-additivity is entirely the toll, and the 64–71% figure was an
  artefact of a large boot)*

H5 matters most. Our owned harness will likely have a much smaller $\hat{b}$ than 29,821, so the
**percentage** saving may be far below 40% even if the mechanism is completely real. The saving is
therefore always decomposed into *toll paid once* (predicted) and *residual sub-additivity*
(measured − predicted). That decomposition distinguishes "the bar was missed" (Narrow) from "there
is no saving" (Stop).

---

## 8. Budget

| Use | Dispatches |
| --- | --- |
| G0 smoke | 8 |
| null + ladder + units⟂bytes probes | 17 |
| base tier (9 bricks × 3 sizes × 2 replicates) | 54 |
| composite tier | 12 |
| **blind — hard fence** | **24** |
| batching: 8 two-task × 3 | 24 |
| batching: 3 four-task × 5 | 15 |
| reserve (infrastructure replacements) | 26 |
| live demo rehearsals | 6 |
| **Total** | **186 / 200** |

Encoder calls are on a **separate ≤ 250 ledger** — they are small, they are not the experiment, and
mixing them corrupts the campaign accounting.

**Token cost.** Unknown until $\hat{b}$ is measured. At a reference-like ~30k boot, 200 runs ≈ 6M
tokens; on a lean owned harness it may be under 1M. Record actual spend in
`experiments/campaign_001/report.md` and reconcile to the provider billing export within **1%**,
or the dollar claims are void.

**Why a fixed cap.** It is not primarily a cost control — the campaign is cheap. It is an
**integrity control**: a pre-registered dispatch count is what stops the study quietly becoming a
search for a flattering result.

**Overrun rule.** Fence the blind 24 first. Then fund the 8 two-task bundles. Then four-task. If
fewer than 8 two-task bundles fit after fence and reserve, batching is recorded **untested** and
the verdict caps at Narrow. **Never cut the blind set to fund batching.**

---

## 9. Phases

Each phase: tasks → **DoD** → **Gate**. A gate returning `stop` ends the project there.

### Phase 0 — Evidence audit and contract freeze · $0 · days 1–3

- [ ] Clone the reference at `c63702d`; run its suite; record the commit hash in
      `experiments/reference/SOURCE.md`. Copy `train_runs.jsonl` and `decompose_cases.jsonl` into
      `experiments/reference/`.
- [ ] Read `trainsuite.py`, `costmodel.py`, `models.py`, `decompose.py`; write the field alias map
      in `schemas.py`.
- [ ] Scaffold the repo (§6), `pyproject.toml`, `ty` console script, pytest config, CI.
- [ ] Implement `vocabulary.py`, `schemas.py`, `features.py`, `baselines.py`, `costmodel.py`,
      `select.py`, `metrics.py`, `verdict.py`.
- [ ] **`ty audit`** — reproduce the reference's six-form total-MAPE table **and, beside it, the
      same six forms scored on the variable portion** with the reference's own null probe as
      $\hat{b}$. Both labelled `replay`.
- [ ] `docs/evidence/claim-ledger.json`: every reference number with formula, n, and provenance
      class. `docs/evidence/excluded-priors.md` + the CI grep gate (§2.2).
- [ ] Freeze `frozen/contract.json`, `verdict-policy.json`, schemas, metrics. Boundary fixtures
      must return Feasible / Narrow / Stop exactly as §7.2.
- [ ] `docs/readiness.md` — the paid-run checklist, all red.

**DoD:** `pytest -q` green with no API key. `ty audit` prints both tables. `ty verdict` passes its
boundary fixtures. Zero API calls in the entire phase.

**Gate — the reframe that matters.** Reproducing 2.55% is a **transcription check**, not a
go/no-go. It proves we read the reference correctly. It cannot authorise anything, because it is
in-harness reconstruction on total error. The variable-portion column beside it is expected to look
far worse, **and that is the honest picture of what we are inheriting.** If bricks add nothing even
on the author's own labels, Phase 4's encoder work is rescoped as a control study — but the new
campaign still runs, because a different runtime is a different experiment.

### Phase 1 — Vocabulary, corpus, seal · $0 + labelling · days 3–8

- [ ] Write `vocabulary/bricks.yaml`: nine reference names unchanged, plus `unit`,
      `nearest_neighbour`, ≥ 2 examples each. Mark the four primary bricks.
- [ ] Collect 30+ real documents, 1 KB–40 KB, from the target workload. `corpus/README.md` records
      source, licence and sizes. `ty corpus index` hashes everything into
      `frozen/corpus-manifest.json`.
- [ ] **`ty corpus split --seal`** — partition by *source document* into `fit` and `blind`. Write
      `frozen/split-seal.json`. Loaders **raise** on any attempt to read blind documents outside
      `ty blind`.
- [ ] Author `gold_requests.jsonl`: 30–50 requests spanning all nine bricks, arities 1–5, including
      deliberately ambiguous phrasing and paraphrase pairs. Blind requests are authored here too,
      then sealed unopened.
- [ ] Two labellers independently decompose; a third adjudicates. `ty vocab agreement` reports
      exact-vector match, per-brick Krippendorff α, and MAE.
- [ ] Disagreements resolved by **tightening `nearest_neighbour` and `unit`**, never by majority
      vote. Commit the diff.

**DoD:** validation passes; the seal file exists and the loader guard is tested; M13 met.

**Gate:** M13 fails → **Stop before spending a single dispatch.** If two humans with the card in
front of them cannot agree what a brick is, an LLM cannot, and the encoder claim is dead. Fallback
where only one labeller exists: same person, ≥ 72 h apart, reshuffled, reported explicitly as
*intra*-annotator on every scorecard.

### Phase 2 — Harness, G0, campaign · weeks 2–3

- [ ] Implement `adapter.py`, `mock_adapter.py` (both modes), `azure_openai_adapter.py`.
- [ ] Implement `probes.py`, `campaign.py`, `runner.py`; `ty design-campaign`, `ty run-campaign`.
- [ ] Full campaign end-to-end on `--adapter mock --mock-mode brick` (recovers planted
      coefficients) **and** `--mock-mode null` (correctly reports no brick skill). Both in CI.
- [ ] **`ty g0`** — ≤ 8 live calls. Requires all of: pinned model echoed; `usage` present and
      itemised; cache tokens 0; reasoning tokens 0; cost arithmetic reconciles; replicate input CV
      < 1%.
- [ ] Run nulls, ladder, units⟂bytes probes, base and composite tiers. Bracket with nulls.
- [ ] `experiments/campaign_001/report.md`: $\hat{b}$, noise floor, tokens/byte with R², drift
      between opening and closing nulls, units⟂bytes finding, actual spend.

**DoD:** ≥ 89 `ok` runs across null/ladder/base/composite; M1, M2, M3 met; every primary brick has
≥ 2 replicates; blind corpus untouched (assert via the seal).

**Gate — G0.** Fails → **Stop**. Note this is *not* the build spec's "boot within 10% of ~29,800"
— that test would fail by construction on any harness that isn't Claude Code, for reasons that have
nothing to do with whether the idea works. G0 asks only: **is measurement possible?** A $\hat{b}$
far below 29,821 is the *expected* result and is documented as a runtime artefact, not a failed
replication.

Also gate on drift (M2) and noise (M1). Noise floor > 3% → **Stop**: cheap bricks cannot be
distinguished from noise at any sample size we can afford.

### Phase 3 — Fit, baselines, selection, intervals · weeks 3–4

- [ ] **Baselines first.** `ty fit` computes B0–B3 and B5 *before* any decoder candidate is fitted.
- [ ] Grouped leave-one-document-out CV on **vWAPE**; selection by the ≥ 15% lift + bootstrap-LB
      rule (§5.5). Report both metrics for every form.
- [ ] Identifiability: condition number, VIF, Spearman(units, bytes). Fail → do not fit per-brick
      coefficients; fall back to `bytes_units` and cap at Narrow.
- [ ] Conformal calibration; freeze the 90% band.
- [ ] Write `coefficients.json`. `ty report` → `docs/rate-card.md` with the form table showing
      **total MAPE and vWAPE side by side**, per-brick marginals with intervals, and the
      pre-registered H1–H5 marked confirmed / refuted.

**DoD:** M4, M5a, M5b, M15 evaluated and recorded; conformal band frozen; `--mock-mode null` still
yields "no skill" in CI.

**Gate:** M5b ≤ 0 → **Narrow** — the runtime estimator is what ships; skip to Phase 5 with the
`bytes_units` model and do not build the invoice's per-brick lines. M5a > 30% → **Stop**.

### Phase 4 — Encoder and its evaluation · weeks 4–5

- [ ] `decompose/prompt.py` renders the vocabulary, unit rules, `nearest_neighbour` guidance, 3–4
      worked examples, the corpus index, and the strict-JSON contract.
- [ ] `parser.py`: strict JSON, all nine keys, non-negative integers, `context_files` must exist.
      One repair retry, counted; then fail loudly. **No silent heuristic fallback** — the keyword
      encoder is a separate scored arm, never a substitute.
- [ ] `keyword.py` **(NEW)** — deterministic regex encoder, written from the card **before** gold
      is opened. If it matches the LLM within confidence intervals, **ship it**: it is free and
      deterministic.
- [ ] `evaluate.py`: exact match, per-brick P/R/F1, count MAE, paraphrase stability, confusion
      pairs, format-failure rate. `ty eval-decomposer`.
- [ ] **Priced error (NEW):** score the encoder in *dollars*, not just accuracy — feed predicted
      and gold vectors through the frozen Phase 3 model and compare. A one-unit Retrieve confusion
      costs 100× a Classify confusion; exact-match treats them as equal.

**DoD:** M7, M8, M9 recorded on the fit-split gold set; confusion pairs documented; both encoder
arms scored.

**Gate:** M7 < 70% **and** priced error > 40% → **Stop** on automated quoting. Bricks may still
ship as a manual scoping vocabulary — record that as the Narrow variant and continue to Phase 5
with human-supplied vectors so the decoder claim can still be settled.

### Phase 5 — Blind test, then batching · the decisive spend · weeks 5–6

**Order is binding: blind before batching.** A prediction failure means batching is not run at all.

**5a — Blind (24 dispatches).**
- [ ] `ty blind quote` — commit all 24 quotes to disk **first**, with hashes. The runner refuses any
      task without a committed quote and enforces
      `quoted_at < dispatched_at < recorded_at`.
- [ ] `ty blind run` — dispatch once. No repeats except infrastructure replacements. **More than 3
      exclusions of 24 voids the blind set.** Bad answers, timeouts and tool loops are *outcomes*,
      not grounds for replacement.
- [ ] `ty blind score` — M6, M6b, M9, M9b, bias, P90 APE. Coverage reported with a Wilson CI;
      **never** claim "90% proven" from n=24. No refit, ever.

**Gate:** M6 or M6b fails materially → **Stop**; do not spend the batching budget. M6b lift-only
failure → **Narrow estimator**; continue to 5b, but the demo must not display per-brick rates.

**5b — Batching × quality (39 dispatches).**
- [ ] Bundles constructed and frozen **before** dispatch: 8 two-task (primary) + 3 four-task
      (scaling), on fit-corpus documents only. The separate arm is **re-run**, not reused.
- [ ] Per-task instructions and source access byte-identical across arms; the batched arm may add
      only neutral delimiters and "answer all sections". Fresh session per dispatch.
- [ ] $s = (y_{sep} - y_{batch}) / y_{sep}$, decomposed into **toll-paid-once** (predicted,
      $(N-1)\hat{b}$) and **residual sub-additivity** (measured − predicted).
- [ ] **Blinded grading.** ≥ 2 graders + adjudicator, engaged **before** generation. 100 points:
      factual 45 / completeness 25 / citations 15 / no fabrication 10 / structure 5. Critical
      failure = wrong numeric, fabricated fact, missed central reconciliation, unusable result.
      Grade files carry no arm label until join. α ≥ 0.67 required before unblinding.
- [ ] Paired bootstrap (10,000) with the **bundle** as the unit of analysis.

**DoD:** M11 and M12 recorded; failure modes logged (last-position dropout, cross-task
contamination, truncation).

**Gate:** M12 fails on ≥ 2 bricks → batching is **not recommended** regardless of how large $s$ is.
A cheaper worse answer is not a saving.

### Phase 6 — Verdict, invoice, demo · week 6

- [ ] `ty verdict` — reads `frozen/verdict-policy.json` only, prints every gate with its bar and
      observed value, emits exactly one of Feasible / Narrow / Stop, exits non-zero on Stop.
- [ ] `invoice.py`, `refit.py`, `coefficients_history.csv`, `ty run`, `ty refit`. Under **Narrow**,
      per-brick lines are suppressed and the invoice shows start-up + context + reconciliation only.
- [ ] `ty demo` + `demo/run_demo.sh` (§10). Replay is offline and byte-identical across runs except
      for a disabled clock; `--live` requires `--confirm-cost`, is non-ledger, and is bannered
      *"single sample, not evidence."*
- [ ] `evidence/` package: `ty evidence build` regenerates every derived artefact into
      `evidence/v1/` with `checksums.sha256`. A fresh clone with **no credentials** must reproduce
      the verdict.
- [ ] Rehearse three times; log each in `demo/rehearsals.md`.

**DoD:** M10 exact on every invoice; M16 met; fault injection flips only the gate it was fed.

### Phase 7 — Conditional extensions · only on Feasible

- [ ] Second-model replication as `campaign_002`: nulls + base + a composite subset. Compare
      *shape* — fixed toll, linear context, Retrieve dominance, sub-additivity — not values.
- [ ] Workflow brick under `allow_extension: true`: one primitive whose work is a system
      interaction (DB query, API call, simulated approval wait), ≥ 2 replicates, refit.
- [ ] Optional static dashboard: quote vs actual over time, per-brick spend, batching savings.

**Under Narrow:** do none of this. Ship the estimator, publish the scorecard, stop.

---

## 10. Demo runbook (`ty demo`, ≤ 5 minutes, no manual steps)

| Step | Command | Proves | Time |
| --- | --- | --- | --- |
| 1 | `ty report --rate-card` | Cost is measurable and stable: $\hat{b}$, tokens/byte, marginals **with intervals**, noise floor | 45 s |
| 2 | `ty report --forms` | The honest comparison — every form's total MAPE **beside** its vWAPE, so the audience sees what bricks actually bought | 45 s |
| 3 | `ty quote "<typed request>"` | Quotable before dispatch: bricks, point estimate, 90% band, and the baselines it beat | 60 s |
| 4 | `ty run "<same request>"` | Actual beside the quote, error %, interval hit/miss, and hashes proving the quote preceded the run | 90 s |
| 5 | `ty batching --bundle-id B006` | Matched arms, saving split into toll-once vs residual, blinded ΔQ — or an explicit *untested* | 30 s |
| 6 | `ty verdict` | One of Feasible / Narrow / Stop from the frozen policy, every gate listed | 30 s |

Step 2 is non-negotiable and is the step that distinguishes this demo from a sales pitch. Currency
uses `TY_PRICE_PER_1K_INPUT` / `TY_PRICE_PER_1K_OUTPUT`; unset → print units only.

---

## 11. Testing strategy

- **Unit, pure, offline:** vocabulary validation; design matrices for all eight forms; OLS and NNLS
  recover planted coefficients to 1e-6; grouped CV on tiny synthetic sets; conformal quantiles;
  metrics against hand-computed fixtures; parser rejects malformed JSON; invoice `sum_check`;
  verdict boundary fixtures.
- **Negative controls (NEW):** `--mock-mode null` must yield "no brick skill" and a Narrow verdict.
  Shuffled-$y$ CV must collapse the decoder to the B1 baseline. A blind-set import from any
  non-`blind` module must raise.
- **Seal tests (NEW):** blind documents unreadable outside `ty blind`; dispatch without a committed
  quote refuses; mutating any frozen field dispatches **zero** API calls.
- **Audit gate:** `ty audit` reproduces the reference table within 0.05 pp, labelled `replay`.
- **Pipeline, offline:** full campaign → fit → quote → run → invoice → refit on `MockAdapter`,
  everything stamped `pipeline_only`.
- **Contract tests:** every JSON the CLI writes validates against §5 (hand-written validators in
  `schemas.py`; no JSON-schema dependency).
- **CI:** `pytest -q`, `ty audit`, and both mock modes on every push. The real adapter is **never**
  called in CI.

---

## 12. Conventions

Python 3.11+, `from __future__ import annotations`, type hints everywhere, dataclasses for records.
Pure functions in `features.py`, `baselines.py`, `costmodel.py`, `select.py`, `metrics.py`,
`conformal.py`, `verdict.py`; I/O only at the edges. No hidden state. Log to stderr, results to
stdout, `--json` for machine-readable output. Docstrings state the **unit** of every number (tokens,
billable units, bytes, dollars, percent). Never estimate tokens, never delete a run, never overwrite
`coefficients.json` (suffix with `data_sha256[:8]`), never clamp a coefficient silently, never print
a total-error number without its variable-portion twin. Commit prefixes `phase0:` … `phase7:`.

---

## 13. Risks

| Risk | Handling |
| --- | --- |
| **Fixed-cost illusion** | vWAPE mandatory everywhere; B0 *is* the illusion, named as a baseline; `--mock-mode null` proves the pipeline can say "nothing here". |
| Noise floor ≫ 0.29% | `temperature=0`; more replicates; gates widen to 3×CV; only claim marginals whose interval excludes 0. |
| Shape does not transfer | Framed as replication with H1–H5 pre-registered; the report states the result either way. |
| Harness artefact | We predict *this* prompt and tool set. Our $\hat{b}$ is ours, not 29,821. |
| **Encoder is the fundability risk** | Own phase; priced error, not just accuracy; keyword control; no silent fallback. |
| Circularity | The same model family encodes and executes. The keyword encoder is the control. Do not claim independent validation. |
| Small owned boot | The 40% batching bar may fail even if the mechanism is real — hence the toll-vs-residual decomposition. |
| Quality confound on batching | Separate arm, blinded grading, ΔQ gate, critical-failure count. |
| Provider hides usage | G0 on day one of Phase 2; fall back to billing export; blocker recorded in `docs/decisions.md`. |
| Corpus unrepresentative | Drawn from the target workload; re-run the base tier when the workload changes. |
| Grader availability | Single point of failure for the quality arm; contract graders **before** generation. |
| Blind set contamination | Sealed by loader guard, hashed, never labelled; > 3 exclusions voids it. |
| Scope creep | Every addition carries a metric in §7 and a DoD in §9, or it does not happen. |

Ask the reference author once for the original runner, traces and corpus. If unavailable, mark the
2026 campaign permanently non-reproducible and move on. Not a blocker.

---

## Appendix A — The nine bricks

| Brick | Category | Bought as | Reference marginal | Primary? |
| --- | --- | --- | --- | --- |
| Review | cross-cutting | contract and compliance review | 52 (below noise once bytes modelled) | |
| Extract | cross-cutting | invoice and claims intake, KYC | 418 | ✅ |
| Classify | cross-cutting | ticket and email triage | below noise | |
| Retrieve | cross-cutting | knowledge discovery, e-discovery | 5,384 | ✅ |
| Reconcile | corrective | financial close, audit, dispute resolution | 1,770 | ✅ |
| Draft | adaptive | proposals, memos, marketing copy | below noise | |
| Remediate | corrective | exception handling, error correction | below noise | |
| Validate | preventive | control testing, quality assurance | 1,038 | ✅ |
| Report | perfective | management and board reporting | 838 | |

Primary bricks carry the gates because they are the four the reference measured above its noise
floor. The other five are still measured, still fitted, and still scored — H4 predicts they stay
below noise, and refuting that would be a real finding.

## Appendix B — Reference quick commands

```bash
git clone https://github.com/ginaecho/lego-bricks-token-prediction
cd lego-bricks-token-prediction && git checkout c63702d
pip install -e .
python -m examples.composition_demo
python -m pytest tests/test_compose.py -q
```

Key files: `token_yield/{trainsuite,decompose,costmodel}.py`,
`experiments/{train_runs,decompose_cases}.jsonl`,
`docs/{composition-findings,calibration-findings}.md`.

## Appendix C — Reference fixtures (`replay` only, never a prior)

Apart vs together (tokens, **modelled**): Review+3×Extract+2×Validate 97,646 → 34,804 ·
Review+2×Remediate+2×Validate 96,542 → 33,699 · Retrieve+Review+Remediate+Validate 132,235 → 37,971.

Held-out (actual / predicted): Review+2×Validate 33,174 / 33,549 · 2×Draft+Report 32,878 / 31,986 ·
6×Review+Draft 38,732 / 38,583 · Retrieve+Review+Remediate+Validate 36,283 / 37,971.

Context ladder (Review): 0 B → 29,821 · 861 → 32,174 · 3,179 → 33,163 · 8,973 → 34,051 ·
20,315 → 38,491 · 34,190 → 43,924.

These exist **solely** as fixtures for `ty audit`. They are the author's measurements on the
author's runtime, and none of them is a prediction about ours.

---

## Appendix D — What changed from the two source documents, and why

| # | Source conflict | Resolution |
| --- | --- | --- |
| 1 | Build spec gated Phase 0 on reproducing 2.55% LOO MAPE | Kept as `ty audit`, a **transcription check** labelled `replay`, now printing the variable-portion rescore beside it. It authorises nothing. |
| 2 | Build spec target was `total_tokens` | **Billable units** — output is ~5× input. The spec's schema already split prompt/completion, so this was one line of arithmetic away. |
| 3 | Build spec had **no Stop branch anywhere** | Every phase gate now returns pass / narrow / stop; `ty verdict` is mechanical and exits non-zero on Stop. |
| 4 | Build spec's held-out = 5 author-chosen compositions | **24 sealed cases on a document-split blind corpus**, quoted before dispatch, hashes proving order. |
| 5 | Build spec's "richer form must strictly beat simpler" | **≥ 15% relative vWAPE lift with bootstrap LB > 0.** A 0.3 pp total-MAPE win is not evidence. |
| 6 | Build spec clamped negative marginals to 0 | **NNLS as an explicit form**, with unclamped OLS reported beside it. |
| 7 | Build spec's `band_pct` = held-out MAPE rounded up | **Split conformal**, with the nonconformity type chosen before any held-out data is opened. |
| 8 | Build spec measured batching cost only | **Cost × blinded quality**, with the saving decomposed into toll-once and residual. |
| 9 | POC plan cut to 4 bricks | **All nine kept** — they are the product vocabulary and the encoder needs them — with four **primary** bricks carrying the gates. |
| 10 | POC plan had no product surface | **Invoice with reconciliation line, refit history, `ty` CLI** all retained from the build spec. The invoice is the best idea in either document. |
| 11 | POC plan's G0 inherited "boot within 10% of 29,800" | **Redefined:** G0 asks whether measurement is possible, not whether our boot matches someone else's runtime. |
| 12 | Neither document had a negative control | **`--mock-mode null`** — CI must prove the system can correctly report "no skill". |

**Net effect.** The build spec alone would have produced a working product around an unverified
claim, with a green CI that only ever proved the estimator could find a signal planted for it. The
POC plan alone would have produced a defensible verdict and no system. This produces both — and if
the verdict is Stop, it produces that cheaply, at Phase 3 or 5, before the product surface is built.
