# Token Yield — Build Specification for GitHub Copilot

> ⚠️ **SUPERSEDED — do not build from this file.**
> `docs/TOKEN_YIELD_PLAN.md` is the single source of truth. It keeps this document's structure,
> artefacts, schemas and CLI, and replaces its gates: the Phase 0 reproduction target is no longer
> a go/no-go, accuracy is scored on the variable portion as well as the total, held-out becomes a
> sealed blind set, batching gains a quality arm, and every phase gains a Stop branch.
> Kept here as a source document and for the sections the final plan cites verbatim.

> **Purpose of this file.** This is the single source of truth for building Token Yield: a system that
> predicts the token cost of AI-agent work *before* it runs by decomposing the work into reusable
> task "bricks", and reconciles that quote to a per-brick invoice afterwards. It is written so that
> GitHub Copilot (agent mode / coding agent) can build the whole project from it, phase by phase.
>
> **Companion documents:** `Token Yield Implementation Plan.docx` (narrative plan for stakeholders)
> and `Token Yield - Pitch Deck.pptx`. This file supersedes both for engineering decisions.
>
> Prepared by Vinay Pratap Singh Bhadauria · 4 September 2026 · Version 0.1

---

## 0. How Copilot should use this document

1. Read the whole file once before writing any code. Section 6 (data contracts) and Section 7
   (component specs) are normative; Section 9 (phased task list) is the execution order.
2. Work **one phase at a time** (Section 9). Do not start a phase until the previous phase's
   *Definition of Done* checklist is fully ticked and its tests pass.
3. Every phase ends with runnable commands and passing tests. If a test cannot pass because a
   real agent stack is not configured, use the `MockAdapter` (Section 7.2) — it exists so the
   entire pipeline is testable offline.
4. **Never invent measurements.** All numbers in fixtures come from the reference repository or
   from real runs recorded by the harness. Placeholders are marked `TODO(measure)`.
5. Keep every number reproducible: fitted coefficients are always written next to the campaign
   identifier and data hash that produced them.
6. Follow the coding conventions in Section 12. Prefer small, typed, pure functions; no runtime
   dependencies beyond the Python standard library for the core package (the reference has none).
   Optional extras (`azure`, `dashboard`) may add dependencies.

---

## 1. Background (what we are reproducing and extending)

### 1.1 The idea

A cost model needs a unit of work, and "one task" is not a unit. Token Yield fixes a vocabulary of
**nine base tasks (bricks)** that real enterprise work decomposes into:

`Review · Extract · Classify · Retrieve · Reconcile · Draft · Remediate · Validate · Report`

The method is a closed loop:

```
① measure the basic bricks  →  ② measure their combinations
            ↓                              ↓
        ③ train the predictor on (1) + (2) as features
            ↓
④ new project → decompose into bricks → recompose → predicted tokens
            ↺  measured actual refits the model
```

The fitted model in the reference implementation is a single linear equation:

```
tokens = 30,969 + 0.3661 × context_bytes + Σ_brick marginal[brick] × units[brick]
```

The decomposition step is framed as an **autoencoder over tasks**: a free-text request is
*encoded* (by an LLM) into brick counts, and the brick counts are *decoded* (by the fitted model)
into a token prediction. Because the round trip ends in a number, the reconstruction error is
measurable the moment the task is actually run.

### 1.2 Reference results we must reproduce first

Source: `docs/composition-findings.md` in
<https://github.com/ginaecho/lego-bricks-token-prediction> (MIT licence, release v0.1.0).
39 agent runs, each a fresh memoryless subagent on `claude-haiku-4-5`, over 33 SEC filings.

| Finding | Measured value |
|---|---|
| Start-up is a fixed toll | Null task ("reply DONE") = **29,821 tokens**; three null runs within 0.06%; 89% of a median task |
| Context cost is linear | **0.41 tokens/byte** on the Review ladder; **0.366** fitted over the campaign; 0.406 on source code |
| Retrieve is the outlier | **5,384 tokens/unit**, ~10× any other brick |
| Four bricks are free at the margin | Review, Classify, Draft, Remediate: marginal below run-to-run noise |
| Composition is sub-additive | Running bricks together vs apart saved **64%, 65%, 71%** on three compositions |
| Model selection | Six nested forms scored by leave-one-out CV; best (bytes + per-brick) **2.55% MAPE**; constant 6.93% |
| Generalisation | Held-out compositions **2.2% mean error**, worst 4.7% (the four-way mix, outside training arity) |
| Noise floor | **0.29%** |

Reference marginals (tokens per unit): Retrieve 5,384 · Reconcile 1,770 · Validate 1,038 ·
Report 838 · Extract 418 · Review 52 (once bytes are in the model) · Classify, Draft, Remediate
below noise.

### 1.3 Gaps the reference author states (and this build closes)

| Stated limitation | Where we address it |
|---|---|
| One model only (`claude-haiku-4-5`); the constant is model-specific | Phase 2 measures on our model; Phase 6 replicates on a second model |
| Small campaign (39 runs, 35 fitted); thin replicates | Phase 2 plans 59 runs with ≥2 replicates per brick |
| **The encoder (decomposer) is never evaluated** | Phase 4 builds a labelled request set and scores it |
| Documents, not workflows | Phase 6 adds a workflow brick (tool call / system read) |

---

## 2. Goals and non-goals

### Goals
- **G1 Reproduce** the reference result from its committed data (tests pass; 2.55% LOO MAPE).
- **G2 Rate card** for our own agent stack and corpus: noise floor, tokens/byte slope, per-brick marginals.
- **G3 Quote** plain-English requests before they run, with a scored decomposer.
- **G4 Demo**: request → bricks → quote → real run → actual → error → per-brick invoice, in five minutes, no manual steps.

### Non-goals (v0.1)
- Predicting wall-clock duration or output quality.
- Pricing human approval steps or waiting time.
- Finance-system / billing integration.
- Production hardening, multi-tenancy, auth.

---

## 3. Success criteria (acceptance metrics)

| ID | Metric | Definition | Reference value | **Our target** |
|---|---|---|---|---|
| M1 | Noise floor | Coefficient of variation of total tokens across replicates of the same instruction | 0.29% | **< 1%** |
| M2 | Start-up constant | Fitted intercept ≈ null-probe cost | 30,969 fitted / 29,821 measured | Reported (model-specific) |
| M3 | Tokens per byte | Fitted coefficient on `context_bytes` | 0.366 | Reported; **R² > 0.9** on the context ladder |
| M4 | Marginal per brick | Fitted coefficient per brick | see §1.2 | Each brick has **≥ 2 replicates** behind it |
| M5 | LOO MAPE | Leave-one-out mean absolute % error over fitted tiers | 2.55% best / 6.93% constant | Best form **≥ 2× better** than constant |
| M6 | Held-out error | MAPE on compositions never used in fitting | 2.2% mean | **< 5% mean** |
| M7 | Decomposer exact match | Share of labelled requests where predicted brick counts == gold | not measured | **≥ 85%** on 30–50 requests |
| M8 | Decomposer per-brick F1 | Per-brick precision/recall on presence and counts | not measured | **≥ 0.9 for Retrieve** |
| M9 | End-to-end quote error | \|quoted − actual\| / actual for a plain-English request | 0.0–3.5% (3 cases) | **< 10% MAPE** |
| M10 | Invoice reconciliation | Σ invoice lines == measured total tokens | — | **exact** |
| M11 | Demo reliability | Consecutive successful rehearsals end-to-end | — | **3** |

---

## 4. Architecture

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                      token_yield (package)               │
                    │                                                          │
 request (text) ──▶ │ decompose ──▶ brick counts + context bytes               │
                    │                 │                                        │
                    │                 ▼                                        │
                    │            costmodel.predict ──▶ quote (tokens, band)    │
                    │                                    │                     │
                    │   harness.run(instruction) ◀───────┘  (dispatch)         │
                    │        │                                                 │
                    │        ▼                                                 │
                    │   runs.jsonl  ──▶ costmodel.fit/select (LOO CV) ──▶ coefficients.json
                    │        │                                                 │
                    │        └──▶ invoice (per-brick lines, Σ == actual)       │
                    └──────────────────────────────────────────────────────────┘
```

### 4.1 Component map (ours ↔ reference)

| Component | Purpose | Reference module(s) to read first | Our module |
|---|---|---|---|
| Vocabulary | Nine bricks, unit rules, examples, taxonomy | `token_yield/tasks.py` | `token_yield/vocabulary.py` + `vocabulary/bricks.yaml` |
| Harness | Fresh memoryless agent per run; log tokens, bytes, brick counts | `trainsuite.py`, `probes.py`, `calibrate.py` | `token_yield/harness/` |
| Cost model | Fit nested forms; select by LOO CV; predict | `costmodel.py`, `models.py`, `learn.py`, `backtest.py` | `token_yield/costmodel.py`, `token_yield/select.py` |
| Decomposer (encoder) | LLM turns a request into brick counts | `decompose.py`, `plan.py` | `token_yield/decompose/` |
| Quote / invoice / refit | Price before dispatch; reconcile after; refit on actuals | `predict.py`, `forecast.py`, `report.py` | `token_yield/quote.py`, `token_yield/invoice.py`, `token_yield/refit.py` |
| CLI | One entry point for every step and the demo | `examples/composition_demo.py` | `token_yield/cli.py` (`ty` command) |
| Dashboard | Quote vs actual, per-brick spend, batching savings | `docs/media/draw_composition.py` | `dashboard/` (notebook or Power BI; optional) |

### 4.2 Proposed repository layout

```
token-yield/
├── README.md
├── pyproject.toml                # package: token_yield; console script: ty
├── vocabulary/
│   └── bricks.yaml               # the nine bricks (schema §6.1)
├── token_yield/
│   ├── __init__.py
│   ├── vocabulary.py             # load/validate bricks.yaml; Brick dataclass
│   ├── schemas.py                # dataclasses + JSON (de)serialisation for all contracts in §6
│   ├── harness/
│   │   ├── __init__.py
│   │   ├── adapter.py            # AgentAdapter protocol + RunResult
│   │   ├── mock_adapter.py       # deterministic offline adapter (reference model + noise)
│   │   ├── azure_openai_adapter.py   # Azure OpenAI / Foundry adapter (extra: azure)
│   │   ├── probes.py             # build null/base/composite/held-out probe instructions
│   │   ├── campaign.py           # campaign design (tiers, sizes, replicates) → probe list
│   │   └── runner.py             # execute probes, append runs.jsonl, bracket with null probes
│   ├── features.py               # runs → design matrix for each model form
│   ├── costmodel.py              # CostModel: forms, fit (OLS), predict, serialise
│   ├── select.py                 # leave-one-out CV, form selection, held-out scoring
│   ├── decompose/
│   │   ├── __init__.py
│   │   ├── prompt.py             # prompt builder from vocabulary + worked examples
│   │   ├── parser.py             # strict JSON parse/validate → Decomposition
│   │   ├── decomposer.py         # LLM call via adapter; retries; confidence
│   │   └── evaluate.py           # exact match, per-brick P/R/F1, end-to-end error
│   ├── quote.py                  # Decomposition + CostModel → Quote (tokens, band, outlier flags)
│   ├── invoice.py                # actual run + decomposition → per-brick Invoice (Σ == actual)
│   ├── refit.py                  # append actual, refit, version coefficients, drift report
│   ├── report.py                 # rate card table, model-form table, predicted-vs-actual
│   └── cli.py                    # `ty` commands (§7.9)
├── experiments/
│   ├── reference/                # copied from reference repo for Phase 0 (train_runs.jsonl, decompose_cases.jsonl)
│   ├── campaign_001/
│   │   ├── campaign.yaml
│   │   ├── runs.jsonl
│   │   ├── coefficients.json
│   │   └── report.md
│   └── decompose_eval/
│       └── gold_requests.jsonl   # 30–50 labelled requests (§6.4)
├── corpus/                       # 30+ real documents, 1 KB–40 KB (not committed if sensitive)
├── tests/
│   ├── test_vocabulary.py
│   ├── test_features.py
│   ├── test_costmodel.py
│   ├── test_select.py
│   ├── test_reference_reproduction.py   # Phase 0 gate: 2.55% LOO MAPE from reference data
│   ├── test_harness_mock.py
│   ├── test_decompose_parser.py
│   ├── test_decompose_evaluate.py
│   ├── test_quote_invoice.py
│   └── test_cli.py
├── demo/
│   ├── run_demo.sh               # the five-minute demo, no manual steps
│   └── demo_requests.jsonl       # requests never used in fitting
└── docs/
    ├── measurement-protocol.md
    ├── rate-card.md              # generated by `ty report`
    └── decisions.md              # ADRs: stack, model, corpus, budget
```

---

## 5. Environment and configuration

- **Python ≥ 3.11.** Core package: standard library only (`json`, `dataclasses`, `statistics`,
  `math`, `hashlib`, `argparse`, `pathlib`). Tests: `pytest`. Optional extras:
  `azure` → `openai` (Azure OpenAI), `azure-identity`; `dashboard` → `matplotlib`, `pandas`.
- **Configuration via environment variables** (never commit secrets; provide `.env.example`):

```
TY_ADAPTER=mock | azure_openai
TY_MODEL_DEPLOYMENT=<deployment name>          # held FIXED for a whole campaign
TY_AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
TY_AZURE_OPENAI_API_VERSION=<api version>
TY_CORPUS_DIR=./corpus
TY_EXPERIMENTS_DIR=./experiments
TY_CAMPAIGN_ID=campaign_001
TY_SEED=1337                                   # MockAdapter noise seed
```

- Authentication for Azure uses `DefaultAzureCredential` (keyless) by default; API key only if
  `TY_AZURE_OPENAI_API_KEY` is set. Token counts are **always read from the provider's usage
  record** (`usage.prompt_tokens`, `usage.completion_tokens`), never estimated.
- Before writing our own logger, check whether Microsoft's Agent Governance Toolkit (the reference
  README says Token Yield composes with it) already exposes per-run token usage; if so, implement
  the adapter as a thin wrapper over it.

---

## 6. Data contracts (normative)

All files are UTF-8. JSONL files have one JSON object per line. Field names mirror the reference
`experiments/train_runs.jsonl` where a corresponding field exists; check that file in Phase 0 and
add an alias map in `schemas.py` if names differ.

### 6.1 `vocabulary/bricks.yaml`

```yaml
version: 1
bricks:
  - name: Review
    category: cross-cutting          # Swanson taxonomy: corrective | adaptive | perfective | preventive | cross-cutting
    bought_as: "Contract and compliance review"
    definition: "Read one or more documents and answer a judgement question about them."
    unit: "One judgement question answered over the supplied documents."
    examples:
      - "Read the filing and state whether revenue guidance was raised, held or cut."
      - "Review this SOW and say whether the acceptance criteria are testable."
  - name: Extract
    category: cross-cutting
    bought_as: "Invoice and claims intake, KYC"
    definition: "Pull named fields or values out of a document."
    unit: "One field or value extracted."
    examples: [...]
  # Classify, Retrieve, Reconcile, Draft, Remediate, Validate, Report — same shape
```

Validation rules (`vocabulary.py`): exactly the nine names above unless `allow_extension: true`
is set (Phase 6); every brick has non-empty `definition`, `unit`, and ≥ 2 `examples`; names are
unique and case-sensitive.

### 6.2 `runs.jsonl` — one record per agent run

```json
{
  "run_id": "c001-base-retrieve-s2-r1",
  "campaign_id": "campaign_001",
  "timestamp": "2026-09-15T10:32:11+05:30",
  "tier": "null | base | composite | heldout | live",
  "probe_id": "base-retrieve-s2",
  "replicate": 1,
  "instruction_sha256": "…",
  "instruction": "Find which of the attached filings reports a 26% decline …",
  "units": {"Review": 0, "Extract": 0, "Classify": 0, "Retrieve": 1, "Reconcile": 0,
            "Draft": 0, "Remediate": 0, "Validate": 0, "Report": 0},
  "context_files": ["corpus/acme-10k-2025.txt"],
  "context_bytes": 20315,
  "adapter": "azure_openai",
  "model": "<deployment name>",
  "sampling": {"temperature": 0, "top_p": 1, "seed": null},
  "prompt_tokens": 38102,
  "completion_tokens": 389,
  "total_tokens": 38491,
  "elapsed_s": 41.7,
  "status": "ok | failed | interrupted",
  "output_sha256": "…",
  "notes": ""
}
```

Rules: `total_tokens == prompt_tokens + completion_tokens`; `context_bytes` is measured on disk
before the run; failed runs are kept with `status != ok` and excluded from fitting, never deleted;
`units` always contains all nine keys.

### 6.3 `coefficients.json` — a fitted model, versioned

```json
{
  "campaign_id": "campaign_001",
  "fitted_at": "2026-09-20T18:00:00+05:30",
  "data_sha256": "<sha256 of the runs.jsonl lines used>",
  "n_runs_fitted": 54,
  "form": "bytes_per_brick",
  "intercept": 30969.0,
  "bytes_coef": 0.3661,
  "marginals": {"Review": 52.0, "Extract": 418.0, "Classify": 0.0, "Retrieve": 5384.0,
                "Reconcile": 1770.0, "Draft": 0.0, "Remediate": 0.0, "Validate": 1038.0, "Report": 838.0},
  "selection": {
    "method": "loo_cv",
    "scores": {"constant": 6.93, "units": 5.44, "bytes": 4.85, "bytes_units": 4.69,
               "per_brick": 3.22, "bytes_per_brick": 2.55}
  },
  "noise_floor_cv_pct": 0.29,
  "heldout": {"mape_pct": 2.2, "worst_pct": 4.7, "n": 4}
}
```

The six **model forms** (nested, in this order) are normative:

| form key | features |
|---|---|
| `constant` | intercept only |
| `units` | intercept + Σ units (all bricks summed) |
| `bytes` | intercept + context_bytes |
| `bytes_units` | intercept + context_bytes + Σ units |
| `per_brick` | intercept + one coefficient per brick |
| `bytes_per_brick` | intercept + context_bytes + one coefficient per brick |

### 6.4 `gold_requests.jsonl` — decomposer evaluation set

```json
{
  "request_id": "g017",
  "request": "read the filing, pull three fields, write two auditor checks",
  "gold_units": {"Review": 1, "Extract": 3, "Classify": 0, "Retrieve": 0, "Reconcile": 0,
                 "Draft": 0, "Remediate": 0, "Validate": 2, "Report": 0},
  "context_files": ["corpus/acme-10k-2025.txt"],
  "labellers": ["VB", "AS"],
  "agreed": true,
  "notes": "ambiguity: 'checks' could be Validate or Draft; resolved to Validate by unit rule"
}
```

### 6.5 Decomposition (LLM output → validated object)

The decomposer must return **strict JSON only**:

```json
{"units": {"Review": 1, "Extract": 3, "Classify": 0, "Retrieve": 0, "Reconcile": 0,
           "Draft": 0, "Remediate": 0, "Validate": 2, "Report": 0},
 "context_files": ["corpus/acme-10k-2025.txt"],
 "rationale": "one read of the filing; three named fields; two auditor checks",
 "confidence": 0.8}
```

### 6.6 Quote and Invoice

```json
// Quote
{"request_id": "demo-003", "decomposition": {…§6.5…}, "context_bytes": 20315,
 "predicted_tokens": 34804, "band_pct": 5.0, "band_tokens": [33064, 36544],
 "outlier_flags": ["Retrieve: 5,384/unit is 10x other bricks; narrow the search space"],
 "coefficients_ref": "campaign_001@<data_sha256[:8]>"}

// Invoice (after the run)
{"run_id": "…", "actual_tokens": 33723, "predicted_tokens": 34804, "error_pct": 3.2,
 "lines": [
   {"item": "start-up", "tokens": 30969},
   {"item": "context", "detail": "20,315 bytes × 0.3661", "tokens": 7437},
   {"item": "Review × 1", "tokens": 52},
   {"item": "Extract × 3", "tokens": 1254},
   {"item": "Validate × 2", "tokens": 2076},
   {"item": "reconciliation", "detail": "actual − Σ model lines", "tokens": -8065}
 ],
 "sum_check": 33723}
```

Rule: the invoice always carries a final `reconciliation` line so that `Σ lines == actual_tokens`
exactly (M10). The model lines show where the model *thought* the cost was; the reconciliation
line is the residual.

---

## 7. Component specifications

### 7.1 `vocabulary.py`
- `load_vocabulary(path) -> Vocabulary` (validates per §6.1; raises `VocabularyError` listing all problems).
- `Vocabulary.names -> list[str]` (stable order as in the YAML).
- `Vocabulary.zero_units() -> dict[str,int]`.
- `Vocabulary.render_for_prompt() -> str` (definitions, unit rules, examples; used by the decomposer).

### 7.2 `harness/adapter.py`, `mock_adapter.py`, `azure_openai_adapter.py`

```python
class RunResult(TypedDict):
    prompt_tokens: int; completion_tokens: int; total_tokens: int
    elapsed_s: float; output_text: str; raw: dict

class AgentAdapter(Protocol):
    name: str
    model: str
    def run(self, instruction: str, context_files: list[Path], *, temperature: float = 0.0) -> RunResult: ...
```

- **Fresh, memoryless run every time**: a new conversation per call; identical system prompt;
  identical tool set (file read + search over `TY_CORPUS_DIR` only); documents attached as
  context by reading the files into the prompt (v0.1) — record how in `notes`.
- `MockAdapter`: deterministic. `total = 29821 + 0.37*bytes + Σ ref_marginal*units`, then
  multiply by `1 + N(0, 0.003)` using `TY_SEED` (≈0.3% noise floor, matching the reference).
  Splits into prompt/completion 97/3. Purpose: offline tests and the CI pipeline.
- `AzureOpenAIAdapter`: chat completions; reads `usage` from the response; retries once on
  transient errors; raises `AdapterError` otherwise. Never estimates tokens.

### 7.3 `harness/probes.py` and `harness/campaign.py`
- Probe instruction templates per brick, parameterised by unit count and context files, plus the
  **null probe**: `"Reply with the single word DONE."` (no context).
- `design_campaign(vocab, corpus, sizes=("s1","s2","s3"), replicates=2, composites=12, heldout=5) -> list[Probe]`
  implementing this tier table (59 probes by default):

| Tier | What | n |
|---|---|---|
| null | reply DONE, no context | 3 at start + 3 at end (drift bracket) |
| base | each brick alone × 3 context sizes (≈1 KB, ≈9 KB, ≈35 KB), +1 replicate per brick | 36 |
| composite | 2–4 bricks in one instruction, every brick appearing ≥ 2 times across the set; includes at least one "same brick twice" case | 12 |
| heldout | compositions excluded from fitting; ≥ 1 with arity outside the training range (5 bricks) | 5 |

- Context sizes are chosen from the corpus by actual byte size to span roughly one order of magnitude.

### 7.4 `harness/runner.py`
- `run_campaign(campaign, adapter, out_path) -> None`: appends one `runs.jsonl` record per probe
  (schema §6.2); resumable (skips `run_id`s already present with `status == ok`); computes
  `instruction_sha256` and `context_bytes` before the call; brackets the campaign with null probes.
- CLI: `ty run-campaign --campaign experiments/campaign_001/campaign.yaml`.

### 7.5 `features.py`, `costmodel.py`
- `design_matrix(runs, form, vocab) -> (X: list[list[float]], y: list[float])` for each form in §6.3.
- `fit_ols(X, y) -> list[float]` — ordinary least squares via normal equations with a tiny ridge
  (`1e-9`) for stability; pure Python (Gaussian elimination). Clamp negative marginals to 0 and
  record that they were clamped (they are "below noise").
- `CostModel.predict(units, context_bytes) -> float`; `CostModel.to_json()/from_json()` (§6.3).
- Unit tests: recover known coefficients from synthetic data to 1e-6; predict reproduces the
  reference held-out numbers when loaded with the reference coefficients.

### 7.6 `select.py`
- `loo_mape(runs, form, vocab) -> float` — leave-one-out over all fitted runs (`tier in {null, base, composite}` and `status == ok`).
- `select_form(runs, vocab) -> (best_form, scores: dict)`; a richer form must **strictly** beat the
  simpler one to be chosen (ties go to the simpler form).
- `score_heldout(model, runs_heldout) -> {mape_pct, worst_pct, per_case: [...]}`.
- `noise_floor(runs) -> cv_pct` from replicate pairs (same `instruction_sha256`).

### 7.7 `decompose/`
- `build_prompt(vocab, request, corpus_index) -> str`: vocabulary rendering, the unit rules, 3–4
  worked examples (from the reference `decompose_cases.jsonl` and our gold set), the list of
  available corpus files, and the strict-JSON output instruction (§6.5).
- `parse(text) -> Decomposition`: strict JSON; all nine keys present; non-negative integers;
  `context_files` must exist; otherwise `DecompositionError`. One repair retry allowed (re-ask
  with the validation error appended); then fail loudly.
- `Decomposer(adapter, vocab).decompose(request) -> Decomposition`.
- `evaluate(gold_path, decomposer) -> EvalReport` with: exact-match rate; per-brick precision,
  recall, F1 on presence; MAE on counts; confusion pairs (e.g. Reconcile↔Review); and, when
  `--run` is passed, end-to-end quote error by actually running each request through the harness.

### 7.8 `quote.py`, `invoice.py`, `refit.py`
- `quote(decomp, model, band_pct)` → §6.6 Quote; `band_pct` defaults to the model's held-out MAPE
  rounded up to the nearest whole percent, never below the noise floor.
- Outlier flag rule: any brick whose `marginal × units` exceeds 25% of the predicted total, or any
  Retrieve unit at all, adds a flag with the operational reading ("narrow the search space").
- `invoice(run, decomp, model)` → §6.6 Invoice; asserts `sum_check == actual_tokens`.
- `refit(campaign_dir)`: appends live runs, re-runs `select_form`, writes a **new**
  `coefficients.json` (never overwrites; suffix with `data_sha256[:8]`), and appends a row to
  `coefficients_history.csv` (fitted_at, n_runs, form, intercept, bytes_coef, LOO MAPE, held-out MAPE).

### 7.9 `cli.py` — the `ty` command

| Command | What it does | Phase |
|---|---|---|
| `ty reproduce` | Load reference data from `experiments/reference/`, run LOO selection, print the form table, assert 2.55% | 0 |
| `ty vocab validate` | Validate `bricks.yaml`; print the one-page vocabulary sheet (markdown) | 1 |
| `ty vocab agreement --a labels_a.jsonl --b labels_b.jsonl` | Inter-rater agreement on brick sets | 1 |
| `ty design-campaign` | Write `campaign.yaml` (probe list) from vocab + corpus | 2 |
| `ty run-campaign` | Execute probes via the configured adapter; append `runs.jsonl` | 2 |
| `ty fit` | Select form by LOO CV; write `coefficients.json`; print scores and noise floor | 3 |
| `ty report` | Rate card (markdown), predicted-vs-actual table, held-out table → `docs/rate-card.md` | 3 |
| `ty decompose "<request>"` | Print the decomposition JSON | 4 |
| `ty eval-decomposer [--run]` | Score against `gold_requests.jsonl`; optional end-to-end | 4 |
| `ty quote "<request>"` | Decompose + price; print Quote JSON and a one-line human summary | 4 |
| `ty run "<request>"` | Quote, dispatch for real, print actual, error %, and the Invoice | 5 |
| `ty refit` | Refit on all `ok` runs; version coefficients; print drift | 5 |
| `ty demo` | The five-minute script end-to-end (`demo/run_demo.sh` calls this) | 5 |

All commands accept `--adapter mock|azure_openai` and `--campaign <id>`; all print machine-readable
JSON with `--json`.

---

## 8. Measurement protocol (applies to every run)

1. Fresh agent, no memory of prior runs; same system prompt; same tool set; same mounted corpus.
2. Sampling fixed for the whole campaign (`temperature=0`, `top_p=1`); the model deployment is held fixed —
   changing either invalidates the fit and starts a new `campaign_id`.
3. The instruction text is stored verbatim and hashed; a replicate is provably the same instruction.
4. `context_bytes` is measured on disk before the run.
5. Token counts come from the provider's usage record only.
6. Failed/interrupted runs are recorded with a status flag and excluded from fitting, never deleted.
7. Null probes bracket the campaign (3 at start, 3 at end) so drift during the campaign is detectable.
8. Every fitted artefact records the `data_sha256` of the exact lines it was fitted on.

---

## 9. Phased task list (execution order for Copilot)

Each phase lists tasks as checkboxes, then the **Definition of Done (DoD)** and the commands that must succeed.

### Phase 0 — Reproduce the reference (days 1–3)
- [ ] Clone `https://github.com/ginaecho/lego-bricks-token-prediction`; run `pip install -e .`,
      `python -m examples.composition_demo`, `python -m pytest tests/test_compose.py -q` (39 tests).
- [ ] Copy `experiments/train_runs.jsonl` and `experiments/decompose_cases.jsonl` into
      `experiments/reference/`; record the upstream commit hash in `experiments/reference/SOURCE.md`.
- [ ] Read `token_yield/trainsuite.py`, `costmodel.py`, `models.py`, `decompose.py`; document the
      run-record field names and add an alias map in `schemas.py` if they differ from §6.2.
- [ ] Scaffold our repo (§4.2), `pyproject.toml`, `ty` console script, `pytest` config.
- [ ] Implement `vocabulary.py`, `schemas.py`, `features.py`, `costmodel.py`, `select.py`.
- [ ] Implement `ty reproduce` and `tests/test_reference_reproduction.py`.

**DoD:** `pytest -q` green; `ty reproduce` prints the six-form table with `bytes_per_brick`
= 2.55% and `constant` = 6.93% (± 0.05 pp, rounding), and held-out mean 2.2%.

### Phase 1 — Fix the vocabulary and corpus (days 3–6)
- [ ] Write `vocabulary/bricks.yaml` with definitions, unit rules, ≥ 2 examples per brick
      (adopt the nine reference names; do not rename).
- [ ] Choose the corpus: 30+ real documents, 1 KB–40 KB, from the target workload; write
      `corpus/README.md` (source, licence, sizes). Add `ty corpus index` to list files with byte sizes.
- [ ] Produce `docs/vocabulary-sheet.md` via `ty vocab validate`.
- [ ] Two labellers independently decompose 20 sample requests → `experiments/decompose_eval/agreement_{a,b}.jsonl`;
      run `ty vocab agreement`.

**DoD:** validation passes; agreement on brick sets ≥ 80%; disagreements resolved by tightening
definitions (commit the diff to `bricks.yaml`), not by majority vote.

### Phase 2 — Harness and measurement campaign (weeks 2–3)
- [ ] Implement `harness/adapter.py`, `mock_adapter.py`, `azure_openai_adapter.py`.
- [ ] Verify on day one that the real adapter returns `usage` token counts; if not, stop and
      record the blocker in `docs/decisions.md`.
- [ ] Implement `probes.py`, `campaign.py`, `runner.py`; `ty design-campaign`; `ty run-campaign`.
- [ ] Run the full campaign with `--adapter mock` first (CI), then for real with the fixed deployment.
- [ ] Compute noise floor, tokens/byte slope on the Review ladder (with R²), and the apart-vs-together
      saving on ≥ 3 compositions; write `experiments/campaign_001/report.md`.

**DoD:** 59 `ok` runs in `runs.jsonl`; M1 noise floor < 1%; M3 R² > 0.9; every brick has ≥ 2
replicates; a rate card table exists next to the reference's for comparison. Budget check: 59 × ~30k
≈ 1.77M tokens (2.2M with 25% headroom) — record actual spend in the report.

### Phase 3 — Fit and select (weeks 3–4)
- [ ] `ty fit` on `campaign_001`: six forms, LOO CV, strict-improvement selection; write `coefficients.json`.
- [ ] `ty report`: model-form table, predicted-vs-actual (held-out highlighted), rate card → `docs/rate-card.md`.
- [ ] Optional: `dashboard/` scatter plot (matplotlib) of predicted vs actual.

**DoD:** M5 best form ≥ 2× better than constant; M6 held-out MAPE < 5%; the highest-arity held-out
case is the worst prediction (record whether this expectation held).

### Phase 4 — Decomposer and its evaluation (weeks 4–5)
- [ ] Implement `decompose/prompt.py`, `parser.py`, `decomposer.py`; `ty decompose`, `ty quote`.
- [ ] Build `gold_requests.jsonl`: 30–50 plain-English requests spanning all nine bricks, arities
      1–5, including deliberately ambiguous phrasings; two-reviewer gold labels.
- [ ] Implement `decompose/evaluate.py`; `ty eval-decomposer` and `ty eval-decomposer --run`.

**DoD:** M7 exact match ≥ 85%; M8 Retrieve F1 ≥ 0.9; M9 end-to-end MAPE < 10% on the gold set;
confusion pairs documented in `experiments/decompose_eval/report.md`.

### Phase 5 — Close the loop, invoice, demo (weeks 5–6)
- [ ] Implement `invoice.py`, `refit.py`, `ty run`, `ty refit`, `coefficients_history.csv`.
- [ ] Implement `ty demo` + `demo/run_demo.sh` (five steps, §10); `demo/demo_requests.jsonl`
      contains only requests never used in fitting.
- [ ] Dashboard (optional): quote vs actual over time; per-brick spend; apart-vs-together savings.
- [ ] Rehearse three times; log each rehearsal's quote, actual and error in `demo/rehearsals.md`.

**DoD:** M10 every invoice reconciles exactly; rolling end-to-end error falls or holds across ≥ 3
refits; M11 three consecutive successful rehearsals.

### Phase 6 (optional) — Extend the evidence (week 7+)
- [ ] Second-model replication: new `campaign_002` with a different deployment; null + base + subset
      of composites; compare shape (fixed toll, linear context, sub-additivity, Retrieve ≥ 5× next brick).
- [ ] Workflow brick: add one primitive whose work is a system interaction (DB query / API call /
      simulated approval wait) with `allow_extension: true`; measure with ≥ 2 replicates; refit.

**DoD:** two-model rate card comparison in `docs/rate-card.md`; workflow brick has a fitted marginal.

---

## 10. Demo runbook (`ty demo`, ≤ 5 minutes, no manual steps)

| Step | Command / output | Proves | Time |
|---|---|---|---|
| 1 Rate card | `ty report --rate-card` — nine marginals, tokens/byte, start-up constant, noise floor | Cost is measurable and stable | 60 s |
| 2 Batching lever | `ty compare-batching --composition "Review+3Extract+2Validate"` — apart vs together, % saved | Composition is sub-additive | 45 s |
| 3 Live quote | `ty quote "<typed request>"` — bricks, tokens, currency at the configured rate | Quotable before dispatch | 60 s |
| 4 Run and reconcile | `ty run "<same request>"` — actual beside the quote, error % | Quote lands inside its band | 90 s |
| 5 Invoice and refit | prints the Invoice; `ty refit` shows the coefficient history row appended | Spend reconciles; model improves with use | 45 s |

Currency conversion uses `TY_PRICE_PER_1K_INPUT` / `TY_PRICE_PER_1K_OUTPUT` from the environment
(`TODO(measure)`: set from the chosen model's published rate at approval time). If unset, print
tokens only.

---

## 11. Testing strategy

- **Unit** (pure functions, no network): vocabulary validation; design matrices for all six forms;
  OLS recovers planted coefficients; LOO CV on tiny synthetic sets; parser rejects malformed JSON;
  invoice `sum_check`; quote band arithmetic.
- **Reproduction gate**: `test_reference_reproduction.py` loads `experiments/reference/train_runs.jsonl`
  and asserts the published LOO table and held-out numbers (tolerance 0.05 pp).
- **Pipeline** (offline): full campaign → fit → quote → run → invoice → refit using `MockAdapter`,
  asserting the pipeline recovers the mock's planted coefficients within noise.
- **Contract tests**: every JSON written by the CLI validates against §6 (write minimal validators in
  `schemas.py`; do not add a JSON-schema dependency).
- **CI**: `pytest -q` and `ty reproduce` on every push; the real adapter is never called in CI.

---

## 12. Coding conventions for Copilot

- Python 3.11+, type hints everywhere, `from __future__ import annotations`, dataclasses for records.
- Pure functions in `features.py`, `costmodel.py`, `select.py`; I/O only at the edges (`runner.py`, `cli.py`).
- No hidden state: every function that fits or predicts takes the data and returns the result.
- Log to stderr; print results to stdout; `--json` makes stdout machine-readable.
- Never estimate tokens; never delete a run; never overwrite a `coefficients.json`.
- Docstrings state the unit of every number (tokens, bytes, percent).
- Keep the core package dependency-free; put `openai`, `azure-identity`, `matplotlib`, `pandas` behind extras.
- Commit messages: `phase0: …`, `phase1: …` so progress maps to this document.

---

## 13. Risks and mitigations (engineering view)

| Risk | Impact | Mitigation in code/process |
|---|---|---|
| Our noise floor ≫ 0.29% | Cheap bricks indistinguishable from noise | `temperature=0`; more replicates; report the floor; only claim marginals above it (clamp + flag) |
| Findings' shape does not transfer | Weaker sub-additivity / linearity on our model | Treat as replication study; the report states the result either way |
| Decomposer unreliable on ambiguous requests | Confidently wrong quotes | Score on gold set before live use; show `confidence`; `ty run` asks for confirmation of the brick set in demo mode |
| Provider does not expose usage per run | No ground truth | Day-one check in Phase 2; fall back to billing logs; consider the Agent Governance Toolkit |
| Corpus unrepresentative | Rate card wrong for real work | Choose corpus from target workload; re-run base tier when workload changes |
| Scope creep | Six weeks becomes a quarter | Every addition must carry a metric in §3 and a DoD in §9 |

---

## Appendix A — The nine bricks (reference)

| Brick | Category | Where it is bought | Reference marginal (tokens/unit) |
|---|---|---|---|
| Review | cross-cutting | contract and compliance review | 52 (below noise once bytes are modelled) |
| Extract | cross-cutting | invoice and claims intake, KYC | 418 |
| Classify | cross-cutting | ticket and email triage, routing | below noise |
| Retrieve | cross-cutting | knowledge discovery, e-discovery | 5,384 |
| Reconcile | corrective | financial close, audit, dispute resolution | 1,770 |
| Draft | adaptive | proposals, memos, marketing copy | below noise |
| Remediate | corrective | exception handling, error correction | below noise |
| Validate | preventive | control testing, quality assurance | 1,038 |
| Report | perfective | management and board reporting | 838 |

## Appendix B — Reference repository quick commands

```bash
git clone https://github.com/ginaecho/lego-bricks-token-prediction
cd lego-bricks-token-prediction
pip install -e .                             # Python ≥ 3.9, no runtime dependencies
python -m examples.composition_demo          # the whole loop, from committed data
python -m pytest tests/test_compose.py -q    # 39 tests over the vocabulary and model
python docs/media/draw_composition.py        # redraw the figure from the data
```

Key files: `token_yield/trainsuite.py`, `token_yield/decompose.py`, `token_yield/costmodel.py`,
`experiments/train_runs.jsonl`, `experiments/decompose_cases.jsonl`, `docs/composition-findings.md`,
`docs/calibration-findings.md`.

## Appendix C — Reference compositions (for fixtures)

Apart vs together (tokens): Review + 3×Extract + 2×Validate — 97,646 → 34,804 (64%);
Review + 2×Remediate + 2×Validate — 96,542 → 33,699 (65%);
Retrieve + Review + Remediate + Validate — 132,235 → 37,971 (71%).

Held-out (actual / predicted): Review + 2×Validate 33,174 / 33,549 (1.1%); 2×Draft + Report
32,878 / 31,986 (2.7%); 6×Review + Draft 38,732 / 38,583 (0.4%); Retrieve + Review + Remediate +
Validate 36,283 / 37,971 (4.7%). Mean 2.2%.

Plain-English cases (decomposed → predicted / actual): "read the filing, pull three fields, write two
auditor checks" → Review + 3×Extract + 2×Validate → 34,804 / 33,723 (3.2%); "find which company
reported the 26% decline, then draft a risk note" → Retrieve + Draft → 36,217 / 37,541 (3.5%);
"compare these two filings, then write a board summary" → Reconcile + Report → 34,968 / 34,968 (0.0%).

Context ladder (Review instruction): 0 B → 29,821; 861 B → 32,174; 3,179 B → 33,163;
8,973 B → 34,051; 20,315 B → 38,491; 34,190 B → 43,924 tokens.

---

*Sources: README and `docs/composition-findings.md` of ginaecho/lego-bricks-token-prediction, read on
4 September 2026. All reference figures are the author's published measurements; reproducing them is
Phase 0.*
