# LEGO-Bricks Token Prediction — Final POC Plan

> ⚠️ **SUPERSEDED — do not build from this file.**
> `docs/TOKEN_YIELD_PLAN.md` is the single source of truth. It carries this document's baselines,
> variable-portion metric, sealed blind set, conformal intervals, batching quality arm and
> mechanical verdict, and adds the product surface this plan lacked.
> Kept here as a source document.

**Status:** pre-registration draft. No paid work is authorised until Phase 1 closes.
**Audit pin:** `ginaecho/lego-bricks-token-prediction` `main` @ `c63702d`. Branch `gc/prototype` is synthetic and excluded.
**Host:** standalone `poc/` tree (or sibling repo). The pinned repo is evidence, not the build target. This document lives in the audited repo for traceability only; adding it does not move the pin.

This plan is a merge of two earlier drafts. Section 12 records which draft each decision came from and why, so the merge itself is auditable.

---

## 1. The question

> For **one** pinned model, **one** owned harness, and **one** public SEC workload, can a four-brick decomposition predict **billable dollars** on unseen tasks better than simple baselines, with usable intervals — and does measured batching save cost without producing worse answers?

Four claims, scored independently so that failure is diagnosable rather than merely disappointing:

| Claim | Question | Failure means |
| --- | --- | --- |
| **G0 — runtime** | Is the pinned snapshot served, with itemised usage, zero cache, zero reasoning? | No measurement is possible. Stop; nothing else is interpretable. |
| **G1 — decoder** | Given human-verified brick counts, does the fitted model price live cost better than baselines? | The composition arithmetic does not hold on this runtime. |
| **G2 — encoder** | Can the model map unseen request text onto the four bricks accurately enough to quote? | Quoting is not automatable; bricks may still be a manual scoping aid. |
| **G3 — end-to-end** | Encoder counts → quote, scored against the same actuals? | The pipeline is not shippable even if both halves work. |
| **G4 — batching** | Is one combined request cheaper than *N* separate ones **at equal answer quality**? | Batching is not a saving, it is a discount on worse output. |

Three outcomes are pre-registered and all three are successes. **Feasible**, **Narrow**, **Stop**. A negative result is a finished POC, not a failed project.

---

## 2. Locked contract

Immutable once Phase 1 is committed and hashed. Changing any row invalidates the campaign.

| Item | Lock |
| --- | --- |
| **Model** | Literal `claude-haiku-4-5-20251001`. `temperature=0`. No thinking / effort budget. No alias fallback. Snapshot withdrawn → **Stop**. Every call asserts `response.model` echoes the request. |
| **Harness** | New runner over the Anthropic Messages API. Frozen system prompt + exactly three read-only tools: `list_dir`, `read_file`, `grep`. Fresh conversation per run. No bash, no web, no nested subagents. |
| **Client** | `anthropic.Anthropic`, synchronous, non-streaming `messages.create`, owned tool loop. Serial. `anthropic` version pinned on freeze day. |
| **Cache** | Never set `cache_control`. Non-zero cache creation/read tokens → row excluded. The primary arm is cache-off. |
| **Tokens** | `response.usage` only. No `len(s)/4`, no local tokeniser, no estimate. Missing usage is a failed measurement, never a zero. |
| **Target** | $y = T_{in} + (p_{out}/p_{in}) \cdot T_{out}$ from freeze-day `pricing.json`. Dollars $= y \cdot p_{in} \cdot 10^{-6}$. Provisional rates \$1 / MTok in, \$5 / MTok out. **Never hardcode `5`.** Cache rates are stored, unused, and must contribute \$0. |
| **Workload** | 20 issuers × 2 primary filings (10-K + subsequent 10-Q) = 40 public EDGAR documents. 96 single-brick tasks, split 48 train / 24 val / 24 blind. |
| **Bricks** | Four original slugs only: `extract`, `validate`, `reconcile`, `retrieve`. Do not rename. The other five are dropped. |
| **Encoder** | Same pinned model + frozen decompose prompt. Keyword regex encoder is a **control**, never a silent fallback. Gold is human. |
| **Fit data** | New runs only. The 39 historical JSONL rows are never mixed into the fit. |
| **Volume** | ≤ 220 accepted experimental dispatches (see §6). Encoder calls on a separate ≤ 250 ledger. Window ≤ 10 days. |
| **Host** | Standalone CLI + generated HTML/JSON. Recorded replay is the demo. One optional live run, labelled illustrative. |
| **Quality** | Frozen answer keys / 100-point rubric. Human graders. No LLM-as-judge. |
| **Python** | ≥ 3.11 (3.13 named in `frozen/runtime.json`). |

**Excluded, not to be re-litigated:** the other five bricks, `gc/prototype`, staffing / ROI / chargeback models, multi-model transfer, OpenHarness L1/L2/L5 as token evidence, a council UI, LLM-as-judge, document slicing, `pandoc`, `sec-edgar-downloader`, mixed-brick *execution* inside the 96, and rebuilding `token_yield` as a product.

---

## 3. What the historical evidence can and cannot support

Verified against the pinned repo while writing this plan:

- `experiments/train_runs.jsonl` row 1 is the null probe at **29,821 tokens** with `context_bytes: 0`. That intercept is Claude Code `subagent_tokens` on a runtime we do not own and cannot reconstruct. **We will not attempt to reproduce 29,821.** Our owned boot will be far smaller, and that is an artefact of the harness, not a failed replication.
- Each row carries a single scalar `tokens` field with **no input/output split**. Billable dollars cannot be recovered from these rows at all, because output tokens cost roughly 5× input. This alone is why the historical numbers cannot be a prior for a dollar claim.
- The fitted form has **nine** free brick coefficients against 35 fitted rows, one of them negative (`draft = −137`). LOO MAPE of 2.55% is in-harness reconstruction and is optimistically biased.
- Batching savings of 64–71% are **model-derived counterfactuals**, not a measured live comparison. They are withdrawn as priors.
- Live encoder evidence is 3 cases. Too small to carry any claim.

**Quarantine rule (CI-enforced):** the strings `64%`, `71%`, `2.55%`, and any encoder-accuracy or interval claim from the historical campaign may not appear in POC outputs as priors. They may appear only in `docs/evidence/excluded-priors.md`.

### The fixed-cost illusion — the single most important control in this plan

With a boot intercept of ~30k tokens and a variable portion of a few hundred, `return 30969` scores a superb MAPE. Any gate phrased as "mean APE < 10%" is therefore passable by a constant. Every accuracy claim in this plan is scored **twice**:

- **Total MAPE** on $y$ — the headline number.
- **Variable-portion WAPE (vWAPE)** on $y_{var} = y - \hat{b}$ — the number that decides whether bricks did anything.

$\hat{b}$ is the median of the owned N0 null probes, frozen in `frozen/runtime.json`. $y_{var} \le 0$ stays in the table as a measurement failure; it is never clamped.

---

## 4. Evidence labelling and exit discipline

Every report, every CLI invocation, and every row carries exactly one **evidence class**:

| Label | Meaning |
| --- | --- |
| `replay` | Recomputed from committed historical JSONL. Reference only. Never a POC result. |
| `pipeline_only` | Fixture-driven, zero API calls. Proves the code path, proves nothing about the model. |
| `feasibility` | Real live dispatch under the locked contract. The only class that can move a gate. |

Rules, enforced in CI:

1. A report mixing classes must state the split per number, or it fails validation.
2. Any CLI that completes a partial run **must not print PASS** and must exit non-zero.
3. Any mandatory-gate failure exits non-zero.
4. Every headline number carries a provenance class: `measured` / `derived` / `synthetic` / `missing`.

---

## 5. Phase sequence

```
P0  Contract + evidence audit          $0
     │  poc-contract.md hashed; verdict evaluator + historical rescore in CI
     ▼
P1  Measurement harness + G0           fixtures $0 → smoke ≤10 → ladder ~35
     │  b̂ from N0; N4 units⟂bytes finding; measurement-meaning.md
     ▼
P2  Workload freeze                    fixtures $0 → ≤10 train smoke
     │  40 docs + 96 tasks hashed; blind sealed by API
     │  (P3 annotation may start once quote text is frozen — EDGAR bytes not required)
     ▼
P3  Taxonomy + encoder                 0 of 96 dispatched; ≤250 encoder calls
     │  human gold on 72; encoder A/B frozen; blind untouched
     ▼
P4  Decoder vs baselines               72 + 4 repeats — unseal train/val only
     │  frozen decoder + B0–B5 + conformal intervals + brick weights w_b
     ▼
P5  Freeze → 24 blind → batching → demo → mechanical verdict
```

### Dependency gates

| Gate | Must be green before |
| --- | --- |
| P0 contract, schemas, verdict fixtures | any paid work |
| P1 fixture suite | any live smoke |
| **G0** (model echo, cache = 0, reasoning = 0, itemised usage) | the full ladder |
| P1 N0 boot + N4 finding + `runtime_hash` | the P4 fit |
| P2 48/24/24 manifests, leakage check, orthogonality | P3 gold packets; P4 dispatch |
| P3 human IAA + construct validity | spending the 96 |
| P3 encoder freeze | P4 priced error; P5 quotes |
| P4 winning decoder + frozen baselines | P5 unseal |
| P5.2 point gates | P5.3 batching budget — a hard prediction fail is **Stop** and batching is not run |

---

## 6. Budget

The earlier drafts disagreed here, and one of them under-counted: it listed the probe ladder at 10 dispatches in the budget table while describing it as 30–40 in the sequence. Corrected:

| Use | Dispatches |
| --- | --- |
| P1 smoke (G0) | 10 |
| P1 full null-probe ladder | 35 |
| P2 workload smoke (excluded from fit) | 10 |
| P4 train + val + 4 repeats | 76 |
| **P5 blind — hard fence** | **24** |
| P5 batching, two-task (8 bundles × 3) | 24 |
| P5 batching, four-task (3 bundles × 5) | 15 |
| Infrastructure reserve | 26 |
| **Total** | **220** |

Encoder calls run on a **separate** ≤ 250 ledger. Blind quoting costs 24 encoder calls. Predicted batching saving is arithmetic from frozen $\hat{b}$ and $w_b$ — it costs nothing.

**Why a cap at all.** At an owned boot of a few thousand tokens, 220 runs costs single-digit dollars. The envelope is not a cost control; it is an **integrity control**. A fixed, pre-registered dispatch count is what stops the campaign from quietly becoming a search for a flattering result.

**Overrun rule.** If upstream spend exceeds plan, fence the 24 blind first, then fund the 8 two-task bundles, then the four-task bundles. If fewer than 8 two-task bundles fit after the fence and reserve, batching is recorded as **untested** and the verdict caps at Narrow. **Never cut the blind 24 to fund batching.**

---

## 7. Measurement accounting

**Call row** (one API attempt): `call_id`, `run_id`, `turn_idx`, `attempt_n`, `model_sent`, `model_echoed`, `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `reasoning_tokens` (absent → 0), `stop_reason`, `http_status`, `latency_ms`, `tool_names[]`, `tool_result_bytes`, `cost_usd`, `request_hash`, `system_prompt_hash`, `tools_hash`, `timestamp_utc`.

**Run row** (one task): `run_id`, `task_id`, `task_type` ∈ {`null_probe`, `single`, `batched`}, brick counts, context/corpus bytes, document hashes, `call_ids[]`, totals, tool counts, `had_retry`, quality score, `accepted`, runtime/pricing hashes, `git_sha`, `evidence_class`.

**Invariant:** `total_input_tokens == sum(call.input_tokens)`, and likewise for output.

**Exclusion codes:** `model_mismatch`, `cache_nonzero`, `reasoning_nonzero`, `truncated_max_tokens`, `max_turns`, `unrecoverable_error`, `path_violation`, `schema_invalid`. Transport 429/5xx → new `run_id`; the failed attempt is retained but not fitted. **Non-transport exclusions above 10% → Stop.**

`cost_usd` is computed from `pricing.json`, never taken from the model. Campaign dollars must reconcile to the provider billing export within **1%**, or every dollar claim is void.

**Idempotency:** `run_id = sha256(task_id | canonical_doc_ids | runtime_hash | pricing_hash | repeat_idx)`. Existing ids are skipped, so a killed grid resumes without duplicates.

**Quote-time features only.** Permitted: `brick_counts.*`, `unit_count`, `available_context_bytes`, `rendered_prompt_bytes`. Forbidden: tool-call counts, observed output length, cache fields, anything post-run.

---

## 8. Verdict policy (frozen before unseal)

Applied mechanically at the end by `verdict-policy.json`. Phase-local gates feed it and cannot override it.

| Gate | PASS | NARROW | STOP |
| --- | --- | --- | --- |
| Blind total \$ MAPE | ≤ 15% | 15–30% | > 30% |
| Variable-portion lift vs best of B0–B4 | ≥ 20% and bootstrap LB > 0 | positive, < 20% | ≤ 0 |
| Intervals | ≥ 20/24 hits at 90%; median rel. half-width ≤ 25%; P90 ≤ 40% | hits 18–19/24, or width gates fail → ranges unusable | hits < 18/24 |
| Encoder | exact-match ≥ 70% or priced error ≤ 20% | priced 20–40% | > 40% |
| End-to-end quote MAPE | ≤ 20% | 20–35% | > 35% |
| Batching saving | point ≥ 40%, 95% LB ≥ 20%, $s>0$ on ≥ 10/12 bundles | positive, below bar | CI includes 0 |
| Batching quality | lower 90% bound $\Delta Q > -5$ pts; batched critical failures ≤ separate + 1 | 1 brick fails → drop it and recompute | ≥ 2 bricks fail |
| Non-transport exclusions | ≤ 10% | — | > 10% |
| Null-probe drift | ≤ 2% | 2–5% → refit and disclose | > 5% → campaign invalid |
| Human IAA (P3) | exact-vector ≥ 85%; α macro ≥ 0.80, no brick < 0.70; MAE ≤ 0.25 | — | fail → do not spend the 96 |
| Identifiability / N4 | units ⟂ bytes estimable; Spearman $\lvert\rho\rvert \le 0.20$ | caps verdict at Narrow if collinearity is unbroken | — |

### Composition matrix

| Prediction | Batching | Verdict |
| --- | --- | --- |
| All point gates pass; intervals usable | ≥ 40% saving, quality non-inferior | **Feasible** |
| Point gates pass | cheaper but quality fails | **Narrow** — quote; do not recommend batching |
| Point gates pass | untested, α fail, or < 8 bundles | **Narrow** — quoting feasible; batching unproven |
| Lift fails, but a units+context baseline still meets MAPE/vWAPE/bias | any | **Narrow** — ship a runtime estimator; **hide per-brick prices** |
| Points pass; intervals too wide or hits < 18/24 | any | **Narrow** — point quotes only |
| MAPE / vWAPE / CIK-majority / bias fail | not run | **Stop** |
| Freeze, leakage, or budget integrity fail | any | **Stop — invalid** |
| N4 collinearity unbroken | any | **Narrow (cap)** |

**Reading the outcomes.**
**Feasible** = bricks are a scoping instrument worth expanding to a second model and harness.
**Narrow** = boot, context and unit count are predictable but bricks add little, or only some bricks batch safely → ship a harness-overhead estimator, not project budgeting.
**Stop** = no skill over baselines, unusable encoder, drift, integrity failure, or batching that wrecks quality.

If total MAPE passes *only because boot dominates*, the variable-portion gate forces Narrow. That is intentional and is the point of the whole design.

---

## 9. Phase detail

### P0 — Contract and evidence audit ($0)

| ID | Work | Acceptance |
| --- | --- | --- |
| P0.1 | Pin every audit reference at `c63702d` | Each cited claim is an immutable commit URL or an explicit "missing". |
| P0.2 | Recompute offline figures from committed JSONL | `docs/evidence/claim-ledger.json`: formula, n, rows, provenance class per number. |
| P0.3 | Variable-portion rescore of the 39 rows vs constant-boot, mean, bytes-only, published form | Deterministic script. Informs the prior (Feasible vs Narrow expectation). **Cannot alone Stop the new campaign** — if bricks add nothing even on author labels, the encoder epic is rescoped as a control study. |
| P0.4 | Quarantine unsupported claims | `docs/evidence/excluded-priors.md`; grep gate in CI. |
| P0.5 | Lock runtime, workload, schemas, metrics, verdict table, spend envelope | `frozen/{runtime,pricing,budget}.json`, `schemas/{call,run,exclusion,pricing}.json`, `analysis/{metrics,verdict}.py`. Boundary fixtures return Feasible / Narrow / Stop exactly as §8. |
| P0.6 | Paid-run readiness checklist | `docs/poc-readiness.md`. All red until later phases fill it. No campaign starts unless every box is green. |

**Done when:** `docs/poc-contract.md` is committed and hashed, and the verdict evaluator plus the historical rescore run in CI with **zero API calls**.

### P1 — Measurement harness and G0

Builds the only runtime we will ever predict. Does not fit a model, pull EDGAR, label bricks, batch, or demo a quote.

Runtime constants in `frozen/runtime.json`: `max_tokens` 2048 (N5 overrides 64 vs 2048), `max_turns` 12, `temperature` 0. `max_tokens` truncation → exclude unless the rung intended it. Retries only on 429/5xx/connection; each attempt is its own call row; the run is flagged `had_retry`.

**Null-probe ladder** — synthetic pads only, never enters the fit:

| Rung | Setup | Isolates |
| --- | --- | --- |
| P0r | `"ping"`, no system, no tools | provider floor vs harness boot |
| N0 | frozen system, no tools, "Reply with the single word DONE" | system-prompt weight → **source of $\hat{b}$** |
| N1 | system + 3 tool schemas declared but unused | boot constant $C_0$ |
| N2 | force one `read_file` of a pinned 1024-byte pad | one tool round-trip |
| N3 | same instruction over pads 1 / 4 / 16 / 64 KB | byte slope; monotonicity |
| N4 | fixed 16 KB total as 1×16 / 2×8 / 4×4 KB docs | **units ⟂ bytes** |
| N5 | N2 input, `max_tokens` 64 vs 2048 | output is demand-driven |
| R | N2 and N3-16KB, 5× each | CV: input ≈ 0, output/turns > 0 |

N4 fits `tokens ~ a + b·bytes + c·n_docs`. If $c \approx 0$ within rung-R noise, the grid may treat bytes as the size knob. If $c$ is real, P2 must orthogonalise. Report it; do not smooth it. 250 KB probes are rejected.

**G0 smoke** (≤ 10 billed calls, only after the fixture suite is green): N0×2, N1×1, N2×2, R×5. Green requires all of: pinned snapshot echoed, `cache_* == 0`, `reasoning == 0`, itemised usage present, cost arithmetic reconciles, N2 input CV < 0.5%.

> **G0 is redefined from the earlier draft.** It does **not** check whether our boot matches ~29,800 — it cannot and should not, because that number belongs to a different runtime. G0 checks that *measurement is possible*. A boot far below 29,821 is the expected result, and is documented as a runtime artefact.

Work items: schemas + frozen stubs (`runtime_hash` equals a committed literal, so a prompt edit fails CI) · tool sandbox (traversal rejected, deterministic bytes, oversize capped, no network, errors returned as strings to the model) · agent loop + fixture transport (fixture N2 multi-turn → byte-identical `calls.jsonl`; suite green with `ANTHROPIC_API_KEY` unset) · aggregation and exclusion (injected fault becomes an exclusion row, never a hole) · pricing (golden vectors; the same function online and in `verify`) · ladder and pads (`fixtures/probe_docs/` at exactly 1/4/8/16/64 KB) · CLI + budget guard (dry-run is \$0 and refuses if the projection exceeds `frozen/budget.json`; kill mid-grid then rerun produces no duplicate `run_id`) · `make repro` + CI green with no key.

### P2 — Workload and dataset manifest

Freezes one public EDGAR workload, a leakage-safe 10/5/5 issuer split, 96 pre-declared single-brick tasks, and a fixture-backed runner.

**Lock:** 20 CIKs × 2 filings — original 10-K (filed calendar 2025) + first subsequent original 10-Q (filed ≤ 2026-06-30). Primary document only; no amendments, exhibits, slicing, or synthetic pads. After freeze, only pinned `(CIK, accession, form, primary_document)` tuples are legal.

- **Train (10):** Apple `0000320193`, AMD `0000002488`, Adobe `0000796343`, Walmart `0000104169`, P&G `0000080424`, JPMorgan `0000019617`, Exxon `0000034088`, Pfizer `0000078003`, Caterpillar `0000018230`, FedEx `0001048911`
- **Val (5):** Microsoft `0000789019`, Costco `0000909832`, Visa `0001403161`, J&J `0000200406`, UPS `0001090727`
- **Blind (5):** NVIDIA `0001045810`, Salesforce `0001108524`, Coca-Cola `0000021344`, Chevron `0000093410`, Deere `0000315189`
- **Pre-freeze substitutes only:** Intel, Target, BofA, Merck, Boeing. Substitution rewrites `sources.jsonl` and restarts checksums.
- **Blocklist:** Progress Software, Franklin Covey, CIKs `0000874817` / `0000886206`, every id implied by the legacy `train_runs.jsonl` / `decompose_cases.jsonl`, and `gc/prototype`.

**Grid (96):** 4 bricks × 3 unit levels {1, 2, 4} × 2 context bands × replicates (train ×2 CIKs, val/blind ×1) = 48 / 24 / 24. Each task is **one** brick; `gold_counts` is a 4-vector with a single non-zero entry equal to units. Mixed and batched prompts are not paid rows here.

**Context bands, without slicing:** low = the minimum documents needed to answer; high = the required set plus same-split decoys, such that median `available_context_bytes` ≥ 2× the low median for that brick×split. The quote-time regressor is `available_context_bytes`. Pre-registered orthogonality on train: Spearman $\lvert\rho(\text{units}, \text{bytes})\rvert \le 0.20$. Failure → rebalance decoys; **never slice documents**.

**Acquisition:** direct HTTPS to EDGAR, `EDGAR_USER_AGENT` from the environment (do not invent one), 2 req/s, raw HTML plus a pinned normalisation (drop `script`/`style`/inline XBRL, NFC, LF), both hashed. No `sec-edgar-downloader`, no `pandoc`, no LLM. Public filings are not encrypted; blind tasks are sealed by API — `run_workload --split blind_test` requires `--frozen-model-id`.

**Brick runnable forms:** `extract` → named facts + citations · `validate` → supported / contradicted / not_found · `reconcile` → same metric across 10-K and 10-Q · `retrieve` → locate within a mounted directory. Targets are generated without model calls.

**Done when:** 40 hash-verified documents and 96 frozen tasks exist; leakage, blocklist and orthogonality checks are green; the fixture runner is isolated, append-only and blind-blocked; and a ≤ 10-call train smoke shows itemised usage with zero cache — or is explicitly marked *blocked*, never silently skipped. The 96 remain unexecuted.

### P3 — Taxonomy and encoder validation

Zero workload dispatches. The blind 24 are sealed for the whole phase.

| Brick | One sentence | One unit | Nearest neighbour |
| --- | --- | --- | --- |
| `extract` | Return named values from specified documents. | one requested field | Named filing + named value → extract, even if the verb is "find". |
| `validate` | One independently scorable check against a specified source. | one assertion | One value vs a rule → validate. Two sourced values made to agree → reconcile. |
| `reconcile` | Same named item across two specified sources or periods. | one comparison | Fetching the two numbers is not reconcile. |
| `retrieve` | Locate a fact when the source is not specified. | one search target | Unknown source → retrieve. Parsing a found section is extract, and is not added if the request only locates. |

Count **requested targets** — not documents, rows, hits, or tool calls. Reading for another brick does not add a brick. `out_of_scope` and `needs_clarification` (all-zero vectors) are first-class outcomes.

**Annotation (trimmed from the earlier draft).** One trained annotator produces gold on all 72 train+val tasks using the frozen card. A **stratified 24-task subsample** is independently double-annotated with a third-party adjudicator, and inter-annotator agreement is computed on that subsample. Fallback if only one person is available: same annotator, ≥ 72 h apart, reshuffled, reported explicitly as *intra*-annotator on the scorecard. This preserves a measurable α while roughly halving the human ask, which the earlier draft itself flagged as a single point of failure.

**Encoder A:** pinned Haiku, `temperature=0`, tools off, JSON contract, fail-closed. One counted parse retry. No keyword fallback on scored rows.
**Encoder B:** deterministic keyword/regex, written from the card *before* gold is seen. If B matches A within confidence intervals, **ship B** — it is free and deterministic.
**Encoder C:** optional second vendor. Never a gate.

**Human gate:** exact-vector ≥ 85%; Krippendorff α (ordinal) macro ≥ 0.80 with no brick below 0.70; MAE ≤ 0.25; zero unresolved. α < 0.67 on any brick halts the phase.
**Encoder gate (val):** exact-vector ≥ 70%; macro presence F1 ≥ 0.85, no brick below 0.75; MAE ≤ 0.35; canonical-vs-paraphrase stability ≥ 85%; format/API failure ≤ 2%.
**Construct validity:** if gold disagrees with generator intent on ≥ 10% of train, **halt** and fix the P2 quote wording. Fitting on ambiguous text is more expensive than rewriting it.

**Priced error:** the formula is fixed now; the weights $w_b$ arrive from P4 train-only data. Provisional dollars must not pass or fail this phase.

**Outcomes, written before val numbers are seen.** Pass → freeze A. Narrow-keyword → ship B. Narrow-human-bricks → P4–P5 run on given vectors and the demo shows human-supplied bricks. Stop-signal → human or construct-validity failure; do not spend the 96.

### P4 — Decoder, baselines and uncertainty

Fit vectors are **human gold 4-D on the 48 train tasks only**. Encoders A and B are applied after the freeze, with no refit.

**Candidates — no others.** D1 OLS, D2 NNLS, D3 Ridge (α ∈ {0.001, 0.01, 0.1, 1, 10, 100}, inner grouped CV) on

$$\hat{y}_{var} = \beta_c \cdot \text{bytes} + \beta_p \cdot \text{prompt\_bytes} + \sum_{j=1}^{4} \beta_j u_j$$

**Selection:** leave-one-CIK-out GroupKFold over the 10 train CIKs; mean fold vWAPE; one-SE rule toward the simpler model, D1 ≺ D2 ≺ D3. Val is opened **once**. A negative brick coefficient in a winning D1 → take D2 if within 1 SE, otherwise the verdict caps at Narrow. Negative coefficients are reported, never silently clipped.

**Baselines, implemented before any decoder:**

- **B0** = $\hat{b}$ — the fixed-cost illusion made explicit
- **B1** = train median $y$
- **B2** = boot + context bytes
- **B3** = boot + prompt bytes
- **B4** = boot + context + prompt + **total units** — the real competitor
- **B5** = cell mean — a diagnostic ceiling, excluded from the beat-baseline gate

**Intervals:** a heteroscedasticity check on train CV chooses absolute vs relative nonconformity *before* val is opened; split conformal on the 24 val residuals; 90% frozen as the demo default; jackknife+ on the 48 as a cross-check. Coverage is **not** gated on val — prospective coverage is a P5 result only.

**Repeats:** 4 (one per brick, units = 1, low context). If CV of $y$ exceeds 5%, the 20% vWAPE gate is restated as $\max(20\%, 3 \cdot \mathrm{CV})$ **before** val is opened.

**Identifiability:** full rank; condition number < 30; VIF < 5; post-execution Spearman $\lvert\rho\rvert \le 0.20$. Failure → do not fit per-brick $\beta_j$; fall back to the B4 form and cap the verdict at Narrow.

High-token rows, refusals and tool loops **stay in**. Only pre-registered infrastructure exclusions are removed. Exclusions above 10% of 72 halt the phase.

**Sanity tests the fit must pass on fixtures:** two fits are byte-identical · group folds hold disjoint CIKs · synthetic recovery of $\hat{m}_b$ within 10% and correct form selected in ≥ 90% of 50 seeds · shuffled-$y$ CV vWAPE collapses to B1 · a price-ratio change updates $y$ · zero units returns $\hat{b}$ · no blind path is importable.

**P4 acceptance (val, gold vectors):** 72 valid rows or permitted exclusions ≤ 10%; identifiability holds; total MAPE ≤ 15%; vWAPE ≤ 20% (or the restated floor); ≥ 15% relative vWAPE improvement over the best of B0–B4; beats that baseline on ≥ 4 of 5 val CIKs; P90 APE ≤ 30%; $\lvert\text{bias}\rvert$ ≤ 10%; Encoder A priced variable error ≤ 20%; the 90% interval is frozen with coverage **unclaimed**.

Stop here → do not spend the blind 24.

### P5 — Freeze, blind run, batching × quality, demo, verdict

Nothing is refit after the seal breaks. Two **separate** experiments: prospective prediction on the sealed 24, and batching × quality on train+val CIKs only. A good predictor cannot rescue failed batching; batching savings cannot rescue a failed predictor.

**P5.0 — Freeze the candidate (no spend).** `frozen/release/v1/manifest.json` with a `system_hash`; `python -m poc.freeze verify` fails on any byte change, and reloading the decoder reproduces P4 train/val predictions bit-for-bit. Live dispatch requires `--release-id --frozen-model-id --ledger-id`; fixture mutation of any protected field must dispatch **zero** API calls. `verdict-policy.json` is hashed into the manifest before unseal, and the generator reads only that file.

**P5.1 — Pre-register the ledger (no spend).** Append-only `evidence/v1/dispatch-ledger.jsonl` written before any P5 call, with a signed randomisation of dispatch order, bundle membership, A/B display order, and bootstrap seed. Verify: exactly 24 prospective rows from the sealed blind sidecar; 3 rows per two-task bundle and 5 per four-task; no blind CIK appears in `experiment=batching`; planned + already-spent ≤ 220; inserting a row after start changes the ledger hash. If fewer than 8 two-task bundles fit, lock **batching: untested** now.

**P5.2 — Prospective blind test (spends 24).**
Blind gold stays sealed — do **not** label or score encoder exact-match on blind. Blind scores quote against `response.usage` only. The invariant `quote_committed_at < dispatch_started_at < actual_recorded_at` is enforced, and the runner refuses any task without a committed quote. No repeats except infrastructure replacements; more than 3 exclusions of 24 **voids the blind set**. Bad answers, timeouts and tool loops are outcomes, not grounds for replacement.

Gates: total MAPE ≤ 15%; vWAPE ≤ 20%; relative lift over the best frozen B0–B4 ≥ 15%; ≥ 4 of 5 CIKs at vWAPE ≤ 20%; P90 APE ≤ 30%; $\lvert\text{bias}\rvert$ ≤ 10%; 90% interval hits ≥ 20/24 with a Wilson CI reported (**never** "90% proven" from n=24); median relative half-width ≤ 25%; P90 half-width ≤ 40%. CIK-clustered bootstrap over 5 clusters is **descriptive only**.

**Spend gate.** Material MAPE / vWAPE / CIK / bias failure → **Stop**, batching is not run. Lift-only failure → **Narrow estimator**, continue to P5.3, and the demo must not show per-brick rates.

**P5.3 — Batching × quality (spends 39 if P5.2 allows).**
Corpus: 8 two-task bundles (primary) + 3 four-task bundles (scaling), on train/val CIKs only. The separate arm is **re-run**, not reused from the 72/24. Per-task instructions and source access are byte-identical across arms; the batched arm may add only neutral delimiters and "answer all sections". Fresh session per dispatch.

$$s = \frac{y_{sep} - y_{batch}}{y_{sep}}$$

decomposed into (a) $\hat{b}$ paid once instead of *N* times — **predicted** — and (b) residual variable subadditivity — **measured minus predicted**. This decomposition is what distinguishes "the 40% bar was missed because our owned boot is small" (Narrow) from "there is no saving" (Stop).

Grading: ≥ 2 graders plus an adjudicator, contracted **before** generation. 100 points — factual 45 / completeness 25 / citations 15 / no fabrication 10 / structure 5. Critical failure = wrong numeric, fabricated fact, missed central reconciliation, or unusable result. Grade files carry no arm label until join; α ≥ 0.67 on weighted scores is required before unblinding.

Gates: unit of analysis is the **bundle**; 10,000 paired bootstrap; aggregate $s$ ≥ 40% both overall and for two-task alone; $s > 0$ on ≥ 10/12 bundles (or ≥ 7/8 if only two-task ran); lower 90% interval for $s$ strictly > 0; lower 90% bound on $\Delta Q$ > −5 points; batched critical failures ≤ separate + 1. Four-task is secondary — saving should rise roughly as $(N-1)\hat{b}$. α < 0.67 → quality inconclusive. Fewer than 8 two-task bundles → untested. Always log last-position dropout, cross-task contamination, and truncation.

**P5.4 — Replayable demo (no eval spend).** Standard library only.

```
python -m demo quote    --task-id BLIND-017
python -m demo replay   --task-id BLIND-017
python -m demo batching --bundle-id BATCH-006
python -m demo verdict
```

`quote` prints the encoder vector, the point estimate, the 90% band, the baselines and the `release_id`. `replay` prints actual $y$ and dollars, APE, interval hit/miss, and the hashes proving the quote preceded the run. Two runs are byte-identical except for a disabled clock; replay makes no network calls; every printed number derives from JSONL; hash or row corruption fails **visibly**. A Narrow-estimator verdict hides per-brick rates. `--live` refuses unless `system_hash` verifies, is non-ledger, and is bannered *"single sample, not evidence."*

**P5.5 — Evidence package and mechanical verdict.** `python -m evidence.build --release frozen/release/v1/manifest.json` rebuilds every derived output into `evidence/releases/v1/` — ledger, quotes, runs, exclusions, prediction results, batch files, grades, blinding manifest, policy, verdicts, transcript, `checksums.sha256`. No rater identity, no API metadata. A fresh checkout with no credentials must reproduce the verdict, and fault injection must flip only the gate it was fed.

---

## 10. Blockers

No live spend in a phase until its blockers are true.

- **P1 smoke:** `ANTHROPIC_API_KEY`; snapshot still served (asserted on call #1); freeze-day prices may remain provisional so long as dollars are internally consistent and `verify` warns.
- **P2 acquisition:** a real `EDGAR_USER_AGENT`. Exact accessions stay open until W1–W2 run.
- **P3 encoder A:** P2 quote text frozen (EDGAR bytes are not required); two annotators or a documented intra-annotator protocol; encoder ledger billed separately.
- **P4:** a valid N0 boot; freeze-day `pricing.json`; P2 48/24 manifests with hashes; P3 gold with IAA passed; snapshot echo; billing reconciliation; EDGAR provenance; the N4 result; identifiability.
- **P5:** P4 wrote a *real* winning decoder plus frozen B0–B5 and conformal intervals — not placeholders; snapshot still served; official usage fields; freeze-day prices stamped (a provisional ratio cannot support a final dollar claim); append-only ledger giving exact prior spend; 24 blind hashes intact; two graders and an adjudicator engaged before batch generation; batch gold and rubric frozen; cache-zero / fixture / jail / manifest tests green; ≥ 8 two-task bundles fit or batching explicitly untested; N4 recorded.

Optional and non-blocking: if Claude Code is available after smoke, ≤ 10 runs reporting `subagent_tokens` / `billed_tokens` for context. It does not change the POC harness. Ask the upstream author once for the original runner, traces and corpus; if absent, mark the historical campaign permanently non-reproducible and move on.

---

## 11. Risk register

Priced into the gates. Not to be "fixed" in analysis.

| Risk | Handling |
| --- | --- |
| Fixed-cost illusion | vWAPE is mandatory; B0 *is* the illusion, stated as a baseline. |
| Encoder is the fundability risk | Own phase, priced error, no silent fallback. |
| Harness artefact | We predict *this* prompt and tool set, not Claude Code's ~30k. |
| Quality confound on batching | Separate experiment; $\Delta Q$ gate; critical-failure count. |
| Circularity | The same model family decomposes and executes. B is the control. Do not claim independent validation unless C runs. |
| Small n / form selection | Pre-registered D1–D3 only; grouped CV; one-SE; no extra forms. |
| n = 24 over 5 CIKs | Coverage is a screen plus a Wilson CI. Bootstrap is descriptive. |
| Tool cascades, truncation, retries | Cap and exclude infrastructure only; `had_retry` rows stay out of slope fits unless P4 says otherwise. |
| Snapshot retirement | Do not retarget. The freeze becomes invalid. |
| Small owned boot | The 40% batching bar may fail even if the mechanism is real; the decomposition distinguishes "bar missed" from "no saving". |
| Grader availability | Single point of failure for the quality arm; contract before generation. |
| Construct validity | Gold ≠ generator on ≥ 10% of train → halt. Cheaper than fitting on noise. |
| Keyword overfit | B is frozen before gold is seen. |
| Intra-annotator fallback | Disclosed on the P5.5 scorecard. |

---

## 12. Merge record — which draft won, and where

Two drafts were reconciled. **Draft 2 (phased contract) is the stronger plan and forms the spine.** Draft 1 (four-gate GO / NO-GO) contributed six controls that Draft 2 lacked. The nine substantive conflicts were resolved as follows.

| # | Conflict | Resolution | Source |
| --- | --- | --- | --- |
| 1 | Target: raw tokens vs billable dollars | **Dollars.** `input+output+reasoning` summed is not cost; output is ~5× input. Historical rows have no split, so they cannot support a dollar claim at all. | Draft 2 |
| 2 | Reuse the committed coefficients vs refit on an owned harness | **Refit.** The 29,821 intercept is Claude Code `subagent_tokens` on an unowned runtime. Reusing it on a new harness compares two different things. | Draft 2 |
| 3 | Nine bricks vs four | **Four.** Nine free coefficients cannot be identified from 8 eval cases, and Draft 1 proposed exactly that. | Draft 2 |
| 4 | Accuracy gate: "mean APE < 10%" vs baselines + vWAPE | **Baselines + vWAPE.** Draft 1's gate is passable by `return 31000`. This is the single largest correction. | Draft 2 |
| 5 | 8 hand-authored cases vs 96 pre-declared, leakage-split tasks | **96 with issuer-level splits.** Draft 1 has no leakage control and no corpus provenance beyond "SEC-style". | Draft 2 |
| 6 | Point estimates vs conformal intervals | **Intervals.** A budget number with no band is not usable, and Draft 1 offered none. | Draft 2 |
| 7 | Batching measured on cost alone vs cost × quality | **Cost × quality.** A cheaper worse answer is not a saving. Draft 1's G4 would have scored one as a pass. | Draft 2 |
| 8 | G0 threshold: "boot within 10% of 29,800" | **Redefined, kept.** Draft 1 was right that a named, cheap, blocking runtime gate belongs first — and wrong about its threshold, which would have failed by construction. G0 now tests *whether measurement is possible*. | **Both** |
| 9 | Ceremony | **Trimmed.** Draft 2's ladder was budgeted at 10 dispatches in one table and 30–40 in another; the envelope is corrected to 220. Three annotators on 72 tasks became one annotator on 72 with a double-annotated 24-task IAA subsample. Four-task bundles cut 4 → 3. | merge |

**Adopted from Draft 1 in full:** the evidence-class labels (`replay` / `pipeline_only` / `feasibility`) stamped on every output · the exit-code discipline, and the rule that a partial run must never print PASS · G0 as a single named blocking artefact · the verdict *matrix* presentation, now driving Draft 2's mechanical `verdict-policy.json` · the four independently-scored claims G1–G4 as the framing device · the 10–15 minute demo script shape.

**Adopted from Draft 2 in full:** the dollar target, the owned harness and null-probe ladder, the frozen workload with issuer splits, human gold with IAA, baselines B0–B5, the vWAPE lift gate, conformal intervals, the batching quality arm, the append-only pre-registered ledger, and the Feasible / Narrow / Stop composition matrix.

**Net effect.** Draft 1 could have produced a confident number that meant nothing — a constant predictor scoring 3% MAPE, on 8 self-authored cases, with reused coefficients from a different runtime, and a batching "saving" measured without ever grading an answer. Draft 2 could have produced nothing at all, by never escaping its own scaffolding. The merge keeps Draft 2's scientific spine and Draft 1's legibility and shipping discipline.

---

## 13. Demo-ability checklist

What "feasible or not" looks like on a laptop, with no credentials:

1. Clone + `make repro` → fixture suite green, \$0, no API key required.
2. `python -m demo quote --task-id BLIND-017` → encoder vector, point estimate, 90% band, baselines, `release_id`.
3. `python -m demo replay --task-id BLIND-017` → actual $y$ and dollars, APE, interval hit/miss, and the hashes proving the quote preceded the run.
4. `python -m demo batching --bundle-id BATCH-006` → matched separate vs batched, saving split into boot-once and residual subadditivity, blinded $\Delta Q$ — or an explicit *untested*.
5. `python -m demo verdict` → exactly one of **feasible** / **narrow** / **stop** from the frozen policy, with every gate, bar and observed value listed.

That is the POC.

- **Feasible** → next phase (not this plan) is a second model, a second harness, larger *n* for real coverage, and mixed bricks.
- **Narrow** → ship a runtime / context / units estimator. Do not sell per-brick pricing.
- **Stop** → publish the negative result. Do not expand the model or the domain.
