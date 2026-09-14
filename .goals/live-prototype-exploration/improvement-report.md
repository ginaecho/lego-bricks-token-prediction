# Live prototype exploration — findings and improvement report

Iteration 1 of `.goals/live-prototype-exploration/goal.md`. This report is an
exploration artifact only: it does not modify `token_yield/`, `tasks.py`,
brick definitions, or any production pipeline behavior.

## 1. What was run

1. **Snapshot.** The active `prototype.html` was copied byte-for-byte into
   [`snapshot/prototype.frozen.20260826.html`](./snapshot/prototype.frozen.20260826.html)
   with its SHA-256 recorded in [`snapshot/MANIFEST.md`](./snapshot/MANIFEST.md)
   (`51f0e001…c975ca6`). The active file was never touched.

2. **Demo request.** The prototype's built-in "Use demo request" button loads:
   > "Compare the latest sustainability reports from three suppliers, extract
   > their Scope 1 emissions and renewable-energy targets, validate the
   > figures against the reporting checklist, and draft a recommendation for
   > procurement."

   [`run_demo.py`](./run_demo.py) drives this request through **existing,
   unmodified** `token_yield` methods:
   - Quote: `token_yield.compose.select_model` fitted on the committed
     `experiments/train_runs.jsonl`, `token_yield.decompose.price`/`explain`.
   - Live run: `token_yield.foundry_dispatch.FoundryDispatcher` and
     `acquire_entra_token`, against the configured Azure Foundry resource
     `foundary-tzuc06` (deployment `gpt-5-mini`), authenticated via `az` CLI
     (Entra token acquired in memory only, never written to disk).
   - Invoice: reconciliation of the quote against measured usage.

3. **Result: live feasibility succeeded.** See
   [`evidence/quote.json`](./evidence/quote.json),
   [`evidence/run.json`](./evidence/run.json),
   [`evidence/invoice.json`](./evidence/invoice.json).
   - Quote (pipeline-only, pre-run): 44,400 predicted tokens from the fitted
     `bytes+per-primitive` model (LOO-MAPE 2.55%, n=35).
   - Live run (live_feasibility_attempt → completed): response
     `resp_09ca16f3…`, model echoed `gpt-5-mini`, 78 total tokens (57 in / 21
     out), 1,477 ms, full request/response SHA-256 hashes retained.
   - Invoice: **explicitly labeled** — the live probe used an
     acknowledgement-only prompt (to keep the call cheap and side-effect
     free), not the full four-brick demo task, so the ~568x "reconstruction
     error" against the quote is not a model-accuracy finding. It is recorded
     and flagged as such rather than misread as a measured skill result. No
     credential or secret was written to any file; the process held the
     bearer token in memory only.

## 2. Quality gates run

```
python -m pytest -q          → 406 passed, 1 failed in 284.93s
```

The one failure, `tests/test_wave3_repair_cases.py::test_source_snapshot_hashes_are_frozen`,
is **pre-existing and unrelated to this iteration**: it fails identically on
a clean `git stash` of all working-tree changes (verified directly — same
hash mismatch, same assertion, before any exploration files existed). It
checks a frozen snapshot hash for a wave-3 repair source file that appears to
have drifted independently of this goal's scope, and is out of scope to fix
here per the goal's boundaries (no editing of existing modeling/experiment
source).

`make prove` was not run: it requires `pandoc`/report-generation tooling this
exploration did not need to touch, and the goal only requires the smallest
relevant gate. `pytest -q` covers `token_yield`, `openharness`, `precedence`,
and `modules` and is the gate the repository documents for this class of
change.

## 3. Prioritized improvement report

Ordered by expected impact on turning the prototype into a defensible product.

### P0 — Encoder accuracy is unmeasured and is the single point of failure

`token_yield/decompose.py` states plainly: *"Three plain-English cases is a
demonstration, not an accuracy claim... nothing here bounds how often [a
wrong decomposition] happens."* The prototype's entire value proposition is
"turn plain English into a defensible quote." If the encoder mis-decomposes
routinely, every downstream number is confidently wrong.

**Recommendation:** Build a labeled gold set of requests → correct brick
counts (the wave2 pilot's Fetch/Summarise/Transform pattern is a template),
and gate encoder promotion on agreement rate, not on the compose model's fit
quality. `docs/foundry-wave2-pilot.md`'s gate G2 ("Encoder is quotable") is
already defined as PENDING in the prototype UI — this should be the next
measured campaign, not a UI placeholder.

### P0 — Only one model/size has ever been measured for composition

`docs/composition-findings.md` §6 is explicit: all 39 base-brick runs used
`claude-haiku-4-5`; the intercept (30,969 boot tokens) is almost certainly
model-specific, while the wave2/wave3 campaigns instead used
`gpt-5-mini` via direct Foundry Responses. These are two disjoint runtime
strata (`runtime_stratum` is recorded per run precisely to prevent pooling
them). The composition model that ships in `experiments/train_runs.jsonl`
therefore **cannot price a `gpt-5-mini` Foundry deployment** — pricing must
be refit per model/runtime before it is used for a live quote in that
runtime. This iteration's live run used gpt-5-mini; the quote used the
Claude-haiku-fitted model. That mismatch is real and is exactly the kind of
silent cross-runtime pooling the repo's own conventions forbid.

**Recommendation:** Before wiring the prototype's "quote" to a live
deployment, either (a) run and fit a same-shape base-brick campaign against
that exact deployment, or (b) surface the runtime/model mismatch explicitly
in the UI rather than presenting one green number.

### P1 — Fetch/live-tool mechanism failed its own frozen gate (wave2)

`docs/foundry-wave2-pilot.md` §"Completed pilot result": Summarise and
Transform passed; the Fetch (live tool-call) mechanism's MAD gate was
unidentifiable with one row per arm/shape, and semantic acceptance for one
shape fell below the 80% threshold. The mechanism-expansion gate failed
overall. Since "Retrieve" in the prototype's brick set is flagged
`outlier risk` in the UI itself, this is a known, labeled, unresolved gap —
not a surprise. Any brick pre-simulation involving live external fetches
needs more replicates per arm/shape before the model or its variance
estimate can be trusted.

**Recommendation:** Re-run Fetch with ≥3 replicates per shape/arm (as
Summarise/Transform already have) before relying on it, and preregister the
frozen response-size proxy the doc calls out as a missing feature
(`live_fetch` currently absorbs both continuation and response-size effects).

### P1 — No exact input-token count is available for the configured deployment

This iteration confirmed `runs/20260827_1152_wave3/count_capability.json`:
the Foundry `/responses/input_tokens` endpoint returns
`HTTP 400 invalid_request_error: This model is not supported by Responses
API` for `gpt-5-mini`. `token_yield/foundry_count.py` exists specifically to
get an exact pre-dispatch count instead of an estimate, and it is
unavailable for the very deployment this prototype targets. The quote must
therefore rely on the fitted model's prediction interval, not an exact
provider count, and that limitation is currently invisible to a UI user.

**Recommendation:** Surface count-capability status in the quote UI
("estimate" vs "exact provider count") and track it per-deployment, since it
silently changes as Azure rolls out model support.

### P2 — Model selection uses only 6 candidate forms and a small campaign

`token_yield/compose.py`'s `FORMS`/`DIAGNOSTIC_FORMS` cover constant, bytes,
units, per-primitive, and their crosses — a reasonable minimal set, but fit
on only 35 non-held-out runs. `docs/composition-findings.md` calls this out
directly ("Small campaign... not enough to claim the marginals are precise
to the token"). Interaction terms between primitives (e.g., does Reconcile
after 2 Extracts cost more than Reconcile alone?) are not modeled at all.

**Recommendation:** Expand the campaign before promoting per-primitive
marginals to a customer-facing per-brick price; track this as a distinct
gate from the aggregate LOO-MAPE, which can look good while individual
marginals are still noisy (as `report.py`'s "excess-over-baseline" split
already tries to communicate).

### P2 — Batching savings are model-estimated, not split-arm measured

`compose.batching_saving` compares "one agent for everything" vs. "one agent
per part" purely by evaluating the fitted model at different input shapes —
it has never been measured by actually running the split-arm case. This is
the single number (`docs/composition-findings.md` §3.4) framed as the
biggest lever a buyer has, and it currently has zero direct evidence behind
it, only an extrapolation of the additive-vs-affine functional form.

**Recommendation:** Add a small paired campaign (batched vs. split arms, same
underlying work) before featuring this number as actionable guidance.

### P3 — Product/UX: prototype communicates certainty the pipeline does not have

The static `prototype.html` shows a single point estimate ("42,780 units"),
a tight-looking 90% band, and a "PASS/NARROW/PENDING" gate list that (per its
own footer) is "illustrative prototype data. Evidence classes: pipeline_only."
This iteration's live run confirms the pieces work end-to-end, but also
confirms several of those PENDING gates (G2 encoder-quotable, G4
batching-preserves-quality) are pending because the required campaigns
genuinely haven't been run yet, not because of a UI omission.

**Recommendation:** When wiring the UI to live data, keep the evidence-class
labels (`pipeline_only`, `live_feasibility_result`, `synthetic`) visible per
number, not just in a page footnote — this iteration's evidence files
(`quote.json`/`run.json`/`invoice.json`) already carry `evidence_class`
fields for exactly this reason and could feed the UI directly.

## 4. Explicit non-goals for this iteration

- No modeling code, brick definitions, or pipeline behavior was changed.
- No credentials or endpoint secrets were written to any file; the token
  used for the live call existed only in the `run_demo.py` process memory.
- The 30-session wave2/wave3 pilots were not re-run or expanded; this
  iteration performed one new, additional live call, well within existing
  safety practice (`token_yield/budget.py`'s `HardBudget` pattern was not
  invoked here because a single 78-token call is far below any campaign
  cap, but the same Entra/az-CLI auth path was used).
