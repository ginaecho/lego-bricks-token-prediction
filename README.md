# Token Yield — Lego bricks for token prediction

[![CI](https://github.com/ginaecho/lego-bricks-token-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/ginaecho/lego-bricks-token-prediction/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/1314056228.svg)](https://zenodo.org/badge/latestdoi/1314056228)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <a href="docs/media/Token_Yield_Explainer.mp4">
    <img src="docs/media/Token_Yield_Explainer.gif" width="820"
         alt="Token Yield — the animated explainer, looping: atomic task blocks are measured, combined, trained on, and used to price an unseen project">
  </a>
</p>

<p align="center">
  ▶ <b><a href="docs/media/Token_Yield_Explainer.mp4">Watch the full video</a></b>
</p>

We predict token consumption for AI projects by decomposing complex workflows
into reusable, **LEGO-like task blocks**. Each atomic task serves as a building
block and is represented as an input feature for model training. For a new
project, the workflow is first broken down into these smaller components, which
are then encoded and reconstructed through an **autoencoder-based architecture**
to capture project complexity and structure. The resulting representation is
used to predict overall token usage accurately.

**Why it matters:** AI budgets are set by guesswork and reconciled after the
money is gone. Token Yield turns them into a line item — a credible cost range
at scoping time, the expensive-outlier risk located before dispatch, and a
per-brick invoice that reconciles spend to accepted work. Budgeting,
forecasting and chargeback, from one model.

## How it works

```
① measure the basic bricks  →  ② measure their combinations
            ↓                              ↓
        ③ train the predictor on (1) + (2) as features
            ↓
④ new project → decompose into bricks → recompose → predicted tokens
            ↺  measured actual refits the model
```

1. **Pre-simulate the bricks** — run each atomic task for real, at several
   sizes. This is the rate card finance can hold.
2. **Pre-simulate combinations** — stack bricks and measure again. They are
   strongly sub-additive in the fitted model: batching work into one agent is
   **estimated to save 64–71%**. A measured split-arm experiment is still
   required before treating that estimate as a realized saving.
3. **Train** — brick counts + context size in, measured tokens out; model form
   chosen by cross-validation, never assumed.
4. **Predict and close the loop** — decompose a real request into bricks,
   quote it *before it runs*, then let the measured actual refit the model.
   The same decomposition is the chargeback invoice afterwards.

## Measured, not asserted

39 real agent runs over 33 genuine SEC filings; nine bricks
(Review · Extract · Classify · Retrieve · Reconcile · Draft · Remediate · Validate · Report):

| finding | business read |
|---|---|
| Start-up is **29,821 tokens** — 89% of a median task | most of what you pay is the invocation, not the work — so batch |
| Context costs **0.37 tokens/byte**, flat | more reading scales linearly; it will not blow the budget |
| **Retrieve = 5,384/unit**, 10× any other brick | the outlier tail is *located* — narrow the search before dispatch |
| Total-token CV **2.55%**; excess-over-startup error **18.73%** | the honest view: strong total prediction, with harder variable-work error exposed |

Full experiment and limitations: **[docs/composition-findings.md](docs/composition-findings.md)**

The first source-backed business-case catalog adds nine atomic tasks, three
compositions, frozen acceptance gates, an auditable live-run contract, and
accepted-work tail economics. Three independent agents accepted 11 of 12
created probes; those proofs are explicitly unmetered and are not used as token
measurements. See
**[docs/base-brick-experiment.md](docs/base-brick-experiment.md)**.

The first fully metered paired wave then trained on 25 isolated atomic and
composite sessions, froze predictions, and executed four unseen cases. Three
one-call cases landed within 2.8-3.8%; one externally verifying risk memo
branched to three model calls and missed by 66%, for **19.1% overall MAPE**.
Cross-validation selected a constant over the LEGO feature form, so this wave
is **inconclusive and underpowered**, not evidence against the architecture:
the runtime re-counted cached harness context, the byte range was compressed,
task shapes were unreplicated, and one multi-call row was highly influential.
It did establish that external branching needs its own brick. Full results:
**[docs/paired-token-experiment.md](docs/paired-token-experiment.md)**.

Wave 2 expands the vocabulary with **Fetch, Score, Summarise, Monitor, Plan,
Notify, Approve, and Transform**. Its preregistered direct-Foundry pilot crosses
realistic context and unit ladders, pairs frozen and live external API arms,
groups replicated shapes during validation, and enforces a USD 200 hard
pre-dispatch ceiling. Protocol:
**[docs/foundry-wave2-pilot.md](docs/foundry-wave2-pilot.md)**.

The completed 30-session pilot measured 94,351 known tokens. Summarise and
Transform produced strong monotonic size signals, and grouped ridge reduced
total-token MAE 69.1% versus the constant (95% grouped-bootstrap CI
48.7–89.5%). Expansion remains paused: one attempt lacked telemetry, the
frozen quality gate failed, and Fetch's requested within-shape noise was not
identifiable with one row per arm/shape. The conservative pre-run Fetch
implementation failed. The hard-cap ledger settled at USD 5.05074, far below
USD 200; this is a conservative safety amount, not reconciled Azure billing.

Wave 3 adds the mechanism-distinct **Correlate, Diagnose, and Provision**
bricks plus an auditable Big-T feature registry. Its repair-first block froze
36 cross-source sessions spanning SEC, eCFR, openFDA, USAspending, and CMS
evidence. The block measured 84,622 provider-reported tokens and accepted
24/36 outputs. It is paused as preregistered: Azure reported that the selected
`gpt-5-mini` deployment does not support `/responses/input_tokens`, six
medium-effort requests exhausted their output caps, and all six SEC Fetch cases
miscounted the source records. Snapshot/live Fetch medians were nevertheless
fully replicated and deterministic within each source/arm. Evidence:
**[runs/20260827_1152_wave3/README.md](runs/20260827_1152_wave3/README.md)**.

The next modeling seam is now implemented offline. It separates cached input,
non-reasoning output, reasoning, unforced branching/retry, attempt cost, and
acceptance; enforces `output = non_reasoning + reasoning`; rejects post-run
predictor leakage; excludes censored magnitudes; and calibrates residuals by
independent group rather than replicate. On the historical Wave 2 records,
visible/non-reasoning output achieved **108.0-token grouped MAE** versus
**113.5** for the grouped constant, only a **4.9% exploratory improvement**.
Reasoning, cache, and unforced branching remain explicitly unidentified, so
the interface returns unknown rather than zero for reconstructed totals and
costs. Protocol:
**[docs/multichannel-model-experiment.md](docs/multichannel-model-experiment.md)**.

The same model has also been retrained offline on all 36 Wave 3 repair rows
using only frozen quote-time features: payload size, declared fetch bytes,
output requirements, declared/free calls, cache assignment, effort, brick
effects, and live/snapshot arm. This calibration-only fit reduced grouped MAE
by **28.1%** for non-reasoning output and **51.9%** for semantic acceptance;
reasoning occurrence improved **57.1%**. These are diagnostic signals, not
promotion results: reasoning magnitude is censored, cached amount has only
four positive groups, and the repair block failed its preregistered gate.
No blind records entered fitting.

## Try it

```bash
pip install -e .                       # Python ≥ 3.9; installs tiktoken
python -m examples.composition_demo    # the whole loop, from the committed data
python -m examples.business_case_demo  # sourced cases + acceptance evidence
python -m examples.multichannel_diagnostic  # offline channel diagnostics
python -m examples.multichannel_wave3_diagnostic  # Wave 3 calibration-only fit
```

---

*Token Yield (`token_yield/`) stands on **HarnessDose**, the measurement layer
(package `openharness`), and composes with Microsoft's
[Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit).
Docs: [calibration](docs/calibration-findings.md) ·
[architecture](docs/architecture.md) · [precedence](docs/precedence.md) ·
[AGT](docs/agt-integration.md) · [Zenodo/citing](docs/zenodo.md). MIT license.*
