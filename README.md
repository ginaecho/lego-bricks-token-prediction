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

We investigate token and API-cost prediction by breaking work into reusable,
**LEGO-like task blocks**. The current customer-brief model learns small
numerical prediction formulas from measured GPT calls, not an autoencoder.
GPT itself is not retrained.

Start with the [short training explanation](docs/lego-model-training.md) or
the [HTML results report](docs/customer-model-report.html). This model covers
preparing a project scope, not completing a whole customer engagement.
The earlier experiments and proposed extensions below have different scopes.

Marketplace direction: see the [requirements and TODO roadmap](docs/marketplace-models/plan.md)
and the [service selection and quoting guide](docs/marketplace-models/quoting.md).
Shared input/output models, versioned scoping contracts, and offline pricing are
implemented; calibrated ranges and production recommendations remain pending.

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
   strongly sub-additive: batching work into one agent **saves 64–71%**, the
   single biggest cost lever a buyer has.
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
| Cross-validated **2.55%**; unseen requests **0–4.7%** | a range you can sign off on at scoping, not a vibe |

Full experiment and limitations: **[docs/composition-findings.md](docs/composition-findings.md)**

## Try it

```bash
pip install -e .                       # Python ≥ 3.9, no runtime dependencies
python -m examples.composition_demo    # the whole loop, from the committed data
```

---

*Token Yield (`token_yield/`) stands on **HarnessDose**, the measurement layer
(package `openharness`), and composes with Microsoft's
[Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit).
Docs: [calibration](docs/calibration-findings.md) ·
[architecture](docs/architecture.md) · [precedence](docs/precedence.md) ·
[AGT](docs/agt-integration.md) · [Zenodo/citing](docs/zenodo.md). MIT license.*
