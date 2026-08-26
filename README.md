# Token Yield — Lego bricks for token prediction

[![CI](https://github.com/ginaecho/lego-bricks-token-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/ginaecho/lego-bricks-token-prediction/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/1314056228.svg)](https://zenodo.org/badge/latestdoi/1314056228)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

[![Token Yield — the animated explainer, looping](docs/media/Token_Yield_Explainer.gif)](docs/media/Token_Yield_Explainer.mp4)

▶ **[Full video](docs/media/Token_Yield_Explainer.mp4)**

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
