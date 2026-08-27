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

## From a token budget to a scoping decision

A token budget is one of five numbers a project manager needs. **Project
Yield** (`project_yield/`) keeps the encoder and the brick vocabulary exactly as
they are and points the same decode step at the other four — what the client
will pay, whether it will land, who it takes, and how long it runs.

```
 description ──encode──▶  9 brick counts             ──decode──▶  tokens
                        + context bytes                        ├▶ contract value
                        + industry, goal                       ├▶ success rate
                        + lineage: depth, siblings,            ├▶ architect · engineer
                          inherited brick share                │  · pm days
                                                               └▶ time to finish
```

Six heads, not one model with six outputs: money multiplies, a win is bounded at
both ends, and elapsed time has a floor no headcount moves. Each head declares
its own link function and **independently selects its functional form** from
seven candidates by leave-one-out cross-validation — the same rule the token
model uses, six times over, and on the shipped data they do not all choose the
same one.

**Lineage is a feature, not a footnote.** Most enterprise use cases continue one
already delivered, and pricing those as greenfield is how both the estimate and
the margin go wrong. A use case can declare a parent and siblings; reuse depth,
sibling count and *inherited brick share* go into the vector, and every head is
free to price them at zero.

**The token head is measured. The value and impact heads are not** — they are
fitted on a synthetic corpus (`experiments/make_engagements.py`) so the
machinery could be built and reviewed before touching real delivery data. Every
card, screen and JSON payload says so, alongside warnings for extrapolated
scope, an unusual brick mix, a keyword-encoded description, or a head that fails
to beat its own baseline.

Runs locally with no cloud; `project_yield/azure.py` is the seam where Azure AI
Foundry (the encoder), Microsoft Fabric (the corpus) and Azure ML (scheduled
retraining, later) plug in.
**[docs/product-prototype.md](docs/product-prototype.md)** has the architecture,
the Fabric view, and what would have to be true before it quotes a real client.

## Try it

```bash
pip install -e .                       # Python ≥ 3.9, no runtime dependencies
python -m examples.composition_demo    # the token model, from the committed data
python -m examples.project_yield_demo  # tokens + value + impact, end to end
python -m project_yield serve --open   # the scoping prototype, in a browser
python -m project_yield batch examples/usecases   # 20 written scoping notes, ranked
```

Feed it the paragraph somebody already wrote — paste it, drop the file on the
page, or point it at a folder. Twenty worked examples are in
[`examples/usecases/`](examples/usecases/), spanning a per-item pipeline at
24,000 runs a month, two continuation chains, one deliberately vague note and
one deliberately enormous programme.

---

*Token Yield (`token_yield/`) stands on **HarnessDose**, the measurement layer
(package `openharness`), and composes with Microsoft's
[Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit).
Docs: [product prototype](docs/product-prototype.md) ·
[calibration](docs/calibration-findings.md) ·
[architecture](docs/architecture.md) · [precedence](docs/precedence.md) ·
[AGT](docs/agt-integration.md) · [Zenodo/citing](docs/zenodo.md). MIT license.*
