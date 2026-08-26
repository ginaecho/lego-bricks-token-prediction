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

1. **Pre-simulate the bricks.** Run each atomic task for real, at several
   sizes, and record its token cost.
2. **Pre-simulate combinations.** Stack bricks (`A+A`, `A+B`, `A+B+C`) and
   measure again — combinations are strongly sub-additive.
3. **Train.** Brick counts + context size are the features, measured tokens the
   target; the model form is chosen by leave-one-out cross-validation.
4. **Predict and close the loop.** A real request is decomposed back into
   bricks, recomposed into a price *before it runs*; the measured actual feeds
   back and refits.

## Measured, not asserted

39 real agent runs over 33 genuine SEC filings, nine bricks
(Review · Extract · Classify · Retrieve · Reconcile · Draft · Remediate · Validate · Report):

| | |
|---|---|
| Agent start-up, before any work | **29,821 tokens** — 89% of the median task |
| Context | **0.37 tokens/byte**, flat |
| Stacking bricks in one agent | **saves 64–71%** vs one agent per brick |
| Model accuracy (LOO-CV) | **2.55%** against a 0.29% noise floor |
| Held-out and plain-English requests, priced in advance | **0–4.7% error** |

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
