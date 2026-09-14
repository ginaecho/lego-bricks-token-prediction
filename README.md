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

## Marking our own homework — the assay

A headline like "cross-validated 2.55%" deserves an adversary, because of the
line two rows above it: if start-up is 89% of a median task, then a model that
ignores your request entirely and always predicts the start-up cost will also
post a small "mean error". Low total error is not, by itself, evidence that
anything was learned about the work.

`assay/` is that adversary — a standalone falsification harness that tries to
break the claim rather than support it. It is deliberately hostile to its own
conclusions:

- **Two error numbers, always together.** Total error, and *variable-portion*
  error — the error on the part that actually responds to your request. The
  scorecard type will not let one be reported without the other.
- **A ladder, not a model.** Every form is scored from the dumbest upward, and a
  per-brick form is promoted only if it beats the best *cheaper* baseline on the
  variable portion, under grouped cross-validation, by a pre-registered margin.
- **Pre-registered, hashed gates.** The bars were written down and SHA-256'd
  before anything was measured. A gate with no observation does not quietly
  pass; it caps the verdict. Moving a bar changes the hash, and a test enforces
  that.
- **A null control.** The same pipeline is run against a runtime with no
  per-brick structure at all. A method that finds signal in noise is worthless,
  so the harness must be shown *refusing*.

Replay this repository's own evidence through it:

```bash
python -m assay audit
```

Limits of what that replay can support: **[docs/assay-evidence-limits.md](docs/assay-evidence-limits.md)**

## The studio — watch it happen

```bash
make studio          # or: python -m assay studio
```

A dependency-free local app (stdlib `http.server`, no build step, no network,
no credentials) that runs the entire method twice — once against a simulated
runtime that *does* price bricks, once against one that does not — and shows
both, live, as it goes.

| page | what it shows |
|---|---|
| **Baseplate** | the sealed corpus, and both arms' builds progressing stage by stage |
| **Brick box** | the vocabulary, each brick with its nearest neighbour |
| **The toll** | the start-up toll, and the same forms scored two ways — the argument in one table |
| **The ladder** | every rung, scored, with the promotion test that was applied |
| **Quote desk** | order bricks and price them; **change *which* bricks you ask for without changing the byte count, and watch the null arm not move** |
| **Invoice** | where the tokens went, with a reconciliation line for what the model could not explain |
| **Verdict** | the pre-registered gates, scored, with the policy hash |

The demo is honest about its own weakness, and says so on screen: every number
in it comes from a simulator, so it carries `evidence_class: pipeline_only`
throughout. It is **not** evidence that bricks predict a real model's cost. It
is evidence that the machinery *discriminates* — it finds a planted signal and
refuses an absent one. Both arms finish at **narrow**, never `feasible`, because
four gates (encoder accuracy, end-to-end error, batching quality, inter-rater
agreement) cannot be observed without a real model and human labellers.

## Try it

```bash
pip install -e .                       # Python ≥ 3.9, no runtime dependencies
python -m examples.composition_demo    # the whole loop, from the committed data

# the assay and its studio need Python ≥ 3.11
python -m demo.generate_corpus         # regenerate the demo corpus
python -m assay studio                 # the visual demo
python -m pytest -q                    # 553 tests
```

---

*Token Yield (`token_yield/`) stands on **HarnessDose**, the measurement layer
(package `openharness`), and composes with Microsoft's
[Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit).
`assay/` is the independent falsification harness that audits it, and
`assay/studio/` is its visual demo.
Docs: [calibration](docs/calibration-findings.md) ·
[architecture](docs/architecture.md) · [precedence](docs/precedence.md) ·
[AGT](docs/agt-integration.md) · [Zenodo/citing](docs/zenodo.md). MIT license.*
