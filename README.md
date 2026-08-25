# Token Yield — Lego bricks for token prediction

### Budget the tokens before you spend them

[![CI](https://github.com/ginaecho/lego-bricks-token-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/ginaecho/lego-bricks-token-prediction/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/1314056228.svg)](https://zenodo.org/badge/latestdoi/1314056228)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

We predict token consumption for AI projects by decomposing complex workflows
into reusable, **LEGO-like task blocks**. Each atomic task is a building block
and becomes an input feature for model training. A new project is first broken
into those components, then encoded and reconstructed — autoencoder-style — to
capture its complexity and structure, and the resulting representation predicts
overall token usage.

![Four panels: the nine measured base blocks and what each unit adds; the context ablation showing 0.37 tokens per byte above a fixed agent start-up cost; stacking blocks into one agent saving 64-71% against running them apart; and three plain-English requests decomposed, priced in advance, then checked against what they actually cost](docs/media/token-yield-composition.svg)

## The four steps

**1 · Pre-simulation — build the basic bricks.** Run each atomic task for real
and record what it costs. Nine blocks, each a task type enterprises already buy:
**Review, Extract, Classify, Retrieve, Reconcile, Draft, Remediate, Validate,
Report**. Each is measured at more than one size, so the *shape* of its cost is
observed, not assumed.

**2 · Pre-simulate the combinations.** Stack the bricks — `A+A`, `A+B`,
`A+B+C` — and measure those too. This is the step a naive model skips, and it
is where the money is: combinations are strongly **sub-additive**.

**3 · Train the predictor on (1) + (2).** Block counts and context size are the
input features; measured tokens are the target. Six nested model forms are
raced by leave-one-out cross-validation, so the winner has to predict points it
never saw.

**4 · Feed in a real case, and close the loop.** A plain-English request is
decomposed back into the blocks of step 1, recomposed through the fitted model,
and priced *before* it runs. Once it does run, the measured cost is compared
against the prediction — that reconstruction error is what refits the model.

```
request (free text) --encode--> block counts --decode--> tokens --measure--> refit
```

## What the measurements say

**39 real agent runs**, each a fresh subagent whose token usage was recorded,
over **33 genuine SEC filings and earnings-call transcripts** spanning pharma,
retail, semiconductors, consumer goods, industrials and fintech.

| finding | number |
|---|---|
| Starting an agent, before any work | **29,821 tokens** — 89% of the median task |
| Cost of context | **0.37 tokens/byte**, flat — 33 KB more reading adds 47%, not 400% |
| Stacking blocks in one agent vs. one agent each | **saves 64–71%** |
| Most expensive block | **Retrieve, 5,384/unit** — an order of magnitude above the rest, because its work is *searching* |
| Model accuracy (cross-validated) | **2.55%**, against a **0.29%** repeat-run noise floor |
| Held-out compositions, never fitted | **2.2% mean error** (incl. a four-way mix the fit never saw) |
| Plain-English requests, priced before running | **0–3.5%** |

Full experiment, method and limitations:
**[docs/composition-findings.md](docs/composition-findings.md)**.

## Quick start

```bash
pip install -e .                          # no runtime dependencies; Python ≥ 3.9
python -m examples.composition_demo       # the whole four-step chain, from committed data
```

```python
from token_yield import load_runs, default_runs_path, select_composition_model
from token_yield import parse_decomposition, explain

model = select_composition_model(load_runs(default_runs_path()))
blocks = parse_decomposition('{"counts": {"review": 1, "extract": 3, "validate": 2}}',
                             context_bytes=1_235)
print(explain(blocks, model))     # itemised: start-up, context, then each block
```

`explain()` prints the forecast as a line per block — the same object that
prices a request in advance becomes its invoice afterwards, which is what
budgeting, forecasting and chargeback need.

## What this doesn't claim

- **One model, one size.** Every run used `claude-haiku-4-5`. The *shape* of the
  finding should transfer; the constant is model-specific.
- **Small campaign.** 39 runs, 35 fitted — enough to choose among six forms, not
  to call the marginals precise to the token.
- **The decomposer isn't separately evaluated.** Three plain-English cases is a
  demonstration; a wrong decomposition yields a confidently wrong price, and
  nothing here bounds how often that happens.
- **Documents, not workflows.** Real processes involve systems, approvals and
  humans waiting. None of that is measured.

## What else is here

Token Yield stands on **HarnessDose** — the measurement layer that makes
per-task token counts observable in the first place, and puts every behavioural
rule on an inspectable card. It composes with Microsoft's
[Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit):
AGT enforces, HarnessDose characterizes and proves.

- [The harness layer, and why it matters](docs/architecture.md) · [proving it works](docs/proving-it-works.md) · [how it was tested](docs/how-it-was-tested.md)
- [Calibration findings](docs/calibration-findings.md) — where the cost data comes from
- [Precedence / conflict layer](docs/precedence.md) · [AGT integration](docs/agt-integration.md) · [evaluation methodology](docs/evaluation-methodology.md)
- [Animated explainer](docs/media/token-yield-explainer.html) (standalone page — download and open)

```
token_yield/     tasks · trainsuite · compose · decompose  (the four steps)
                 taxonomy · costmodel · probes · learn · backtest · plan · mine · duration
experiments/     train_runs.jsonl (39 measured runs) · decompose_cases.jsonl
openharness/     the measurement layer  ·  modules/ skills/ the rules it measures
benchmark/ precedence/ integrations/    L1–L5 proof harness
examples/ tests/ docs/                  demos · 225 tests · findings and method
```

> **Names.** The repository is `lego-bricks-token-prediction`; **Token Yield**
> (`token_yield/`) is the prediction layer; **HarnessDose** is the measurement
> layer beneath it; the installable package for both is `openharness`.

## Citing & License

Citable via [Zenodo](https://zenodo.org) on every release — see
[docs/zenodo.md](docs/zenodo.md) and [CITATION.cff](CITATION.cff).
MIT — see [LICENSE](LICENSE).
