# Paired token experiment — wave 1

This folder is the durable evidence for the first complete
**train → predict → execute → compare → feedback** wave.

## Protocol

- Runtime: GitHub Copilot CLI `1.0.81-9`
- Model: `claude-haiku-4.5`
- Isolation: one fresh non-interactive CLI session per case
- Usage source: Copilot `--usage-output-file`
- Usage target: provider-reported `inputTokens + outputTokens`
- Billing boundary: the usage target re-counts cached harness context and is
  not a provider charge; every session reported 0.33 premium-request units
- Training: 25 measured sessions
  - 2 null probes
  - 18 atomic cases (two shapes for each of nine primitives)
  - 5 fitted compositions
- Held out: 4 sessions
  - 1 created three-brick composition
  - 3 source-backed Apple 2021 Form 10-K tasks

`predictions.jsonl` was written at 12:43:51Z. The first held-out execution was
not dispatched until 12:44:30Z, so every selected-model prediction was frozen
before the actual usage existed.

## Measured totals

| quantity | result |
|---|---:|
| isolated sessions | 29 |
| training tokens | 525,718 |
| held-out tokens | 122,917 |
| total measured tokens | 648,635 |
| reported premium-request cost | 9.57 |
| accepted training outputs | 17 / 25 |
| accepted held-out outputs | 4 / 4 |

Rejected training outputs remain in model fitting because failed work still
consumes budget.

## Selected model

Cross-validation selected the constant form:

```text
tokens = 21,029
```

| metric | result |
|---|---:|
| training LOO MAPE | 6.11% |
| held-out MAPE | 19.09% |

| held-out case | frozen prediction | actual | error | model calls |
|---|---:|---:|---:|---:|
| close report | 21,029 | 20,449 | 2.84% | 1 |
| Apple performance | 21,029 | 20,291 | 3.64% | 1 |
| Apple risk memo | 21,029 | 61,927 | 66.04% | 3 |
| Apple margin check | 21,029 | 20,250 | 3.85% | 1 |

For the three one-call held-out tasks, mean absolute percentage error is 3.44%.
The risk memo is the expensive tail: its extra model calls re-paid roughly the
20k-token agent startup.

## LEGO hypothesis diagnostic

The richer `bytes + per-primitive` form was fitted from the same training
partition but was not allowed to replace the cross-validated winner after
held-out results were known.

| metric | constant winner | forced LEGO form |
|---|---:|---:|
| training LOO MAPE | 6.11% | 9.32% |
| held-out MAPE | 19.09% | 30.86% |

Wave 1 is **inconclusive and underpowered** for the brick-count hypothesis. The
selected constant is the correct result for this partition, but the experiment
cannot distinguish "no brick signal" from "signal hidden by the runtime
harness and compressed design": 22 of 25 training rows used only 0–466 context
bytes, task shapes were not replicated, and one influential two-call row drove
model ranking and coefficients. The durable finding is narrower: execution
branching was missing from the taxonomy and the Copilot usage target is not a
defensible billing target.

## Tail evidence

Training attempt distribution:

| percentile | tokens |
|---|---:|
| p50 | 20,170 |
| p90 | 20,567 |
| p95 | 20,639 |
| p99 | 35,959 |
| max | 40,794 |

The 61,927-token held-out outlier exceeded even the training maximum. A central
prediction plus an empirical p95 reserve would not have protected the budget.
The raw usage proves the branching mechanism: 24 one-call training runs
averaged 20,205 tokens; the only two-call training run cost 40,794; the
three-call held-out run cost 61,927.

## Files

- `train_records.jsonl` — measured atomic and composite training attempts
- `feature_matrix.csv` — model inputs and measured target
- `model.json` — selected form, coefficients, scores, and training IDs
- `predictions.jsonl` — predictions frozen before held-out dispatch
- `held_out_records.jsonl` — actual outputs, acceptance, and token usage
- `evaluation.json` — selected-model comparison
- `diagnostics.json` — forced LEGO form and tail/request-count diagnostics
- `feedback.jsonl` — observed decomposition failure and next hypothesis
- `raw/` — exact agent outputs and provider usage JSON for every session

## Reproduce

This command spends AI credits:

```bash
python -m examples.paired_experiment \
  --run-dir runs\20260826_1416_wave1 \
  --phase all
```

The runner is resumable by case ID. Delete or choose a new run directory for a
clean independent wave.
