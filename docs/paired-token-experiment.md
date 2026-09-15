# Paired token experiment: prediction versus the same work executed

The first base-brick catalog established that tasks could be dispatched and
graded. It did not provide token measurements. This experiment completes the
loop the project requires:

```text
atomic measurements + composition measurements
                    ↓
             engineered features
                    ↓
             fit on training only
                    ↓
       freeze predictions for unseen cases
                    ↓
          execute those exact same cases
                    ↓
        compare actuals and feed failures back
```

The complete raw wave is committed at
[`runs/20260826_1416_wave1`](../runs/20260826_1416_wave1/).

## Feature engineering and training

The matrix has one row per isolated agent session and these pre-run features:

- context bytes;
- one count for each of the nine primitives;
- total units; and
- composition arity.

The target is provider-reported `inputTokens + outputTokens`. It is a runtime
usage target, **not a billing target**: Copilot also reported cached context and
all 29 sessions consumed the same 0.33 premium-request units despite different
token totals. Training contains two null probes, two shapes for every
primitive, and five compositions. Failed outputs remain in the data because
they still incur spend.

Six pre-run model forms were compared by leave-one-out cross-validation. The
winner was the **constant**, not the LEGO form:

| form | training LOO MAPE |
|---|---:|
| constant | **6.11%** |
| units | 6.38% |
| bytes | 8.31% |
| bytes + units | 8.55% |
| per primitive | 9.09% |
| bytes + per primitive | 9.32% |

For this runtime and these short probes, a fresh Copilot session reported about
20k tokens before any task-shape signal could be distinguished. The design had
only 25 training rows, 22 of them at 0–466 context bytes, no replicated task
shapes, and one influential multi-call row. The selected constant is therefore
an **inconclusive, underpowered wave-1 result**, not evidence that brick features
cannot work.

## Paired held-out result

The model and predictions were serialized before held-out dispatch. Then the
same four cases were run through fresh agents:

| case | decomposition | predicted | actual | error | accepted |
|---|---|---:|---:|---:|---|
| close report | Reconcile + Remediate + Report | 21,029 | 20,449 | 2.84% | yes |
| Apple performance | 2xExtract + Reconcile + Report | 21,029 | 20,291 | 3.64% | yes |
| Apple risk memo | Review + Draft | 21,029 | 61,927 | 66.04% | yes |
| Apple margin check | 3xExtract + 2xValidate | 21,029 | 20,250 | 3.85% | yes |

Overall held-out MAPE is **19.09%**. Excluding nothing, that is the honest
scoping result. The three ordinary one-call cases average 3.44%; the one
branched case is an expensive outlier.

The forced `bytes + per-primitive` model scored **30.86%** held-out MAPE, worse
than the selected constant. It is reported as a post-hoc hypothesis diagnostic,
not substituted for the model that won before evaluation.

Row-wise leave-one-out was valid for this unreplicated wave, but it must not be
used after shape replicates are added: duplicated prompts would leak across
folds. Wave 2 consequently groups validation by task shape and source.

## What caused the outlier

The raw provider records show:

- 24 one-call training runs averaged 20,205 tokens;
- the only two-call training run cost 40,794;
- the Apple risk memo expanded to three model calls and cost 61,927.

The prompt supplied a risk summary, but the agent announced an external
fetch/verification step before producing the review and memo. The planned
decomposition (`Review + Draft`) therefore missed an execution branch.

This is the first real feedback item for the encoder:

```text
planned:  Review + Draft
observed: external retrieval/verification + Review + Draft
```

Simply changing the count to the existing `Retrieve` brick would not solve the
problem: that brick was calibrated on one-call retrieval over an embedded
corpus. Wave 2 must separately measure **external retrieval / continuation
risk** and add a pre-run feature such as `external_source_reference`.

## Budget implication

The training p95 was 20,639 tokens and p99 was 35,959. The held-out outlier was
61,927. An empirical p95 reserve from this small sample would therefore
underfund the task by more than 41k tokens.

For chargeback, actual spend is still straightforward: all 61,927 tokens belong
to the accepted risk memo. For scoping, however, the quote needs two components:

1. a central one-call estimate near 20-21k; and
2. an escalation reserve based on the probability and cost of additional model
   calls.

This connects the LEGO decomposition to unit economics without claiming the
current bricks explain a tail they demonstrably missed. It also motivates a
different direct-API runtime for wave 2, where uncached input, cached input,
output, reasoning, model calls, and tool calls are retained separately.

## Reproduce

The adapter in
[`token_yield/copilot_dispatch.py`](../token_yield/copilot_dispatch.py) launches
one fresh non-interactive Copilot session and reads `--usage-output-file`.
[`examples/paired_experiment.py`](../examples/paired_experiment.py) enforces the
train-before-predict-before-execute order.

```bash
# spends AI credits; use a new run directory for an independent repeat
python -m examples.paired_experiment --run-dir runs\<timestamp>_wave2 --phase all
```

The test suite never launches paid sessions.
