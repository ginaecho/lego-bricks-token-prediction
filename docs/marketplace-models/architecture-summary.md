# Marketplace model architecture

## Existing evidence

The current call-level learner predicts recorded USD cost. Its private target
fitter already handles token targets for the whole-project learner. Reuse that
fitter for public call-level input/output token channels rather than creating
four independent estimators or a second regression implementation.

Source contracts and modules are listed in
[project-structure.md](project-structure.md) and [tech-stack.md](tech-stack.md).

## First milestone

```text
Frozen protocol + recorded calls + raw request/response files
  -> verify measurements against the protocol and ledger hashes
  -> preserve train/holdout project split
  -> fit input channel and output channel on training calls
  -> select each form using training-only grouped validation
  -> save artifact with runtime and provenance
  -> reload and evaluate historical holdout calls
```

The first candidate set remains constant, size, size+units, and LEGO.
LEGO uses shared coefficients for prompt size, output allowance, and four
operation counts. These are predictive associations, not identified causal
prices per operation. Simpler forms may win; never force LEGO to win.

## Data and API contracts

Reuse the measured call row: call/project ID, split, status, quote features,
usage channels, recorded cost, and quality metadata. No actual response length,
quality outcome, or realized retry count enters quote features.

Add three public operations to
[customer_models.py](../../token_yield/customer_models.py):

* `fit_token_models(rows, runtime)` returns a versioned two-channel artifact.
* `forecast_tokens(artifact, quote, runtime)` returns nonnegative input/output
  estimates and their sum, support reasons, and no calibrated interval.
* `evaluate_token_models(artifact, rows)` scores only disjoint historical
  holdout projects without selecting or refitting.

An exact runtime-contract mismatch returns no estimate and a reason. Counts
must describe a nonempty selection. Marginal ranges and observed operation
combinations flag extrapolation; they do not establish joint/domain support.
The initial artifact describes the original template, not arbitrary new
marketplace functions.

Expose offline training through a separate example command. Reuse existing
protocol/response verification and output persistence helpers in the pilot
module. Keep current execution paths and historical artifacts unchanged.

## Later marketplace path

```text
Explicit service selection
  -> versioned registry and contract checks
  -> quote-time feature extraction
  -> supported shared/task-family predictor
  -> measured composition adjustment
  -> versioned API rate card
  -> separately disclosed tools/review/margin
```

A natural-language decomposer is optional upstream of selection. It is not
required when the buyer already chooses the services.

## Evidence gaps and decisions

Only standalone operations and the all-four independent batch were measured.
Intermediate combinations, sequential handoffs, tools, retries, provider
changes, arbitrary quantities, and new domains are not established by these
data. Runtime fields are compatibility gates, not learned effects.

Defer registry implementation and broader measurements until the offline
baseline is saved. Defer calibrated intervals and specialist promotion until
independent data and preregistered acceptance thresholds exist. API prices are
not features of token consumption and will remain outside the token model.
