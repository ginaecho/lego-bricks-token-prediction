# Selecting and quoting marketplace services

The offline API in [marketplace.py](../../token_yield/marketplace.py) implements
T006-T008 of the [TODO plan](plan.md). It does not execute customer work or
promise a calibrated price.

## Service contracts

`service_catalog(template, artifact)` returns the four services bound to the
model's template hash. Each contains a version, unit, input/output schema,
instruction, dependency list, acceptance-check reference, and observed feature
ranges. A template change creates a different service version and requires a
matching model contract.

Extract copies supplied requirement evidence. Classify preserves supplied
annotations; it is not classification of previously unlabelled text. Plan
proposes scoping artifacts with review required. Report describes the requested
scope without claiming delivery.

The input remains the existing `customer-requests-v1` project object. Its
`split` field is inherited research metadata, not a learned predictor or a
customer segmentation feature. Schema descriptions cover structure; the
existing validators enforce additional grounding, HTTPS, ordering, whitespace,
and trimmed text-length constraints.

## Explicit selection

Each selection must supply `service_id`, `version`, and integer `quantity`.
For Extract, Classify, and Plan, quantity must equal the number of supplied
requirements. Report requires quantity one. There is no silent truncation,
repetition, service substitution, or natural-language decomposition.

`quote_selection` supports two explicitly chosen execution modes:

* `separate`: fresh independent calls over the original brief. The total sums
  these individual call estimates; no generated output is passed downstream.
* `batched`: one direct prediction of the selected batch, with no fixed
  batching discount.

Unobserved batches, out-of-range features, and runtime mismatches receive
`status: unsupported`, reasons, and no marketplace token total or API price.
Sequential handoffs are not supported by this quote API. Marginal training
ranges do not establish joint or domain support, so even accepted inputs return
research estimates with null calibrated intervals and production recommendations.

`check_selection_output` reuses the existing grounded acceptance checks.
A contract pass still requires human review; it is not quality certification.

## Pricing

`rate_quote` takes a frozen selection quote and an explicit versioned USD rate
card. The card includes `id`, `version`, `currency`, `runtime_sha256`, `as_of`,
`rates`, and `source`. A runtime mismatch produces no price. Quote hashes detect
accidental mutation; they are content identifiers, not authenticity signatures.

The rate calculation reuses the existing `Pricing.attempt_cost`. All estimated
input is rated as uncached because there is no measured cache predictor. The
cached rate remains in the rate card, but does not imply a cache discount.
Output tokens already include reasoning where applicable; there is no second
reasoning-token charge.

Tool cost, human-review cost, and gross-margin fraction are optional explicit
commercial assumptions. Missing values remain null, not zero. API cost can be
reported without those assumptions; proposed selling price cannot.

```text
cost basis = estimated API cost + declared tool cost + declared review cost
selling price = cost basis / (1 - gross-margin fraction)
```

Gross margin must be between zero inclusive and one exclusive. It is not a
markup percentage. Fees are caller-declared, not learned or measured. Prices
exclude any undeclared tax, discounts, infrastructure, or invoice reconciliation.

## Run the offline example

The [example command](../../examples/marketplace_quote.py) loads the frozen
source protocol and trained artifact, verifies protocol provenance, and saves
a catalog, explicit selections, token quote, rated quote, and summary.

```powershell
.\.venv\Scripts\python.exe -m examples.marketplace_quote `
  --project-id paradigm-website-redesign `
  --services extract report `
  --execution-mode separate `
  --recorded-runtime `
  --historical-rates `
  --run-dir runs\marketplace_quote_example
```

The two acknowledgment flags are required: this example assumes the saved
execution runtime and uses the recorded September 14 rates. It neither verifies
the current deployment nor fetches current prices. It spends no API credits.
Choose a new output directory for each run; existing artifacts are never
overwritten.

To illustrate a selling-price scenario, explicitly supply
`--tool-cost-usd`, `--human-review-cost-usd`, and `--gross-margin-fraction`.
Those numbers remain assumptions, not evidence that tools or human review ran.
