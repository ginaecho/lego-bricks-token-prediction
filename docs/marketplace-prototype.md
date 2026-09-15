# Functionality marketplace prototype

Token Yield can give clients a marketplace for composing a product from
LEGO-like functionality bricks. Instead of choosing technical tasks first,
clients shop for customer-facing capabilities and see their underlying task
composition, estimated token usage, AI budget, and potential business benefits.

The [interactive HTML prototype](../marketplace-prototype.html) explores this
idea on the `feat/marketplace` branch. It is a standalone design experiment,
not a production storefront or a purchasing system.

> [!IMPORTANT]
> All token coefficients and pricing defaults are illustrative. The prototype
> does not call the repository's trained predictor or a model provider.
> Business benefits are hypotheses, not measured improvements.

## Try the prototype

Open [marketplace-prototype.html](../marketplace-prototype.html) in a modern
browser. No server, package installation, API key, or network connection is
required. From the repository root on Windows:

```powershell
Start-Process .\marketplace-prototype.html
```

Use the floating design explorer to compare three layouts:

| Variant | Layout | Question to explore |
|---------|--------|---------------------|
| `?variant=A` | Marketplace | Can clients select capabilities through familiar shopping cards? |
| `?variant=B` | Compare and configure | Does a compact catalog alongside configuration controls support comparison? |
| `?variant=C` | Composition canvas | Does a visual assembly help explain how capabilities connect? |

The left and right arrow keys also switch layouts, except while editing form
fields or using the review dialog. The URL preserves the layout on reload.
Selections and assumptions live in memory and reset when the page reloads.
No winning layout has been selected yet.

## Client experience

1. Search or filter the catalog by customer experience, operations, or intelligence.
2. Add functionality bricks to the build, or remove them from the catalog or cart.
3. Adjust monthly runs, integration complexity, budget, and planning reserve.
4. Inspect recurring functionality tokens, integration routing tokens, and
   one-time integration AI cost separately.
5. Explore potential scalability, resource-use, and engagement benefits.
6. Review the build and download a JSON estimate containing selected features,
   atomic coefficients, assumptions, token totals, costs, and limitations.

The download is local. Nothing is ordered, submitted to a server, or provisioned.

## From product capabilities to atomic tasks

The interface follows the repository's
[decompose-and-recompose approach](../README.md#how-it-works).
Each purchasable-looking capability is a composition of smaller task types.
These tasks provide a vocabulary for future measurement and prediction, rather
than treating each product feature as an unrelated fixed price.

| Functionality | Atomic tasks | Demo tokens per run |
|---------------|--------------|--------------------:|
| AI customer support | Retrieve, Draft, Validate | 3,960 |
| Smart product search | Retrieve, Classify | 2,490 |
| Personalized discovery | Classify, Score, Draft | 2,570 |
| Customer insights | Classify, Summarise, Report | 3,270 |
| Document automation | Extract, Transform, Validate | 2,970 |
| Guided onboarding | Plan, Draft, Notify | 2,990 |

Shared task types make reuse opportunities visible. The prototype still charges
each task occurrence independently: it does not assume that shared task names
automatically eliminate work or save tokens.

## Estimation rules

### Monthly functionality usage

One run executes a selected functionality once. The same monthly run count
applies to every selected functionality; it is not a unique-customer count or
a single end-to-end product transaction.

```text
feature input tokens = runs x sum(selected feature input tokens per run)
feature output tokens = runs x sum(selected feature output tokens per run)
```

### Integration usage

For `n` selected functionalities, the prototype assumes every pair requires a
connection:

```text
connections = n x (n - 1) / 2
routing input tokens = round(120 x connections x runs x complexity)
routing output tokens = round(30 x connections x runs x complexity)
setup input tokens = round(18,000 x connections x complexity)
setup output tokens = round(6,000 x connections x complexity)
```

Complexity is `1` for standard shared APIs, `1.6` for multiple custom systems,
or `2.4` for legacy integrations. This multiplier affects integration only,
not base functionality tokens.

Routing is recurring AI overhead. Setup represents hypothetical one-time
AI-assisted Plan, Transform, and Validate work, not developer labor.
Zero monthly runs remove recurring costs but do not remove setup costs for
selected connections. Zero or one selected functionality has no connections.
The canvas is conceptual, not a dependency graph or execution order.

### Pricing and budget

Input and output rates are editable under the estimate details. Defaults are
USD 2 per million input tokens and USD 8 per million output tokens.
They are placeholders, not provider price quotes.

```text
cost = (input tokens x input rate + output tokens x output rate) / 1,000,000
cost with reserve = cost x (1 + reserve)
first-month AI subtotal = recurring cost with reserve + setup cost with reserve
```

The reserve can be 0%, 20%, or 40%. It adds a planning allowance to costs,
not tokens, and is not a statistical confidence interval. The budget indicator
compares the monthly budget with recurring AI cost including reserve;
it excludes setup.

### Default worked example

The initial build contains AI customer support and smart product search,
each running 10,000 times per month, with standard integration, a 20% reserve,
and a USD 500 monthly budget.

| Item | Estimate |
|------|---------:|
| Functionality tokens per month | 64,500,000 |
| Integration routing tokens per month | 1,500,000 |
| Total monthly input tokens | 55,200,000 |
| Total monthly output tokens | 10,800,000 |
| Total monthly tokens | 66,000,000 |
| Monthly AI cost before reserve | USD 196.80 |
| Monthly AI cost including reserve | USD 236.16 |
| Budget remaining | USD 263.84 |
| One-time integration tokens | 24,000 |
| One-time integration AI cost including reserve | USD 0.1008 |
| First-month AI subtotal, displayed to cents | USD 236.26 |

Displayed prices round to cents. The JSON export retains numeric precision.

## Potential benefits and evidence needed

| Benefit | What the prototype shows | What must be validated |
|---------|--------------------------|------------------------|
| Scalability | Modeled runs across selected capabilities | Throughput, latency, rate limits, and reliability under load |
| Sustainability | Distinct task types shared by multiple capabilities | Energy per successful task against a baseline; no carbon conversion or savings assumed |
| Customer engagement | Count of customer-facing capabilities | Engagement, conversion, satisfaction, and retention in controlled experiments |

These indicators describe the selected build, not a return-on-investment score.
More functionalities do not necessarily improve business outcomes. Fewer tokens
alone do not establish lower energy use or emissions.

## Limits and next steps

The estimate excludes engineering labor, hosting, licenses, tax, maintenance,
cache behavior, and reasoning-token modeling. It is not a total project quote.
There is no backend, authentication, checkout, saved project store, live
telemetry, or trained-model integration.

To move beyond the prototype:

1. Validate the shopping flow and select a layout with representative clients.
2. Replace the all-pairs connection assumption with explicit dependencies and
   per-feature usage volumes from a real product.
3. Map compositions to quote-time features supported by the repository's
   prediction pipeline and replace demo coefficients with measured evidence.
4. Track input, output, cache, reasoning, retries, and acceptance separately,
   preserving unknown values where measurements are unavailable.
5. Evaluate predictions on independent projects and expose calibrated ranges
   only when the evidence supports them.
6. Measure business benefits independently from token-cost prediction and
   present implementation and operating costs separately from AI usage.

## Verification

Browser checks covered adding and removing bricks, clearing the build, search,
category filters, exact default cost calculations, complexity changes, budget
warnings, zero usage, zero rates, invalid-number recovery, the review dialog,
JSON export arithmetic, and reload-stable layout selection. All three layouts
were checked for horizontal overflow at a 390-pixel viewport. Editor diagnostics
reported no errors in the HTML prototype.
