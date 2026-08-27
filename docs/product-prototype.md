# Project Yield — the product prototype

A scoping tool for Microsoft project managers and solution architects. You
describe a use case once — goals, scope, industry, and what it continues — and
it returns the five numbers a scoping decision actually needs:

| | question |
|---|---|
| **Annual impact** | what the client gets: handling displaced, less running cost |
| **Token budget and cost** | what one run costs to execute, and a year of it |
| **Contract value** | what a client like this has paid for work like this |
| **Success rate** | how often a use case shaped like this lands |
| **Working days** | kickoff to acceptance, in elapsed days |
| **Who you need** | days per role — architect, consultants, data engineers, data scientists, security experts, and whatever else is on your roster |

plus the arithmetic that turns them into a decision: delivery cost, gross
margin, expected margin, payback period, and the win rate below which the
engagement stops being worth staffing.

## Run it

```bash
pip install -e .
python -m project_yield serve --open      # the web prototype, on localhost:8765
python -m examples.project_yield_demo     # the same thing, on the terminal
```

No Azure subscription, no API key, no network. The point of a prototype is that
the person who has to judge it can run it.

```bash
python -m project_yield model                       # how each head was chosen
python -m project_yield predict --description "..." # one use case, as a card
python -m project_yield batch examples/usecases     # a whole folder, ranked
python -m project_yield batch examples/usecases --cards   # ... with every caveat
python -m project_yield batch examples/usecases --json    # ... as JSON
```

## Feeding it real descriptions

The natural input is not a form — it is the paragraph somebody already wrote in
a scoping note, an email or a statement of work. Three ways in:

* **Paste it** into the description box and press Estimate.
* **Drop the file** on the description box, or use *Attach a file*. Plain text
  or Markdown, up to 512 kB.
* **Point it at a folder** with `project_yield batch`, and get the whole
  portfolio ranked in one pass.

Twenty worked examples ship in [`examples/usecases/`](../examples/usecases/),
chosen to span the things that actually move a forecast — a per-item pipeline at
24,000 runs a month, a four-times-a-year review with a large per-run scope, two
continuation chains, one deliberately vague note and one deliberately enormous
programme. Their `README.md` says what each one exercises.

A folder needs no configuration. An optional `manifest.jsonl` beside the files
declares **lineage only** — which use case continues which — because that is
the one thing a description cannot carry: "follow-on to the pipeline we
delivered for Northwind" is obvious to a reader and not recoverable by an
encoder. Where a continuation never repeats the client's sector, which is the
normal case, the industry is taken from its parent and the substitution is
recorded rather than made silently.

## How it works

It is the repository's existing model, widened. `token_yield` decomposes a
request into a fixed vocabulary of nine **task bricks** — Review, Extract,
Classify, Retrieve, Reconcile, Draft, Remediate, Validate, Report — and
recomposes them through a fitted cost model to predict tokens. The encoder is
an agent; the decoder is the model; the round trip is checked against what the
task actually cost.

`project_yield` keeps that structure and adds a second half to both ends.

```
 description ──encode──▶  9 brick counts             ──decode──▶  tokens
                        + context bytes                        ├▶ contract value
                        + industry, goal                       ├▶ success rate
                        + lineage: depth, siblings,            ├▶ architect days
                          inherited brick share                ├▶ engineer days
                                                               ├▶ pm days
                                                               └▶ calendar days
```

**Many heads, not one model with many outputs.** Money multiplies and is
long-tailed, a win is bounded at both ends, and elapsed time has a floor no
headcount moves. Each head declares its own link function and its own scoring
metric, and each **independently selects its functional form** from seven
candidates by leave-one-out cross-validation — the same rule
`token_yield.compose` uses, applied nineteen times. On the shipped corpus they
do not all choose the same form, which is the point: the data decides, per head.

### The roster is yours

Staffing is not three lines in the code. `roles.json` is a file you edit, and
every role in it gets **two heads** — is this role needed at all, and how many
days if it is — plus its own day rate and its own line on every card. The
shipped roster is solution architect, data scientist, data engineer, software
engineer, security expert, industry consultant, project manager, change
manager. Add "MLOps engineer" and, given a column in the corpus, it is fitted,
priced and reported with no code change; without a column it is reported as
*unfitted* rather than silently absent, because a missing role looks exactly
like a role nobody needs.

Presence and days are separate on purpose. Averaging a data scientist's days
across the jobs that never used one gives "3.1 days", which is not a thing
anybody can book. So the plan reads *"48% likely, 7.4 days when needed"* and
the cost uses the product.

A project manager who **knows** the team overrides all of that: name the roles
on a use case and those become certain and the rest are excluded; enter days
for one and that number wins outright. Their knowledge beats a base rate, and
that is the whole reason the override exists.

### What the client gets

Everything above prices the *engagement*. `impact.py` prices the *outcome*: the
handling time the pipeline displaces, at the client's own loaded cost, less
what the inference costs to run. Task bricks say how much handling there is per
run; the production run rate says how often.

This is the number usually missing at scoping time, and its absence is why AI
business cases get argued on cost. A build costing $60,000 is expensive or
cheap depending entirely on whether it displaces $40,000 a year or $4 million,
and no delivery estimate can tell you which.

The per-brick handling times, the loaded hourly cost, and the **deflection
rate** are all placeholders and all printed on every card. Deflection is
deliberately below 1.0: no automation removes all the handling, and assuming it
does is the commonest way a benefit case is overstated.

Leave-one-out is exact and closed-form for the five continuous heads (the PRESS
identity: the deletion residual is `eᵢ / (1 - hᵢ)`), and the standard one-step
approximation for the logistic head. Cross-validating seven forms across six
heads therefore costs seven fits, not seven hundred — the whole model refits in
under half a second.

Every estimate is a band, not a point. The band is the empirical 10th-to-90th
percentile of that head's own leave-one-out residuals; no normality is assumed.

### Lineage: the part a token model cannot see

Most enterprise use cases are not greenfield. A client who bought invoice
extraction buys claims triage next; a proven compliance pipeline is re-pointed
at a second regulator. Pricing those as new projects is how both the estimate
and the margin go wrong.

So a use case can declare a **parent** (it continues that one) and **siblings**
(delivered alongside it), and three features fall out:

- `reuse_depth` — generations from a greenfield root, capped at 4
- `sibling_count` — how many run alongside, inferred from a shared parent as
  well as declared, so naming one sibling is not penalised versus naming all
- `inherited_fraction` — the share of this use case's bricks that already appear
  upstream. This is the feature that carries the signal: a "continuation" that
  is 90% new bricks is a new project wearing a continuation's badge, and this
  number says so.

None of the three is given a coefficient by hand. They are three features among
twenty-two and every head is free to price them at zero. On the shipped corpus,
the same work priced as a continuation rather than greenfield comes out at
roughly a third less engineering, a fifth less elapsed time, a markedly higher
success rate — and a *lower* price, because a client who knows the hard part is
done expects to pay less for the second phase.

## What is measured and what is not

**This matters more than anything else on this page.**

| | evidence |
|---|---|
| Token head | **Measured.** 39 real agent runs over 33 genuine SEC filings, `experiments/train_runs.jsonl`. Used unchanged from `token_yield`. |
| Value, staffing and duration heads | **Synthetic.** `experiments/engagements.jsonl`, generated by `experiments/make_engagements.py` from a documented latent process, seeded and committed. |
| Client impact | **Assumption, not a model.** Arithmetic over placeholder handling times, a placeholder hourly cost and a placeholder deflection rate. It is the most load-bearing and least evidenced number on the page, which is why it names its own assumptions inline. |

The synthetic corpus exists so the machinery can be built, tested and reviewed
before it touches real delivery data — and because the latent process is written
down, a bad fit is unambiguously the fitting code rather than a weak signal,
which is a check a corpus of real engagements cannot give you.

It is **not evidence about any real client**, and every surface says so: the web
app leads with the warning, the terminal card prints it above the numbers, and
`Forecast.warnings` carries it in the JSON.

The forecast also warns when it is extrapolating past the corpus in size or in
brick mix, when the keyword encoder was used instead of a model, when a named
parent is missing from the library, and when a head fails to beat its own
baseline on held-out data. A wrong number with a caveat is still usable; a wrong
number that looks like every other number is not.

## Where this runs on Azure

The recommendation, shortest first: **Azure AI Foundry for the encoder, Fabric
for the corpus, Container Apps for the app. Defer Azure ML.**

| piece | service | why |
|---|---|---|
| **Encoder** | **Azure AI Foundry** | The only part needing a model at inference time, and it is one chat completion with a fixed prompt — not an agent, not an orchestration. `project_yield.azure.FoundryEncoder`. |
| **Corpus** | **Microsoft Fabric** | Historical engagements come from delivery and CRM systems that already land in Fabric. This is a Lakehouse view, not a new pipeline. `project_yield.azure.FabricCorpus`. |
| **The app** | **Azure Container Apps** (or App Service) with **Entra ID** in front | `project_yield.app` is the container's contents. Pricing and win rates are commercially sensitive; authentication is not optional past the prototype. |
| **Comparables** | **Azure AI Search**, once the library outgrows a scan | The ranking function is `LineageIndex.nearest`; the vector is nine numbers. A few thousand use cases still scan faster than a network hop. |
| **Retraining** | **Azure ML**, later | See below. |
| **Portfolio view** | **Power BI** over the same Fabric tables | The per-use-case card is the prototype; the pipeline roll-up is a report, and Fabric already has one. |

### Why defer Azure ML

The honest answer is that the fit does not need it. Six heads over a few hundred
rows of a twenty-two-column feature vector refit in under half a second in pure
Python with no dependencies. A compute cluster, a registered model and a managed
online endpoint are all real costs — in money, in latency, and in the number of
people who have to be involved to change a coefficient — bought against a
training job that finishes faster than the HTTP request that would trigger it.

Adopt Azure ML when one of these becomes true, not before:

- the corpus is large enough that fitting is no longer instant;
- retraining has to be **scheduled and audited** — a registered model version per
  quarter, with the lineage from corpus snapshot to deployed coefficients;
- the scoring must be **a service** consumed by other systems (MSX, a Teams app)
  rather than a library inside one app.

The seam is already there for when that day comes: `project_yield.azure.init`
and `run` are an Azure ML scoring script, three lines each, because `YieldApp`
already separates what the thing does from how it is being served.

### Setting up the Foundry encoder

```bash
export AZURE_AI_FOUNDRY_ENDPOINT="https://<resource>.services.ai.azure.com"
export AZURE_AI_FOUNDRY_DEPLOYMENT="<your-deployment-name>"
export AZURE_AI_FOUNDRY_KEY="<key>"     # or use a managed identity, below
python -m project_yield serve
```

The app picks it up per request, so no restart is needed. With no configuration
it falls back to the keyword encoder and marks every estimate accordingly.

For a managed identity instead of a key — which is what a deployed app should
use — pass a token provider:

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from project_yield.azure import FoundryEncoder

encoder = FoundryEncoder(token_provider=get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"))
```

`temperature` is pinned to zero. This is an encoder, not a writer: the same
description must produce the same feature vector on Tuesday as it did on Monday,
or the estimate moves without the use case having changed.

### The Fabric contract

The entire integration is one view presenting these columns. Everything else in
the pipeline is unchanged.

```sql
CREATE VIEW dbo.vw_engagement_outcomes AS
SELECT
    e.engagement_id                       AS id,
    e.name                                AS title,
    c.account_name                        AS client,
    c.industry_segment                    AS industry,      -- must map to INDUSTRIES
    e.primary_business_goal               AS goal,          -- must map to GOALS
    e.brick_counts_json                   AS counts,        -- {"extract": 9, ...}
    e.context_bytes_per_run               AS context_bytes,
    f.signed_amount_usd                   AS contract_value,
    CASE WHEN e.outcome IN ('accepted', 'renewed')
         THEN 1 ELSE 0 END                AS won,
    t.architect_days, t.engineer_days, t.pm_days,
    DATEDIFF(day, e.kickoff_date, e.acceptance_date) AS calendar_days,
    e.continues_engagement_id             AS parent_id,
    e.sibling_ids_json                    AS sibling_ids,
    CONVERT(varchar(10), e.kickoff_date, 23) AS started
FROM   dbo.engagements e
JOIN   dbo.accounts   c ON c.account_id = e.account_id
JOIN   dbo.financials f ON f.engagement_id = e.engagement_id
JOIN   dbo.timesheet_rollup t ON t.engagement_id = e.engagement_id
WHERE  e.acceptance_date IS NOT NULL;      -- only finished work has outcomes
```

Two columns will not exist yet in most delivery systems and are the real
adoption cost of this tool:

- **`brick_counts_json`** — past engagements were never encoded into bricks.
  Backfill it by running the Foundry encoder over the statements of work you
  already have. That is a one-off batch job, and it is the step that turns a
  delivery archive into a training set.
- **`continues_engagement_id`** — lineage is usually in people's heads. Where it
  is absent, `LineageIndex.nearest` over the backfilled brick vectors proposes
  candidates for a human to confirm.

Start by exporting the view to JSONL nightly and pointing
`load_engagements(path)` at it. A prototype reading a nightly extract answers
the same question as one holding a live connection, and can be reviewed by
someone without database credentials. `FabricCorpus` reads the SQL endpoint
directly (via `pyodbc` and `ActiveDirectoryDefault`) when you want that.

## Before this quotes a real client

In order of how much each one matters:

1. **Replace the corpus.** Everything except the token budget is currently
   fitted on invented data. Nothing here is a pricing recommendation until the
   Fabric view is real.
2. **Replace the rates.** `project_yield.economics.Rates` ships placeholder day
   rates, marked `PLACEHOLDER` and printed on every card.
3. **Re-validate the form selection.** Real delivery data will have different
   structure, missing outcomes and survivorship in it — engagements that were
   never signed do not appear in a delivery system at all, which biases the
   success-rate head upward. Fitting on won-and-lost *opportunities* rather than
   on delivered engagements is the correct fix and needs CRM data, not delivery
   data.
4. **Calibrate the win head.** It is the weakest of the six on the shipped
   corpus and it is the one people will lean on hardest. A reliability
   diagram — predicted probability against observed frequency, bucketed —
   belongs on the model card before it is trusted.
5. **Get the handling times.** The impact figure moves proportionally with
   minutes-per-run and the deflection rate, and both are currently guesses.
   A morning of time-and-motion with the team doing the work today is worth
   more to this number than any amount of modelling.
6. **Measure `BUILD_RUNS`.** The number of times a pipeline is exercised during
   development is currently an assumption (250). Real telemetry from a handful
   of delivered projects replaces it.

## Layout

```
project_yield/
  outcomes.py     what is predicted; link functions and why each one
  usecase.py      the input: description, industry, goal, bricks, lineage
  corpus.py       Engagement — one delivered use case with realised outcomes
  lineage.py      parent/sibling graph -> reuse features; comparable lookup
  features.py     the feature vector and the seven candidate forms
  multihead.py    per-outcome fitting, form selection, intervals
  linalg.py       least squares, IRLS, leverage — dependency-free
  encode.py       description -> use case (agent prompt + keyword fallback)
  casefiles.py    reading a folder of written descriptions, plus its manifest
  predict.py      the end-to-end Predictor and Forecast
  roles.py        the delivery roster — read from roles.json, or built in
  economics.py    margin, expected margin, breakeven, run rate
  impact.py       what the client gets: handling displaced, net of running it
  report.py       terminal cards
  app.py          the web prototype (stdlib http.server)
  azure.py        Foundry encoder, Fabric reader, Azure ML entry points
  cli.py          serve / predict / model / portfolio / encode
experiments/
  make_engagements.py   the synthetic corpus generator — read this first
  engagements.jsonl     its committed output
examples/usecases/
  *.md                  twenty written scoping notes to feed it
  manifest.jsonl        their lineage, and nothing else
```
