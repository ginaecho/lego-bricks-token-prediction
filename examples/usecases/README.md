# Use case descriptions

Twenty written scoping notes, of the kind a Microsoft PM or architect actually
has: a paragraph in an email, a page of a statement of work, the output of a
discovery workshop. They are the input to the prototype — paste one in, drop the
file on the page, or run the whole folder at once.

They are **fictional**. The clients are the standard Microsoft sample companies
(Northwind, Contoso, Woodgrove, Fabrikam, Tailspin, Lamna, Relecloud) and no
number in them describes any real engagement.

## Three ways to use them

```bash
# the whole folder, encoded and ranked
python -m project_yield batch examples/usecases

# one card per use case, with every caveat
python -m project_yield batch examples/usecases --cards

# JSON, for a spreadsheet or a notebook
python -m project_yield batch examples/usecases --json > forecasts.json

# or one at a time in the browser: drop the .md file on the description box
python -m project_yield serve --open
```

## What is in the folder

Chosen to span the things that move a forecast, not just to be twenty of them.

| | use case | exercises |
|---|---|---|
| 01 | Northwind supplier invoice intake | the common shape: small per-run scope, 24k runs a month |
| 02 | Northwind warranty claims — phase 2 | **continuation of 01** — reuse discount, higher win rate |
| 03 | Woodgrove periodic KYC refresh | regulated, prose with no countable scope in it |
| 04 | Quarterly disclosure review | large per-run scope, four runs a year |
| 05 | Credit memo first drafts | drafting-heavy, revenue-growth framing |
| 06 | Prior authorisation triage | very high volume, clinical governance load |
| 07 | Clinical coding audit sampling | reconciliation-heavy, compliance funding |
| 08 | Patient portal message drafting | tiny scope, enormous volume |
| 09 | Supplier non-conformance handling | retrieval against history, corrective work |
| 10 | Tender response assembly | 80 units in one run, low volume, big context |
| 11 | Contoso service ticket triage | the classic triage pipeline |
| 12 | Contoso triage — phase 2, Europe | **continuation of 11**, and the industry is only knowable from the parent |
| 13 | Product description generation | pure drafting, sibling of 11 |
| 14 | Benefits claim first assessment | public sector governance multiplier |
| 15 | FOI request handling | retrieval-dominated, statutory deadline |
| 16 | Public consultation analysis | one-off, and a context size the token model has never seen |
| 17 | Outage report reconciliation | monthly batch, two-source reconciliation |
| 18 | Well permit pack preparation | low volume, very large context per run |
| 19 | AI assistant for the operations team | **deliberately vague** — watch what the tool refuses to pretend |
| 20 | Fabrikam claims platform migration | **deliberately enormous** — should trigger extrapolation warnings |

Team-wise: 04 and 06 name a data scientist and a consultant, 14 names six roles
including a change manager, 16 names two data scientists and no change manager,
18 names a consultant as non-negotiable. The other ten leave it open. Security
experts are never named in a note and are always inferred — which is realistic,
and is why they are the role most often missing from a scoping conversation.

## The manifest

`manifest.jsonl` declares the two things prose cannot safely carry — **lineage**
and the **team** — and nothing else. Bricks, industry, goal, scope and volume
are all read out of the description.

Lineage is not recoverable by an encoder: "follow-on to the pipeline we
delivered for Northwind" is obvious to a reader and refers to something outside
the document. A stated team is recoverable in principle but should not be
guessed, because reading it wrongly silently deletes a person from the bill.

```json
{"file": "02-invoice-intake-phase-2-warranty.md", "id": "UC-02",
 "continues": "01-invoice-intake-manufacturing.md",
 "roles": ["solution_architect", "software_engineer", "project_manager",
           "data_engineer"]}
```

Naming `roles` makes those roles certain and excludes every other role on the
roster. Omit it and the base rates from comparable engagements decide who is
needed — "48% likely, 7.4 days when needed". **Ten of the twenty state a team
and ten deliberately do not**, so both paths are exercised. The roster itself
(who exists, what they cost) is [`roles.json`](../../roles.json) at the repo
root.

Files are processed in filename order, so a continuation must sort after its
parent. Drop your own `.md` or `.txt` files in and they are picked up with no
manifest entry at all — they simply have no lineage.

## Read the caveats, they are the point

With no Azure AI Foundry model configured these are encoded by the keyword
fallback, and it says so on every estimate. It reads spelled-out quantities
("nine header fields") and attaches each to the nearest task, and it separates
throughput from scope ("400 claims a day" is 12,000 runs a month of a one-claim
pipeline, not a 400-claim one) — but it sees words, not intent.

Where it has to guess, it records the guess: 19 names no industry and no goal
and says so; 12 does not name its industry either and takes it from its parent.
Those notes travel with the forecast into every card and every JSON payload.

## The impact figure

The headline number is what the client gets: handling time displaced, at their
own loaded cost, less what the pipeline costs to run. It is the number that
separates these twenty from each other more than anything else does — 01 runs
24,000 times a month and 04 runs four times a year, from a similar amount of
build.

It is also the least evidenced thing on the page: arithmetic over placeholder
handling times, a placeholder hourly cost, and a placeholder deflection rate
(the share of the work automation actually removes, deliberately below 1.0).
Every card names those assumptions inline.

And the standing caveat applies to all twenty: the token budget comes from a
model fitted on real measured agent runs, and everything else comes from a model
fitted on a **synthetic** engagement corpus. See
[`docs/product-prototype.md`](../../docs/product-prototype.md).
