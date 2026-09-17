# Functionality marketplace prototype

Token Yield can give clients a marketplace for composing a product from
LEGO-like functionality bricks. Instead of choosing technical tasks first,
clients shop for customer-facing capabilities and see their underlying task
composition, estimated token usage, AI budget, and potential business benefits.

The [interactive HTML prototype](../marketplace-prototype.html) explores this
idea on the `feat/marketplace` branch. It is a standalone design experiment,
not a production storefront or a purchasing system.

> [!IMPORTANT]
> The original standalone marketplace uses illustrative coefficients and prices.
> The paired sales and operations pages also support an explicitly enabled
> Foundry agent pilot. Keep its measured-data predictions separate from the
> offline simulation. Business benefits remain hypotheses, not measured improvements.

## Working local project pipeline

The sales page also supports an actual local backend. For offline execution, start it from the demo
worktree using a Python environment with the repository available:

```text
python -m examples.marketplace_demo_server --port 8765
```

Open <http://127.0.0.1:8765/marketplace-sales-demo.html?mode=custom>.
If that port is already occupied, use `--port 8766` and change the URL to match.
Choose **Load security project**, then **Start step-by-step + open operations**.
The companion [operations page](../marketplace-operations-demo.html) opens
automatically. If the browser blocks popups, use the operations link in the
sales page. Both pages read the same backend run, identified by its run ID.

The new run pauses before its first stage. Click **Next step** in operations to
execute one real backend stage, inspect its output, then continue when ready.
This is not playback: future stages have not executed, and no final estimate
exists until prediction and artifact saving finish. Cancel an unfinished run
from operations to release its worker. A stage left paused for 30 minutes fails
explicitly and releases its worker. If the sales page loses its connection, use
**Refresh project status** to reconnect to the same run without restarting it.
Older completed runs remain inspectable,
but cannot be stepped through again; start a new project to demonstrate execution.

The HTTP API retains automatic execution when `execution_mode` is omitted.
The sales page defaults to `execution_mode: "step"` and also offers automatic
execution explicitly.

The offline backend performs these operations rather than animating canned progress:

1. Decompose the security-documentation brief with explicit offline rules.
2. Build a wiki from original fictional sample documents and document links.
3. Compare text evidence against a demo reference-requirements checklist.
4. Engineer the features, including any new-function feature.
5. Fit the initial synthetic catalog and predict, or abstain for a new feature.
6. Generate labeled synthetic token samples for the expanded schema.
7. Fit the expanded regression models using the repository's `RidgeLinearModel`.
8. Evaluate on separate synthetic samples.
9. Predict again using the newly fitted model.
10. Save the run, training evidence and model artifacts locally.

Choose **Try a new function** to add Policy-as-code validation and rerun.
Operations shows the new feature, any initial unsupported prediction, sample
rows, fitted coefficients, evaluation metrics, and the resulting token estimate.
Its event console reflects backend events also printed in the server terminal.
Opening operations alone never starts a training run.

Review the proposed functions and approve **Replace build with this complete
pipeline** to use the returned input/output prediction in the sales estimate.
The whole project is one composite brick, so its internal functions are not
charged a second time. Its model and document features are frozen to that run;
rerun to change them. Monthly volume and commercial assumptions remain editable.
Staffing, tool costs and ROI are not outputs of the trained regression.

> [!IMPORTANT]
> In offline mode, model fitting and document processing are real local operations, but the
> token measurements are synthetic. Low test error against that simulator is
> not evidence of accuracy on real LLM calls. Named model rates remain scenario
> assumptions. No LLM provider, live web search or AI deep-research service runs.

The reference documents are a fictional demo baseline, not an exhaustive
security standard or certified ground truth. Found, partial and missing text
evidence are documentation diagnostics, not proof that controls work. An expert
must confirm the applicable requirements, validate the evidence, and approve
any compliance conclusion.

Run artifacts are stored in the ignored `.demo-runs/` directory by default.
Each offline run starts from a reproducible synthetic baseline; it does not accumulate
learning across submitted projects or execute the newly named function itself.
The run list covers the current server process. Saved artifacts remain on disk
after a restart and can be inspected through the original run URL, but are not
automatically reloaded into that list. Interrupted runs cannot resume.
Do not enter confidential briefs into a shared prototype. The server is for
localhost demonstrations, not production hosting.

## Real Foundry agent pilot

The live mode uses the existing Azure deployment rather than creating hosted
agent infrastructure. Agent roles run in the local orchestrator and make separate
Foundry requests. Distinct role instructions do not imply different underlying
models: this pilot uses the pinned GPT-5.4 deployment for every role and workload.

Enable paid execution only with a new, explicit campaign approval:

```text
python -m examples.marketplace_demo_server --port 8766 --enable-foundry --agent-budget-usd 25 --agent-approval-id marketplace-agent-pilot-usd25-v1
```

The budget is for the entire campaign, not each run. The default durable state
directory is `.demo-runs/agent-state`. Reuse that directory on restart; do not
change directories to bypass the approved total. Starting without
`--enable-foundry` makes no paid calls and rejects live-run submissions.
The browser cannot configure an endpoint, increase the budget, or change the
approved deployment. Only one Foundry run can execute or wait at a time.

The live workflow separates three kinds of evidence:

1. Agent proposals, critiques and decisions describe a suggested decomposition.
   They are not measured labels or proof of completeness.
2. Task-execution usage returned by Foundry supplies the regression labels.
   Discussion and coordination tokens are recorded separately as pilot overhead.
3. Fitted token predictions estimate repeated execution of the scoped prompt
   contracts. Staffing and ROI still come from the client's editable assumptions.

The operations view exposes public role responses and actual tool outputs, not
private chain-of-thought. In step mode, approving a stage authorizes its work;
opening either page or reading a catalog does not start inference. Cancellation
stops before subsequent calls, but cannot undo a request already sent to Azure.
Unknown outcomes remain reserved against the campaign budget rather than being
assumed free.

Submitted custom descriptions are sent to the configured Azure deployment.
Use fictional demonstration briefs, not confidential customer information.
The bundled documents remain fictional, and this pilot does not ingest user
documents, browse the web, validate operating controls, or certify compliance.
Source-bound research is not a replacement for a deployed deep-research service.

The design draws on the bounded orchestration and finite feature-building
patterns in [agentic-labeling at its inspected revision](https://github.com/ginaecho/agentic-labeling/tree/efca794b3df6c15549e67ca8ac8dc26fed13dc3c).
That repository's three judges critique independently. This pilot adds an
explicit peer-review round before the orchestrator makes a decision. Agent
agreement must not be mistaken for verified ground truth.

After a live pilot completes, **Pre-predicted GPT-5.4 bricks** reads the stored
catalog without calling the provider. Select multiple supported variants and
set each one's monthly volume. Their saved input/output predictions stay fixed;
editing the shared context, illustrative atom coefficients or scenario LLMs
does not alter those tokens. Commercial rates remain editable planning inputs.
Measured catalog choices cannot be mixed with illustrative choices, and no
synthetic routing or setup tokens are added to a measured forecast.

For a custom project, review the accepted functions and approve its complete
workflow instead. It replaces the build with one composite forecast rather
than charging its internal functions twice. Unsupported results have no
financial apply action. Reference-context pilot forecasts do not establish
accuracy for arbitrary customer documents or live-web research.

## Feedback workshop and execution console

The sales studio remains additive: manual variant selection, empty initial build,
ROI at the top, offline fitting and explicitly enabled Foundry runs are preserved.
In **Describe your idea**, the new-function workshop provides three original
invented briefs. Loading a brief never submits it or switches to paid mode.
All provider workloads still use bundled fictional reference documents, not
the cited repositories' code or real scheduling/support/archive records.

The new [execution console](../marketplace-console.html) displays the same run ID
as sales and detailed operations. It polls actual backend snapshots, appends
chronological persisted events and exposes operation starts, waits, roles,
validated numeric rows, training reuse, fresh holdouts, metric reviews, forecasts
and publication. Expand each event for public output. It is an **in-process
operation stream, not a shell**; there is no shell endpoint, generated code
execution, fake command typing or hidden model reasoning.
Read-only persisted replay, missing runs, disconnection, failure and cancellation
are explicit. Next releases exactly the current gate; cancellation cannot undo
an in-flight request. Opening any page does not invoke a provider.

### Original public-metadata-inspired scenarios

Metadata was independently queried on 2026-09-17 through the GitHub repository API.
Only domain ideas were used; no licensed repository source was copied into prompts.
These are invented fixtures, not customers, production results or endorsements.

| Story | Public metadata source | Original invented exercise |
| --- | --- | --- |
| Archive Atelier | [paperless-ngx](https://github.com/paperless-ngx/paperless-ngx), [API](https://api.github.com/repos/paperless-ngx/paperless-ngx): document scanning/indexing/archiving; GPL-3.0 metadata | Request a source-only retention exception ledger and review novelty |
| Booking Blocks | [cal.diy](https://github.com/calcom/cal.diy), [API lookup](https://api.github.com/repos/calcom/cal.com): scheduling; the cal.com lookup redirected to cal.diy; MIT metadata | Compare a retention exception register with the previously published ledger |
| Care Crew | [Chatwoot](https://github.com/chatwoot/chatwoot), [API](https://api.github.com/repos/chatwoot/chatwoot): chat/email/omnichannel support; license metadata NOASSERTION | Leave the function field empty and infer a handoff obligation mapping gap |

Run them in order against one persistent state directory to inspect compatible
training-row reuse. Live outcomes are not predetermined: agents may reuse,
establish, abstain or disagree. The mock provider has explicitly labeled,
rule-based fixture outputs, including register/ledger normalization and a handoff
keyword. Those rules exist only in the mock module, not the live orchestrator.

### Contract and prediction boundaries

Both explicit requests and description-inferred gaps enter the same capability
resolver. Exact normalized names reuse. Name-token Jaccard scores provide
candidate evidence only, never semantic equivalence. An orchestrator compares
contract scope and may leave ambiguity unresolved.

Creation now requires three independent atom proposals, three peer reviews of
all proposals, and orchestrator reconciliation. Disagreement, differing final
atom counts or out-of-scope work prevents establishing that contract. The finite
atom counts compile into ordered source-only steps, and workload outputs must
contain exactly one result for each step in order, plus valid source citations.
This is a bounded document transformation, not arbitrary named-function execution.
Output shape and citations are checked; semantic correctness still requires
human review, and mock outputs are not evidence of live model capability.

Contract prompts label `output_contract` values as formatting examples: numeric
ones are not proposed or required allocations, and `true`/`[]` do not request
assent or omission of dissent. Reviewers judge their final substantive contract;
resolved differences from discarded proposals belong in the rationale. Genuine
unresolved objections remain dissent even if every final count matches.
Reconciliation must carry forward final-review false votes and dissent, not
reinterpret them as assent. The existing schema, validation and unanimous
no-dissent gate remain unchanged; prior public messages remain intact.
Offline regressions check transmitted prompt semantics and veto preservation,
not whether a future live model will interpret the clarification successfully.

The `source-atoms-v2` compatibility fingerprint excludes older execution contracts
from predictions and training reuse without deleting historical evidence.
Contracts, steps, hashes, feature builders, reference-context bounds and model
versions are saved. Reused training rows require intact original evidence hashes;
old holdouts are never promoted or reused. Holdouts are fresh run-specific groups
but share fictional templates: this is a small exploratory pilot, not independent
production validation. Rejected metric reviews do not publish a new current model.

**Measured combinations are unavailable.** Project composition is explicitly
the sum of independent brick forecasts, not a jointly executed workflow.
No cross-brick interaction costs or shared-context savings are measured.
Numeric forecasts remain withheld for unsupported scope, unresolved project
decisions, incompatible models/contracts or predictors outside measured support.
The local input/output ridge models are retrained; GPT is never fine-tuned.

### Forecast provenance in sales estimates

Stored models supply `source` to the public catalog, each brick prediction, and
whole-project forecasts. Sales selections, quotes, configuration dialogs, reviews
and JSON exports retain that evidence source. `measured-foundry`,
`mocked-test-provider` and offline `synthetic` evidence have distinct labels and
export bases. Absent or conflicting provenance is labeled `unknown`; neither a
Foundry runtime route nor a GPT-5.4 model name establishes measured evidence.
Saved legacy results can use their explicit result/training source, never the
currently connected runtime as a substitute.

`forecastMode: reference-context` controls frozen GPT-5.4 forecast arithmetic
separately from `predictionSource`. Mock reference forecasts use the same frozen
tokens and commercial rates as measured reference forecasts, with no invented
routing/setup tokens. Context and token-factor changes do not change these
snapshots. Manual/offline scenario calculations retain their existing behavior.
Mixed reference evidence sources cannot form a single quote.

Exported `liveLLMConnected` requires both enabled server runtime metadata and
`runtimeSource: measured-foundry`; a mock runtime never sets it. This reports
server configuration, not a new authentication/availability probe or a claim that
reading a saved quote made a model call. Historical evidence provenance remains
independent from current runtime availability.

### No-paid-call validation

Use a separate loopback port and scratch state; do not point tests at a live ledger:

```powershell
python -m examples.marketplace_demo_server --port 8772 --run-dir .feedback-browser --mock-agents
```

Open <http://127.0.0.1:8772/marketplace-sales-demo.html?mode=custom>.
Select the agent runtime, which is visibly labeled **MOCK**, then start explicitly.
`--mock-agents` cannot combine with paid flags, uses no authentication and never
dispatches network requests. Its durable accounting amounts test the budget
protocol with synthetic usage; they are not actual spending.

```powershell
python -m pytest tests\test_marketplace_agents.py tests\test_marketplace_demo.py tests\test_marketplace_feedback.py -q --basetemp=.feedback-tests
npm install --prefix .feedback-tools --no-save --no-package-lock playwright
$env:PLAYWRIGHT_MODULE = Join-Path (Get-Location) '.feedback-tools\node_modules\playwright'
$env:PLAYWRIGHT_CHANNEL = 'msedge'
node tests\marketplace_feedback_browser.cjs
node tests\marketplace_provenance_browser.cjs
```

Use the repository's configured Python environment. Browser tests use installed
Edge; omit `PLAYWRIGHT_CHANNEL` when Playwright Chromium is installed.
They refuse non-mock provenance before any POST, exercise three persistent runs,
stepping/cancellation, 1440px desktop and 390px mobile layouts, no paid auto-start,
run-ID links, disconnection/reconnect and stale-output clearing. Both whole-project
approval and multi-brick catalog selection are checked through quote/export,
including source labels, the live-connection flag and unchanged frozen arithmetic.
The separate provenance browser suite intercepts **all HTTP** with local fixtures
to cover measured, mock, offline, missing and conflicting evidence, plus a stored
measured forecast viewed on a mock runtime. No real provider is called by these
fixtures, including the measured-provenance response-path test. Its report is
saved under `.feedback-provenance-check`.
Screenshots and
a JSON report are written under `.feedback-browser-check`, not committed.
For a repeat of the creation tests, use a **new mock-only** directory/server;
do not reset or replace any paid campaign state.

### Prepared later campaign

`examples/marketplace-feedback-campaign.json` is a **disabled preparation artifact**:
USD 50 maximum additional authorization, USD 48 operational stop, a new fixed
approval identity, and only the three unchanged reviewed scenario briefs.
It neither executes calls nor changes the historical USD 25 campaign.
The legacy CLI cap remains USD 25.

Only after independent phase-one verification may the operator enable a reviewed
copy and explicitly provide `--enable-foundry --campaign-file <reviewed-copy>`
with the existing pinned connection and a new, durably retained
`--agent-state-dir`. Do not execute this during phase-one validation.
Campaign configuration and scenario metadata are pinned in the new ledger.
Changing that pin cannot reset spending; another campaign cannot open the old
ledger. Unknown usage retains reservations and halts execution. All attempts,
retries and role calls count against the cap. No cross-campaign row import is
implemented: prior evidence is conservatively excluded, not silently reused.

## Sales conversation demo

Open [marketplace-sales-demo.html](../marketplace-sales-demo.html) directly in
a browser. This separate prototype builds on marketplace layout A; the original
three-layout explorer remains available. The original manual and quick-preview
flows need no server; the working project pipeline above requires localhost.

The original illustrative mode provides two entry paths to the same estimate:

1. Choose **Build with bricks**, click Research or Personalized discovery, and
   select multiple variations. Set each variation's execution LLM, monthly runs,
   setup hours, review minutes, and tool fee.
2. Choose **Describe your idea**, expand **Quick catalog preview**, load the research example or enter a brief,
   preview its simulated decomposition, inspect the matched phrases, and
   approve the mapping. Existing matching variants are replaced with defaults,
   not duplicated; unrelated selections remain.

This quick-preview mapper uses local keyword rules, not an LLM. Unknown intent is
reported without adding anything. Negations, conditions and unrecognized
requirements require human interpretation. Editing the brief invalidates
its pending proposal. Approval preserves the description and catalog mapping.

Both flows show input/output tokens, AI cost, tool fees, recurring review,
one-time setup labor, and role capacity for data scientists, architects and
consultants. Staffing is an hours-based scenario, not a learned prediction.
FTE divides setup hours by delivery weeks and weekly capacity; rounded-up
people counts are not full-time hiring recommendations.

The client value case uses unique monthly cases, before/after manual effort,
labor value, a realization percentage and optional incremental contribution.
Benefits are not multiplied by the number of selected bricks.

```text
monthly gross benefit = released hours x labor value x realization + contribution
monthly net value = monthly gross benefit - recurring cost
first-year cost = setup cost + 12 x recurring cost
12-month ROI = (12 x monthly gross benefit - first-year cost) / first-year cost
payback months = setup cost / positive monthly net value
```

Zero cost makes ROI undefined; nonpositive net value does not reach payback.
Zero workloads produce no assumed benefit but can still incur setup and
declared operating costs. Time increases remain negative benefits.
This is an undiscounted scenario with full operation from month one, not a
cash-flow forecast or causal impact estimate.

Manual catalog coefficients and rates are illustrative and editable. No trained
predictor supports those manual variant/model combinations. Named scenario LLM rate cards
are placeholders, not verified provider prices. The same pure calculation
feeds the sidebar, review, ROI and JSON export. Review the build to download
the scenario or copy its full JSON if the browser blocks downloads.

Browser checks cover multi-selection, per-variant LLMs, approval and stale
proposals, unknown briefs, invalid inputs, empty builds, zero workloads,
negative ROI, role capacity, numeric reconciliation, and mobile layout.
These checks verify the prototype's behavior, not empirical model accuracy.

## Try the original prototype

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
