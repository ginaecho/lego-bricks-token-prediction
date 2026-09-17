# Public customer project requests

Six buyer-issued public RFPs provide project-level context for deterministic
LEGO task templates. The saved research verified the organizations' public HTML
pages on September 14, 2026. These are authentic buyer requests, not vendor case
studies or sample RFPs. The catalog preserves links and paraphrases; it does not
reproduce full briefs or supply private client inputs.

Published dates describe the requests, not current procurement availability.
They do not establish contract awards, project completion, or permission to
contact a buyer. No new source verification or paid execution was performed
when creating this catalog.

## Catalog and counting contract

The machine-readable catalog is
[catalog.json](../experiments/customer_requests/catalog.json), with schema
version `customer-requests-v1` and verification date `2026-09-14`.
Project and requirement IDs are stable identifiers. Requirement IDs restart at
`R1` for each project.

| Project ID | Buyer | Split | Listed deliverable units |
|------------|-------|-------|--------------------------|
| `paradigm-website-redesign` | Paradigm Initiative | train | 8 |
| `slls-disaster-dashboard` | Southeast Louisiana Legal Services | train | 4 |
| `climateworks-plastic-demand-review` | ClimateWorks Foundation | train | 6 |
| `campus-compact-evaluation` | Campus Compact | train | 8 |
| `handmade-arcade-strategic-plan` | Handmade Arcade | holdout | 8 |
| `a4l-security-assessment` | Access 4 Learning Community | holdout | 8 |

Each requirement lists one bounded buyer-requested deliverable or component.
Its `brick` and `owner` are our experimental scope annotations, not buyer
terminology, staffing commitments, or evidence of autonomous delivery.
Bricks use existing task slugs from [tasks.py](../token_yield/tasks.py).

* `agent_draft` permits a proposed scoping draft or template, not a completed
  client deliverable. Substantive claims still require authorized evidence and
  human review.
* `human_required` marks facilitation, judgment, implementation authorization,
  certification, or other work that cannot be delegated by this public brief.
* `blocked_private_input` marks delivery dependent on unavailable client
  materials or access. A placeholder can describe the dependency, but cannot
  substitute for the missing evidence.

For deterministic decomposition, traverse projects and requirements in catalog
order and assign one unit to each listed requirement. Repeated brick slugs
accumulate their units; owners and exclusions remain attached to their
requirements. The catalog has 42 listed units: 26 train and 16 holdout.
This is not an estimate of the total work or token cost of any engagement.

One unit is a catalog component, not one page, region, interview, record, product,
support incident, or completed customer project. For example, Handmade Arcade's
four to six sessions form one listed engagement component, not a claim that a
single session meets the RFP. Unknown corpus sizes, page counts, dataset volumes,
assessment counts, and operational workloads remain unknown. A single brick is
a modeling simplification, not an exhaustive delivery workflow.

## Paradigm Initiative website redesign

Source: [Website Redesign Request for Proposal](https://paradigmhq.org/working-zone/website-redesign-request-for-proposal/).
The saved verification used Scope of Work, Deliverables, Timeline, and
Submission Guidelines. Proposals were due November 22, 2024; implementation was
specified as three months from contract signing.

The buyer requested an audit and redesign of its website and subdomains,
including information architecture, accessibility, responsiveness, SEO, and
performance. Requested outputs include a working website, reusable templates,
source code, maintenance documentation, training materials, and six months of
debugging support.

Our proposed adaptation is an audit checklist, navigation plan, implementation
scope, and handover templates. A later authorized delivery could use page
snapshots and source access to implement and test changes; those inputs are not
approved for this pilot. The brief supplies no CMS credentials, source access,
design approval, or deployment authorization. Six months of support is a buyer
service obligation, not a measured model run.

## Southeast Louisiana Legal Services disaster dashboard

Source: [Disaster Data-Driven Outreach Project Technology Consultant](https://slls.org/en/disaster-technology-rfp/).
The saved verification used Disaster Data Dashboard, Outreach Business Process
Improvement, and Timeline. Issued January 27, 2023; proposals were due
February 19, 2023.

The buyer requested a GIS dashboard combining FEMA disaster-assistance
indicators with Census/ACS demographics to guide outreach. It also requested a
technology-enabled staff and volunteer outreach scheduling process, its
implementation, and staff training. FEMA denial counts and reasons may require
additional advocacy to obtain; their availability must not be assumed.

Our proposed adaptation is a dashboard specification, geographic reconciliation
checklist, process plan, and training outline. A future aggregate public-data
pipeline would require separate checks of dataset availability and licenses.
Public datasets are not part of the currently approved pilot input. Client
records, staff and volunteer schedules, and operational system access are
excluded. Actual workflow implementation and staff training require humans.

## ClimateWorks Foundation plastic demand review

Source: [Plastic demand reduction literature review](https://www.climateworks.org/programs/industry/request-for-proposals-plastic-demand-reduction-literature-review/).
The saved verification used Overview, Scope, Submission format, and dates.
Published June 29, 2023. The page conflicts on the submission deadline:
July 29 in the narrative and July 31 in Key dates.

The buyer sought three or four regional reviews of policies and interventions
that reduce plastic demand. North America and Europe must be covered among the
contracts. Requested analysis addresses strengths, limitations, implementation,
applicability elsewhere, plastic volumes covered, and reductions achieved.
Sources should include peer-reviewed and reputable non-academic literature.
The guideline was approximately USD 30,000 per contract over six months,
preferably faster.

Our proposed adaptation is a regional review outline and an empty evidence
matrix with fields for implementation, transferability, quantities, citations,
and uncertainty. The six catalog components describe review outputs, not six
regions or a known number of studies. A substantive synthesis requires licensed
or openly available literature and human review of causal claims. No literature
corpus, paywalled source, private analysis, or invented finding is authorized
for this pilot.

## Campus Compact program evaluation

Source: [Measurement & Evaluation Consultant](https://compact.org/news/request-for-proposals-measurement-evaluation-consultant).
The saved verification used Project Overview, Scope of Work, Specific
Deliverables, and Expectations. Proposals were due April 29, 2026; the stated
contract term is May 2026 through March 2027. The budget is up to USD 40,000.

The buyer requested mixed-methods evaluation of its Campus Action Planning for
Civic & Community Engagement initiative. Requested outputs are an evaluation
plan, data-collection tools, interim and outcome reports, a future measurement
framework, reporting templates, a monitoring dashboard, and evaluation of a
Chief Engagement Officer learning community.

Our proposed adaptation is draft instruments, metric definitions, report
skeletons, and a dashboard specification. These are not validated instruments
or evaluation results. Participant responses, baselines, consent, and internal
implementation records are absent. An actual evaluation would require
authorized evidence, appropriate de-identification, and human interpretation.
Do not invent results or conduct participant outreach from this brief.

## Handmade Arcade strategic planning

Source: [Strategic Planning Consultant](https://www.handmadearcade.org/rfp-strategic-planning-consultant).
The saved verification used Project Description, Deliverables, Project
Timeline, and Budget. Proposals were due January 30, 2026, with preferred
kickoff in early April and completion in December 2026. The budget is up to
USD 18,000, with possible scope adjustments.

The buyer requested organizational and program assessment, constituent
engagement, environmental and competitive analysis, financial sustainability
review, and a three-to-five-year strategic plan. Deliverables include a
workplan, four to six engagement sessions, an assessment report, draft and final
plans, executive summary, budget/staffing framework, and board presentation.
The catalog groups draft and final plans as one planning component.

Our proposed adaptation is a workplan, engagement agenda, assessment outline,
strategic-plan skeleton, and budget/staffing template. These are proposed
formats, not buyer-supplied templates or organizational findings. Internal
documents, financial records, and constituent notes are unavailable.
Facilitation, financial judgment, final priorities, and board approval remain
human responsibilities; no sessions or presentation are authorized.

## Access 4 Learning Community security assessment

Source: [Request For Proposal (RFP)](https://a4l.org/request-for-proposal-rfp/).
The saved verification used Purpose and sections 1.3 and 3.1-3.2. Published
opening and closing dates were August 26 and October 31, 2025; the source labels
these dates approximate.

The buyer requested a Global Education Security Standard third-party
certification assessment framework and subsequent product assessments.
Requested components include risk/size tiers, reuse of applicable certification
evidence, product-level assessment, remediation windows and action plans,
assessment documentation, validity/recertification rules, and breach handling.
Assessment of A4L itself or schools/districts directly is excluded.

Our proposed adaptation is a framework outline, evidence checklist, coverage-gap
template, and remediation template. None is an actual product assessment,
certification, security test, or permission to access systems.

Section 3.2 restricts assessment-data reuse: it remains the provider's property,
ordinarily must be destroyed after assessment, and cannot be used for secondary
purposes. Future private assessment datasets are not available for training.
Product evidence, incident details, and certification decisions are outside the
pilot; framework drafts do not override these restrictions.

## Delivery decomposition versus paid scoping pilot

The project catalog models requested client delivery components. The separately
approved paid pilot uses GPT-5.4 on the user's own Azure deployment with a total
USD 50 budget cap. Approval covers only these paraphrased summaries and
deterministic templates as model inputs. Source URLs provide provenance, not
authorization to fetch full briefs or linked data into prompts.

Paid outputs are scoping artifacts: proposed task breakdowns, assumptions,
dependency registers, acceptance-check templates, and human approval points.
They are not completed websites, dashboards, literature reviews, evaluations,
strategic engagements, or certified assessments. Owner annotations do not expand
the input or action authorization. Procurement budgets and contract durations
are not inference budgets or measured costs.

The pilot must not ingest full client inputs, conduct customer contact, submit
proposals, deploy systems, retrieve new evidence, or execute operational client
work. This catalog records authorization boundaries; it does not run the pilot,
assert model availability, or claim that any paid result already exists.

## Split, rights, and measurement limits

The first four projects, Paradigm, SLLS, ClimateWorks, and Campus Compact, are
`train`. Handmade Arcade and A4L are execution `holdout`. Both groups' source
summaries and template design are already visible. This is a new outcome
holdout, not a sealed-source or independent-template confirmatory experiment.
Here, `train` identifies the development/calibration partition; it does not
authorize model training on buyer data.

Freeze the deterministic templates, acceptance checks, and prediction procedure
before examining holdout execution outcomes. Do not tune on those outcomes and
then describe the same observations as untouched holdout evidence.

Keep three layers separate:

1. Source facts: buyer, title, URL, paraphrased requests, dates, and caveats.
2. Experimental specification: selected components, brick/owner annotations,
   permitted inputs, dependency placeholders, checks, and split.
3. Measured labels: actual model and harness, token channels, tool calls,
   retries, acceptance, and cost using applicable dated rates.

No full client inputs, reference answers, or measured token/cost labels are
available in this catalog. Any later pilot labels measure scoping artifacts
under its restricted inputs, not total customer delivery cost. Six selected
briefs add realism and variety but are not a representative or execution-ready
training benchmark and do not replace controlled experiments.

Public availability is not a blanket reuse license. Retain links and
paraphrases, check source and dataset terms before further reuse, and obtain
authorization before accessing nonpublic materials. Missing evidence stays
explicitly missing. Private data rights, especially A4L's secondary-use
restriction, take precedence over proposed experimental adaptations.

## Deterministic decomposition and optimization

The executable contract is [template.json](../experiments/customer_requests/template.json).
[customer_decomposition.py](../token_yield/customer_decomposition.py) validates
the catalog and compiles a versioned plan. The same catalog and template produce
the same counts, ordering, prompts, and SHA-256 identifiers. Changing any
requirement, annotation, exclusion, or instruction changes the identity.
GPT output wording is not promised to be deterministic.

Two decompositions must not be confused:

| Representation | Counting rule | Purpose |
|----------------|---------------|---------|
| Customer deliverable view | One annotated brick per catalog requirement | Explain the proposed project scope using the existing `Decomposition` interface; full workload is unknown |
| Executable scoping template | Extract, Classify, and Plan each have one unit per requirement; Report has one unit per project | Measure producing a scoping packet from the approved public paraphrase |

The scoping packet contains:

1. Exact requirement evidence extracted from the supplied paraphrase.
2. A classification preserving the reviewed brick and ownership annotations.
3. A proposed scoping artifact for each requirement, with prerequisites marked
   unverified and human review required.
4. A scope report preserving the provenance URL, coverage, excluded actions,
   and explicit `not_executed` delivery status.

These four operations are independent projections of the same frozen input.
They do not consume another operation's generated response. This permits
work-preserving batching without inventing delivery dependencies. Actual client
work needs its own input discovery, dependency graph, and acceptance process.

The optimizer enumerates all eight contiguous partitions of the four operations.
It rejects dropped, repeated, or reordered operations and preserves the total
output allowance and all required units. Only the fully separate four-call and
fully batched one-call layouts are measured in this pilot. Intermediate layouts
receive model estimates but remain ineligible for a quality-backed recommendation.

Automatic checks verify exact requirement coverage and evidence, ownership,
source identity, exclusions, review gates, and delivery status. Proposed
artifact text and report prose still need independent human review. Consequently,
the optimizer reports the cheapest fully measured, contract-passing arm as a
candidate, but leaves `production_recommendation` null. It does not claim
quality noninferiority from JSON checks.

## Running the bounded pilot

[pilot.json](../experiments/customer_requests/pilot.json) freezes the approved
GPT-5.4 deployment/version, reasoning `none`, verbosity `low`, two replicates,
and randomized job order. The total approval is USD 50, with a USD 48
operational stop. The configuration deducts USD 0.1168 of reconciled safety
accounting for an earlier stopped attempt, leaving a USD 49.8832 run cap and
USD 47.8832 stop. That earlier attempt's retail-rate cost was USD 0.0094.
There are 40 training calls and 20 outcome-holdout calls. Each operation receives
a 1,600-token output allowance; the combined call receives 6,400.
No tools, external retrieval, or automatic retries are enabled.

The existing `FoundryDispatcher`, `HardBudget`, and `Pricing` implementations
are reused by [customer_pilot.py](../token_yield/customer_pilot.py).
Credentials are obtained in memory through Entra authentication and are never
written into requests or artifacts. The stored request bodies contain only the
approved public paraphrases and scoping instructions, not HTTP authorization
headers.

Azure responses return the alias `gpt-5.4`, not the dated string initially
expected. The runner verifies version `2026-03-05`, SKU `GlobalStandard`, and
successful provisioning through Azure CLI before generation, before the holdout
phase, and after completion. A mismatch halts execution. The deployment's
existing upgrade policy is `OnceNewDefaultVersionAvailable`; it was not changed.
These metadata observations bracket execution but are not independent per-call
backend-version attestations. This limitation prevents claiming an immutable
backend pin from the response alias alone.

Azure's public retail-price API was checked on September 14, 2026 for
`germanywestcentral`, GPT-5.4 Global Standard, normal synchronous short-context
requests: USD 2.50 per million uncached input tokens, USD 0.25 cached input,
and USD 15.00 output. The meter IDs and effective date are in the configuration.
Reasoning tokens are a subset of output, not a second billing category.
These are rate-calculated costs, not reconciled invoices, tax, or discounts.
Merging operations into one request does not use the discounted Azure Batch API.

Before generation, the runner persists a conservative reservation using
32,768 input tokens and twice the output allowance at the existing safety rates
of USD 20/M input and USD 200/M output. Prompts exceeding the byte-based guard
are rejected before dispatch. This is a conservative bound for the controlled
text-only payload, not an exact tokenizer claim. Observed bound violations halt
the campaign. Unknown usage retains its reservation; incomplete responses with
usage retain their incurred cost. Missing raw cache, reasoning, or total
telemetry halts execution rather than silently substituting zeros. Settlement
failures still persist known usage and updated accounting before propagating.
The input-only counting capability probe omits generation settings and is not
an exact generation-request baseline. Unsupported counting is reported as
unavailable and blocks a confirmatory exact-baseline claim.

Use the existing Python environment. Preview is offline:

```powershell
.\.venv\Scripts\python.exe -m examples.customer_request_pilot --run-dir runs\customer-preview
```

The following command spends API credits and requires valid Azure authentication:

```powershell
.\.venv\Scripts\python.exe -m examples.customer_request_pilot --run-dir runs\customer-pilot --execute
```

Each directory is single-use, including previews. Reusing a directory fails
before dispatch. After any failure or interruption, do not create a replacement
paid run without reconciling prior reservations and remaining authorization.
The USD 50 approval is a total additional spending limit, not a new allowance
per directory. No automatic restart or ambiguous-attempt replay is provided.

## Augmented campaign setup: September 16, 2026

`experiments\customer_requests_augmented_20260916` contains a separate catalog
with 27 projects and 230 requirements, its ingestion report, and matching
`template.json` and `pilot.json` files. The original six projects and their
splits are unchanged. The expansion includes two explicitly labeled job
postings and one sample template; not every source is an issued RFP.

The new `customer-pilot-v2` configuration declares `project_counts` explicitly:
19 training projects and 8 holdout projects. The runner requires these counts
to match the catalog, with at least three training projects for the existing
leave-one-project-out fitting procedure and at least one holdout project.
The original `customer-pilot-v1` configuration still requires four training
and two holdout projects.

The v2 configuration also pins the canonical JSON SHA-256 hashes of its catalog
and template. These are `content_hash()` values, not raw-file hashes from the
ingestion report. Changes to requirements, annotations, splits, instructions,
or output limits require review and updated pins; changing whitespace alone
does not change the canonical identity.

The operation template is unchanged from the original pilot: four independent
scoping operations, a 1,600-token allowance per operation, and 6,400 tokens for
the combined call. GPT-5.4 version `2026-03-05`, Global Standard, reasoning
`none`, verbosity `low`, two replicates, and the 32,768-token input ceiling are
preserved. Preparing the entire augmented catalog describes a fresh 270-call
campaign: 190 training calls and 80 holdout calls. Preparation executes nothing
and does not merge existing measurements. Executing this full plan later would
remeasure the original six projects alongside the 21 new sources.

The user selected carry-forward budgeting rather than a new allowance. The
configuration's `prior_attempt` aggregates both earlier customer-scoping runs:
USD 0.38838 cumulative retail-rate cost and USD 4.70392 conservative safety
accounting, with no active reservations. This leaves a USD 45.29608 run cap
and USD 43.29608 operational stop within the original USD 50 / USD 48 ceilings.
The price metadata retains its actual September 14 retrieval date; no fresh
price or deployment observation is implied by offline setup. Budget stops can
halt collection before all planned calls complete.

**Paid execution is disabled.** The new configuration has
`execution_approved: false`; the runner rejects it before creating a run
directory, authenticating, counting tokens, or generating responses, even if
`--execute` is supplied. Enabling that field requires separate explicit
authorization after reviewing the inputs, protocol, and remaining budget.
No new metered labels or completed run are claimed by this setup.

An offline preview can be created in a new, single-use directory:

```powershell
python -m examples.customer_request_pilot `
  --experiment-dir experiments\customer_requests_augmented_20260916 `
  --run-dir runs\customer-augmented-preview
```

All calls from a project inherit its catalog split. The original two holdout
projects already have observed outcomes, so the combined holdout is not wholly
new blind evidence. Labels would still measure the four-operation scoping
protocol, not full customer delivery or every annotated brick type.

## Replacement Foundry deployment and fresh holdout: September 17, 2026

`experiments\customer_requests_fresh_holdout_20260917` is a separate revision,
not an edit to the earlier catalogs, previews, or measured runs. It retains
all 27 source briefs and 230 requirements. All six previously examined
projects are now training/development sources; eight of the 21 new projects
are holdout. Selection uses `random.Random(20260916).sample(sorted(new_ids), 8)`
without generation outcomes. `split_revision.json` records every old and new
assignment and links back to the original ingestion evidence.

The `customer-pilot-v3` configuration uses the resource's OpenAI v1 endpoint,
`https://msfoundry-hackathon.openai.azure.com/openai/v1`, rather than the
Foundry project URL ending in `/api/projects/foundry-hackathon`. Its
`azure_resource` object records the tenant, subscription, resource group,
resource name, and region used for deployment metadata and authentication.
The model remains GPT-5.4 `2026-03-05`, Global Standard, in `eastus`.
The observed allocation is 5,000 requests/minute and 500,000 tokens/minute;
this shared quota does not guarantee that throttling cannot occur.

Scoped authentication obtains a token for the configured subscription and
checks the returned tenant/subscription, without changing the Azure CLI's
default account or using unrelated service-principal environment credentials.
The refreshable CLI cache is consulted before each generation request so a
long campaign does not retain one expiring bearer token for the entire run.
Tokens and authorization headers are not saved. Deployment metadata is still
checked before generation, before holdout, and after completion.

The user authorized paid collection without a budget restriction. Instead of
accepting infinity or disabling spending protection, v3 records a finite
`budget_approval`: a USD 453.43 run cap/stop covers the sum of all 270
worst-case safety reservations (USD 453.4272). The cumulative approval is
USD 458.14, including USD 4.70392 of prior Azure safety accounting. These
conservative safety values are not expected charges. Normal synchronous
eastus retail rates were retrieved on September 17: USD 2.50/M input,
USD 0.25/M cached input, and USD 15.00/M output. The exact meters and
control-plane observations are retained in `integration_report.json`.

V3 retains explicit execution approval, reviewed catalog/template hashes,
the fixed call count, and finite cumulative budget validation. It also
fingerprints all prepared calls and rejects changed inputs/configuration or
calls before execution. V1/v2 retain their historical endpoint and USD 50
cumulative cap restrictions. Output limits, exact usage requirements,
single-use run directories, and no-automatic-retry behavior remain unchanged.
Creating this integration configuration does not itself generate labels.

### Completed fresh-holdout token collection

The completed dataset is
`runs\20260917_customer_scoping_fresh_holdout_v1\records.json`.
All 270 requests completed with measured usage: 190 training calls from
19 projects and 80 holdout calls from eight previously unmeasured projects.
No generation attempts were retried. The earlier Azure datasets and the
separate Copilot metering probe are not merged into these records.

The campaign consumed 212,892 input tokens and 119,139 output tokens:
332,031 total tokens. Cached input was 2,048 tokens (a subset of input), and
reasoning usage was explicitly zero. Retail-rated cost was USD 2.314707,
not a reconciled invoice. Conservative settled safety accounting was
USD 28.08564, with no outstanding reservations.

Of the 270 responses, 244 met the automatic artifact contract and 26 were
flagged for format or requirement-coverage defects. All still have valid
consumption labels; quality flags remain available for separate analysis
rather than silently filtering incurred usage. These labels measure scoping
requests, not successful delivery of the underlying customer projects.

For token regression, the target is `usage.total_tokens` (or the separate
input/output channels), not `rated_cost_usd`. `quote` contains the
pre-execution features. The runner's saved `models.json` and `predictions.json`
remain USD-cost models; their holdout predictions were frozen before the
first holdout request. The input-only counting capability was unavailable on
this deployment, but generation responses supplied the actual usage channels.

## Training and evidence artifacts

### Where the training data points are

The measured dataset is the root JSON array in
[records.json](../runs/20260914_customer_scoping_v2/records.json).
Each row is one actual GPT-5.4 API call, not an entire customer project.
Select `split == "train"` for the **40 training points from four projects**.
The remaining **20 rows**, with `split == "holdout"`, are evaluation points
from two other projects; do not mix them into the original training partition.

For each training point, `quote` contains the pre-execution features
(prompt bytes, planned output allowance, and counts by scoping operation).
`rated_cost_usd` is the regression target, calculated from the actual `usage`
token channels and recorded retail rates. The `prompt`, `output`, `quality`,
project ID, execution arm, and replicate provide context and audit evidence.
The exact fitted row IDs are recorded as `training_call_ids` in
[models.json](../runs/20260914_customer_scoping_v2/models.json).

The six paraphrased requests in [catalog.json](../experiments/customer_requests/catalog.json)
are source inputs, not measured cost targets. The earlier stopped
[attempt](../runs/20260914_customer_scoping_v1/records.json) is retained for
spending reconciliation and is excluded from these 40 fitted training points.

### Model fitting and audit files

[customer_models.py](../token_yield/customer_models.py) fits four competing
models to completed, metered training calls:

| Form | Quote-time inputs |
|------|-------------------|
| Constant | Training mean cost |
| Size | Prompt bytes |
| Size + units | Prompt bytes, total scoping units, declared output allowance |
| LEGO | Prompt bytes, declared output allowance, separate Extract/Classify/Plan/Report counts |

Ridge regularization is fixed at alpha 10 before observing targets. Selection
uses leave-one-project-out cross-validation with equal project weighting; all
replicates and arms of a project stay together. Failed-quality completions
remain cost targets. No holdout rows may enter fitting. These models estimate
scoping-call cost only, not customer project delivery cost or vocabulary-wide
brick prices.

The runner writes model parameters and all per-form holdout predictions before
the first holdout generation. It also predicts the eight candidate layouts.
Four training projects and a shared template do not support calibrated tails,
so `calibrated_interval` remains null rather than displaying an invented range.

| Artifact | Evidence |
|----------|----------|
| `protocol.json` | Frozen catalog, template, configuration, planned calls, candidate layouts, and code hashes |
| `input_count.json` | Exact counting capability or explicit unavailability |
| `records.json` | Pre-dispatch quotes, reservations, outputs, usage, identity, timestamps, and quality checks |
| `*.request.json`, `*.response.json` | Raw generation bodies, without credentials |
| `budget.json` | Separate settled safety amounts and uncertain reservations |
| `deployment-*.json` | Timestamped deployment version, SKU, state, and upgrade-policy observations |
| `models.json` | Training-only model parameters, group splits, CV predictions, and errors |
| `predictions.json` | Timestamped outcome-holdout predictions and candidate estimates |
| `analysis.json` | Held-out errors, measured paired costs, contract checks, and unresolved human review |
| `state.json` | Completion or explicit halt reason |

## Measured pilot results: September 14, 2026

The completed campaign is
[20260914_customer_scoping_v2](../runs/20260914_customer_scoping_v2/).
Its [analysis](../runs/20260914_customer_scoping_v2/analysis.json),
[records](../runs/20260914_customer_scoping_v2/records.json),
[models](../runs/20260914_customer_scoping_v2/models.json), and
[frozen predictions](../runs/20260914_customer_scoping_v2/predictions.json)
retain the evidence. The directory suffix describes the execution attempt;
the decomposition template remains `customer-scoping-v1`.

### Cost prediction

All 60 calls completed with explicit telemetry: 40 training and 20
outcome-holdout calls. The four training projects selected LEGO through
leave-one-project-out validation before the first holdout call. Predictions
were saved before holdout dispatch; code hashes were independently checked
against the frozen protocol after completion.

The metric below is mean absolute error in **USD per call**, averaged equally
across projects. It is not percentage error, and is not directly comparable to
the older token-MAPE experiments.

| Predictor | Training group-CV MAE | Outcome-holdout MAE |
|-----------|----------------------|---------------------|
| Constant | $0.003312 | $0.003519 |
| Size | $0.001633 | $0.001859 |
| Size + units | $0.001226 | $0.001316 |
| LEGO | $0.000653 | $0.000544 |

LEGO reduced held-out MAE by **58.7% relative to the strongest simple baseline,
Size + units**, in this pilot. This supports distinguishing the four scoping
operations for these short prompts. It does not validate the entire delivery
taxonomy, full-project estimates, other models, independent templates, or
calibrated uncertainty intervals. There are only four training and two
outcome-holdout projects. Both held-out briefs have eight catalog requirements,
and all briefs share a template. Their outcomes are now observed and must not
be described as unseen in subsequent tuning.

### Batching and contract fidelity

Separate execution cost $0.214315 across its 48 calls. Batched execution cost
$0.164665 across its 12 calls: **23.2% lower aggregate measured cost**.
All 12 batched responses passed the automatic contract checks; 41 of 48
separate responses passed. No failed output was repaired or selectively retried.
All completed calls, including contract failures, remain cost targets.

| Project | Observed cost reduction with batching | Both arms passed every contract check |
|---------|--------------------------------------|--------------------------------------|
| Paradigm website redesign | 23.6% | Yes |
| SLLS disaster dashboard | 31.7% | No |
| ClimateWorks literature review | 17.7% | No |
| Campus Compact evaluation | 22.3% | No |
| Handmade Arcade strategic plan | 20.9% | No |
| A4L security assessment | 24.1% | Yes |

Seven separate-call responses failed:

* Two ClimateWorks extraction responses omitted required coverage.
* Two Campus Compact and two Handmade Arcade reports returned only a subset
  of the required IDs.
* One SLLS report did not match the requested top-level artifact keys.

For example, one Campus Compact report included only `R3` and `R4`, although
the contract requires all eight IDs. The checker correctly rejected it.
This motivates a future template revision explicitly separating the current
scoping operation from the delivery-brick annotation, plus schema-constrained
coverage. That is a future repair hypothesis, not a post-hoc change to this
campaign's acceptance rules. Any tuning on these results needs new untouched
outcomes for subsequent validation.

Only Paradigm and A4L receive a provisional cheapest contract-passing arm.
Every production recommendation remains null. Human assessment of usefulness,
completeness, and noninferiority is still required; lower token cost and correct
JSON alone do not prove equally good answers. The six intermediate batching
layouts per project remain estimated rather than empirically quality-validated.

### Spending and execution audit

The completed campaign recorded 34,946 input and 19,441 output tokens.
Cached and reasoning tokens were explicitly reported as zero.
Every call's price was independently recomputed from these channels.

| Attempt | Outcome | Retail-rate calculated cost |
|---------|---------|-----------------------------|
| [Initial preflight](../runs/20260914_customer_scoping/state.json) | Count API rejected `max_output_tokens`; no generation or reservation | $0 |
| [Alias check](../runs/20260914_customer_scoping_v1/state.json) | One fully metered training-side call, then halt on the unversioned response alias; excluded from fitting | $0.009400 |
| [Completed pilot](../runs/20260914_customer_scoping_v2/analysis.json) | 60 calls, frozen models and holdout predictions, 53 contract passes | $0.378980 |
| Total | 61 generation calls; no uncertain reservations remain | **$0.388380** |

Total conservative safety accounting was $4.70392, including the earlier
attempt, below the USD 50 cumulative approval. These are retail-rate estimates
and safety amounts, not a reconciled Azure invoice. The corrected input-only
count endpoint reported this model unsupported, while generation succeeded;
the exact input-token baseline therefore remains unavailable.

Validation: **134 targeted tests passed**, including synthetic end-to-end
execution, strict raw telemetry, durable settlement failures, cumulative
prior-spend constraints, deployment drift, frozen holdout predictions, model
serialization, and work-preserving batching. No further paid run was started
after completion. The saved configuration is an audit record for this attempt,
not a reusable fresh spending authorization.
