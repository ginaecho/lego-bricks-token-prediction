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

## Offline project-level forecasting

The same measured campaign can train whole-scoping-project forecasts without
another model call. This extends the existing customer pipeline rather than
introducing a separate framework or changing the historical experiment.

```powershell
.\.venv\Scripts\python.exe -m examples.customer_request_pilot `
  --train-from-run runs\20260914_customer_scoping_v2 `
  --run-dir runs\20260915_customer_project_forecast_v1
```

The destination must be new and outside the source run. The command performs
no authentication, network access, or paid inference. It validates frozen
catalog/configuration hashes, declared prompts and layouts, raw request/response
hashes, usage telemetry, and usage-to-rate cost reconciliation before training.
It refuses incomplete executions instead of treating partial spend as the cost
of a completed project.

### Data and model reuse

* [customer_pilot.py](../token_yield/customer_pilot.py) aggregates calls and
  writes the training artifacts.
* [customer_models.py](../token_yield/customer_models.py) reuses the existing
  standardization, ridge regression, serialization, and project-grouped
  evaluation utilities from [robust.py](../token_yield/robust.py).
* [customer_request_pilot.py](../examples/customer_request_pilot.py) exposes
  the offline command alongside the existing preview and explicit execution
  modes. Training and paid execution flags are mutually exclusive.

One observation is one project, execution arm, and replicate. The sixty calls
become twenty-four observations: six projects, two arms, and two repeats.
The original split remains four training projects (sixteen observations) and
two historical holdout projects (eight observations). Neither repeated runs
nor constituent calls are counted as independent projects.

The command separately fits input-token, output-token, and retail-rate cost
targets. Each target compares five forms:

| Form | Declared predictors |
|------|---------------------|
| `constant` | Intercept only |
| `size` | Total planned prompt bytes |
| `size+units` | Prompt bytes, total scoping units, output-token allowance |
| `lego` | Prompt bytes, output-token allowance, four scoping-operation counts |
| `workflow` | LEGO features plus planned call count and largest operation batch |

The largest form has eight nominal predictors; constant columns are not fitted.
Correlated counts are not independently identified per-brick prices. Ridge
strength remains fixed at 10, and each target's form is chosen using
leave-one-training-project-out, equal-project-weight MAE. All repetitions and
arms of the validation project remain outside its training fold.

The four measured operations remain `extract`, `classify`, `plan`, and `report`.
They are not silently relabeled as measurements of thirteen new operations.
The model has no measured tool, retrieval, clarification, retry, or full-delivery
effects. Runtime and execution policy are compatibility metadata, not learned
effects without variation. Industry labels are not fitted from one or two
examples per industry.

### Saved forecasts and their limitations

The output directory contains:

* `project_records.json`: the aggregated measurements, source call IDs,
  quote-time features, original splits, and public provenance URLs.
* `models.json`: reloadable parameters, feature definitions, training-only
  support ranges, runtime settings, and source/code hashes.
* `predictions.json`: forecasts labeled in-sample or historical holdout.
* `analysis.json`: comparisons, accounting totals, and limitations.

Use `forecast_project(artifact, quote, runtime)` from
[customer_models.py](../token_yield/customer_models.py) to predict from saved
parameters without refitting. A quote contains `context_bytes`, `prompt_bytes`,
`planned_output_tokens`, `counts`, `planned_calls`, and
`max_operations_per_call`. These describe the four independent scoping
operations over the same supplied brief, not a general-purpose agent graph.

The forecast reports marginal-range/layout extrapolation warnings. Changed
runtime settings produce `unsupported` with no point estimate. Being inside
individual training ranges does not prove joint, industry, or customer support.
Input and output forecasts sum to the total-token forecast. The dollar forecast
is a separately selected regression, not a repricing of those token forecasts.

Both `calibrated_interval` and `upper_budget_bound` remain null. The calibration
status is `insufficient_independent_groups_shared_template`; no independent
calibration projects have been allocated. Training-CV residuals are not
calibration data. The historical holdout has already been examined, so this
comparison is retrospective rather than a new sealed evaluation.

Automatic quality failures retain their incurred costs. Contract-pass counts
are reported separately and do not establish human acceptance, cost until
success, or quality-preserving batching. The six RFPs remain scoping contexts,
not completed customer engagements.

### September 15 training result

The offline command completed and saved the
[trained models](../runs/20260915_customer_project_forecast_v1/models.json),
[project-level measurements](../runs/20260915_customer_project_forecast_v1/project_records.json),
[forecasts](../runs/20260915_customer_project_forecast_v1/predictions.json), and
[evaluation](../runs/20260915_customer_project_forecast_v1/analysis.json).
It made no new API calls and incurred no new API cost.

Training-only cross-validation selected `workflow` for input tokens and cost,
and `lego` for output tokens. The selected forms' historical holdout MAEs are
186.85 input tokens, 48.83 output tokens, and USD 0.0008869157 per complete
scoping execution.

| Cost predictor | Historical holdout MAE, USD |
|----------------|----------------------------|
| Constant | 0.0027128125 |
| Size | 0.0025055833 |
| Size plus units | 0.0012560179 |
| LEGO | 0.0009583653 |
| Workflow | 0.0008869157 |

These eight held-out executions come from only two independent projects.
Both projects exceed the training range for source-context bytes, so all eight
forecasts carry an extrapolation warning. The results do not establish
calibrated coverage or performance on arbitrary customer engagements.

Validation passed 153 focused customer-model/pilot tests. All twenty-four saved
forecasts reloaded without fitting, all 122 source-artifact hashes and five
training-code hashes matched, and the original call-level trained model
reproduced exactly using the shared utilities. Historical run files were not
modified.

## HTML explanation and new-source testing

Open the [model evidence report](customer-model-report.html) for the four
measured GPT tasks, input features, regression method, project split, error
metrics and additional public-source scoping examples.

An operation means a GPT task: Extract, Classify, Plan or Report. The four
boxes describing input preparation and measurement are experiment-building
steps, not these four tasks. An annotation is a requirement label or note;
"frozen" means kept unchanged for the comparison. The trained predictor is
a numerical regression, not a fine-tuned GPT model.

The [new source catalog](../experiments/customer_requests/prospective_catalog.json)
contains four held-out scoping inputs derived from three UNFPA requests and
one IPU request. It also preserves source sections, selection caveats and two
rejected examples. All complete customer engagements remain excluded:
none was verified as fully deliverable by the four-operation harness.
Four project IDs across two buyers do not establish four independent
organizations or industry-wide generalization.

Render the report offline with the existing reporting module:

```powershell
.\.venv\Scripts\python.exe -m token_yield.report `
  --model-run runs\20260915_customer_project_forecast_v1 `
  --new-catalog experiments\customer_requests\prospective_catalog.json `
  --prospective-run runs\20260915_customer_prospective_test_v1 `
  --output docs\customer-model-report.html
```

Omit `--prospective-run` when no evaluation has been executed. A missing or
unfinished new evaluation is not presented as measured accuracy. The
renderer checks the original model and source-file hashes, matches
observation IDs, and recalculates selected-model errors from saved labels.

The existing pilot CLI also supports a no-spend preview:

```powershell
.\.venv\Scripts\python.exe -m examples.customer_request_pilot `
  --test-model runs\20260915_customer_project_forecast_v1 `
  --catalog experiments\customer_requests\prospective_catalog.json `
  --run-dir runs\customer_prospective_preview
```

Adding `--execute` and using a fresh output directory makes real model calls.
This extension uses a USD 10 conservative safety ceiling and USD 9.50 stop,
no inference retries, and two repeats of each split/batched layout.
Do not treat those limits or an old configuration as authorization for
another paid campaign.

Before dispatch, the runner saves all predictions and fixes the model,
catalog, call plan and selected forms. It reuses the existing prompt
compiler, dispatcher, usage ledger and budget safeguards. Research metadata
is preserved separately from the strict prompt-input schema.

New input permissions and the revised test controller break exact identity
with the original runtime contract. The forecast API's `unsupported` state
is retained. The new measurement scores are therefore an explicit transfer
evaluation of conditional predictions, not newly approved production
quotes. No model is fitted or selected using these new outcomes, and no
calibrated prediction interval is claimed.

### September 15 new-source measurements

The [new test results](../runs/20260915_customer_prospective_test_v1/prospective_evaluation.json)
contain 40 measured calls aggregated into 16 complete executions across
four new scoping briefs. The selected formulas were unchanged:

| Target | Selected form | Average absolute error | Average percentage error |
|--------|---------------|------------------------|--------------------------|
| Input tokens | Workflow | 113.37 tokens | 10.39% |
| Output tokens | LEGO | 27.10 tokens | 3.73% |
| Retail-rate cost | Workflow | USD 0.0005800159 | 3.94% |

Metrics weight project groups equally. These are conditional transfer-test
results, not calibrated estimates for full customer engagements.
The [pre-dispatch predictions](../runs/20260915_customer_prospective_test_v1/predictions.json)
were saved before the first request. The
[project-level measurements](../runs/20260915_customer_prospective_test_v1/project_records.json)
contain 22,274 input and 11,486 output tokens, with explicitly zero cached
and reasoning tokens. Total calculated API cost was USD 0.227975;
conservative safety accounting settled USD 2.74268, below the USD 9.50 stop,
with no outstanding reservations.

All 40 calls and all 16 complete executions passed automatic contract
checks. This does not establish human acceptance or semantic quality.
All 16 forecasts retain the actual-runtime `unsupported` warning.

Verification matched all 80 new request/response files to their ledgers,
confirmed all 122 historical source files remained unchanged, and reloaded
every selected and comparator prediction without fitting. The original
model artifact remained byte-identical. Integration validation passed 66
targeted tests covering the customer pilot, HTML reporting and legacy report
behavior.

### Reading and sharing these results

The [short training explanation](lego-model-training.md) introduces the model
without requiring knowledge of statistics.

The published new-test files retain project-level measurements, predictions,
evaluation results and the budget ledger. The evaluation's local model path
is replaced with its run name; measurements, predictions and model bytes are
unchanged. Raw request/response files, cloud deployment audits, the execution
protocol and duplicate model/analysis snapshots remain outside this branch.
The standard-named new-test directories ignore those files by default.

The raw-file checks above describe verification performed on the complete
local run before publication. This smaller public bundle can reproduce the
report and prediction-error calculations, but cannot independently repeat
the new test's raw-response audit.
