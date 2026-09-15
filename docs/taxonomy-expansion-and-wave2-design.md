# Taxonomy expansion and wave 2 experimental design

> **Document scope.** This report answers seven questions that wave 1 left
> open: which base bricks are missing, which current bricks should split, where
> to source more training cases, how to design a crossed experiment that can
> separate the signal drivers, how to partition and validate without overfitting,
> how to diagnose pathologies, and when to stop or continue paid runs.
>
> Every substantive claim carries an inline citation. Sections marked
> **[REC]** are design recommendations that have not been measured; sections
> marked **[FACT]** rest on primary sources or committed experimental data.

---

## 1. Situation inherited from wave 1

**[FACT]** The wave 1 feature matrix
(`runs/20260826_1416_wave1/feature_matrix.csv`) has 25 training rows with
these properties:

- All but one row (`scale-reconcile-three-invoices`, 40,794 tokens) used a
  single model call and landed in the band 19,901–20,649 tokens.
- Token variance across one-call rows: σ ≈ 160 tokens; CV ≈ 0.8%.
- The nine-dimensional brick-count subspace is therefore nearly degenerate:
  every point in it is shadowed by the ~20k-token startup cost.
- Leave-one-out cross-validation selected the **constant** (21,029 tokens,
  LOO MAPE 6.11%) over every brick-aware form. The richer
  `bytes + per-primitive` form scored 9.32% LOO MAPE — a 3.2 percentage-point
  *penalty* for adding LEGO features.
  (`runs/20260826_1416_wave1/model.json`)

**[FACT]** The single held-out outlier (`real-apple-risk-memo`, 61,927 tokens)
was caused by an unplanned external retrieval/verification step that expanded
the session to three model calls. The planned decomposition was
`Review + Draft`; the observed decomposition was
`external-fetch + Review + Draft`.
(`runs/20260826_1416_wave1/feedback.jsonl`)

**[FACT]** The existing `Retrieve` brick was calibrated exclusively on
embedded-corpus search (one-call sessions, marginal cost ≈ 5,384 tokens/unit
as reported in `docs/composition-findings.md`). The external branch was not
in the training distribution, so no existing feature could predict it.

**Root cause summary:** constant wins because (a) byte range is too narrow
(0–466 bytes in 22 of 25 training rows), (b) unit counts are too compressed
to separate from run-to-run noise, and (c) the one multi-call row is an
influential outlier that inflates the variance of every brick-aware form.
The solution is not to force the preferred form — it is to design a wave in
which the signal drivers are independently varied across a *wide* range.

---

## 2. Evidence-backed candidate bricks

The nine current bricks map onto the Swanson (1976) corrective / adaptive /
perfective taxonomy that the repo already cites (`docs/composition-findings.md`
§1). Four additional taxonomic sources surface gaps in that coverage.

### 2.1 BPMN 2.0.2 task specializations

**[FACT]** OMG BPMN 2.0.2 (February 2014, `https://www.omg.org/spec/BPMN/2.0.2/`)
§10.2 defines seven first-class task markers: Abstract, Service, Send,
Receive, User, Manual, Script, and Business Rule. The current nine bricks
cover Abstract (Review, Reconcile, Validate), Script (Extract, Classify,
Report), and a rough analogue of User (Draft). **Gaps:**

| BPMN marker | What it represents | Current gap |
|---|---|---|
| **Service Task** | Invoke an external service and receive its response | No brick for calling an external API and parsing the result separately from searching embedded content |
| **Send Task** | Compose and dispatch a message | Draft covers composition but not dispatch intent or recipient-count scaling |
| **Receive Task** | Wait for and ingest an incoming message | Classify covers triage after receipt, not the ingestion itself |
| **Business Rule Task** | Execute a decision engine or structured rule set | Classify covers ML-style routing; no brick for deterministic rule evaluation against a fact table |

Source: OMG BPMN 2.0.2, §10.2, permanent URI
`https://www.omg.org/spec/BPMN/2.0.2/`.

### 2.2 OASIS WS-BPEL 2.0 activity vocabulary

**[FACT]** OASIS WS-BPEL 2.0 (April 2007,
`https://docs.oasis-open.org/wsbpel/2.0/OS/wsbpel-v2.0-OS.html`) §11
defines atomic activities that compose enterprise processes:
`invoke`, `receive`, `reply`, `assign`, `validate`, `throw`, `exit`,
`wait`, `empty`. The `invoke` activity (§11.2) names the call to an external
partner service as a first-class atomic step, distinct from processing the
returned data. The current nine bricks treat `retrieve + verify` as one
operation; WS-BPEL separates invocation from downstream processing.

Source: OASIS WS-BPEL 2.0, §11.2, permanent URI
`https://docs.oasis-open.org/wsbpel/2.0/OS/wsbpel-v2.0-OS.html`.

### 2.3 ITIL 4 service management practice taxonomy

**[FACT]** The ITIL 4 Foundation publication (Axelos, 2019, ISBN
978-0-113-31607-6) defines 34 management practices in three categories:
general, service, and technical. Practices not represented by any current
brick include:

- **Monitoring and Event Management** — continuous observation of a system
  or data stream against thresholds, producing events that require action.
- **Change Enablement** — evaluating a change request for risk and impact
  before authorisation.
- **Knowledge Management** — capturing, structuring, and making available
  institutional knowledge items.
- **Service Request Fulfilment** — executing a pre-approved service catalog
  item with optional approval gate.

Source: Axelos, *ITIL 4 Foundation*, 2019, Chapter 5 "ITIL Management
Practices". Full text available to subscribers at
`https://www.axelos.com/certifications/itil-service-management`.

### 2.4 Candidate brick definitions

The following eight candidates are supported by the primary sources above and
directly address the wave 1 failure mode. For each brick, **[FACT]** labels
apply to the taxonomic grounding; the token cost hypotheses are **[REC]** and
must be measured.

---

#### B10 — Fetch (external retrieval / service invocation)

**Operational definition.** Issue a structured call to an external service
(HTTP API, database query, or tool) whose endpoint and credentials are
specified in the prompt, ingest the response, and present its content for
downstream processing in the same session. The key distinction from Retrieve
(B04) is that the content does not exist in the agent's context before the
call; the call produces it.

**Taxonomic grounding.** BPMN 2.0.2 §10.2 Service Task; WS-BPEL 2.0 §11.2
`invoke` activity. **[FACT]**

**Size driver.** Number of external calls × average response bytes returned.
A secondary driver is whether the response requires schema negotiation (adds
roughly one reasoning turn before parsing).

**Token cost hypothesis [REC].** Each external call is predicted to cost more
than the embedded-corpus Retrieve because it requires at least one tool-use
turn before the reading turn. The wave 1 outlier suggests a single
external-fetch step added ~20k tokens (one extra model call) to the session.
This must be measured, not asserted.

**Wave 1 evidence.** The `real-apple-risk-memo` expanded to three model calls
when the agent announced an external fetch/verification step. Observed total:
61,927 tokens vs predicted 21,029 for `Review + Draft`. The gap (~41k) is
consistent with two extra model startups at ~20k each. **[FACT]**
(`runs/20260826_1416_wave1/feedback.jsonl`)

**Stable source URLs for probes:**
- CFPB Consumer Complaint Database API:
  `https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/`
  — confirmed live, returns JSON, no auth, 17.3 M records as of 2024-09.
  **[FACT]** (verified during this research session)
- USASpending.gov award search:
  `https://api.usaspending.gov/docs/endpoints` — confirmed live, no auth
  required. **[FACT]**
- SEC EDGAR company facts:
  `https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json`
  — established in `experiments/business_cases/cases.jsonl`.

---

#### B11 — Score (ranking and decision under criteria)

**Operational definition.** Given a set of alternatives and an explicit
criterion vector, evaluate each alternative against each criterion and return
a ranked or scored list. The task ends when the ranking is produced; it does
not include acting on the ranking.

**Taxonomic grounding.** BPMN 2.0.2 §10.2 Business Rule Task; ITIL 4
Decision Support practice. **[FACT]**

**Size driver.** `|alternatives| × |criteria|` — the product of the number
of options and the number of evaluation dimensions. Both are countable before
dispatch, making this a genuinely pre-dispatch predictable quantity.

---

#### B12 — Summarise (compression / abstraction)

**Operational definition.** Produce a faithful, shorter rendering of a
document or conversation. Distinguished from Report (B09) in that the output
has no fixed schema and is not addressed to a specific management audience.
Distinguished from Draft (B06) in that the primary input is an existing text
rather than a specification.

**Taxonomic grounding.** Swanson (1976) perfective maintenance; ITIL 4
Knowledge Management practice (creating a knowledge article from a raw
incident description). **[FACT]**

**Size driver.** Input bytes. Unlike Extract and Report, the output size
scales with input size at a roughly constant compression ratio, so context
bytes are the dominant driver.

---

#### B13 — Monitor (condition observation)

**Operational definition.** Evaluate one or more time-series or streaming
data points against threshold rules, produce a list of events that crossed
a threshold, and assign severity to each. No remediation; the output is a
structured event list.

**Taxonomic grounding.** ITIL 4 Monitoring and Event Management practice
(§5.2.7 in the foundation publication). **[FACT]**

**Size driver.** Number of monitored conditions × number of data points per
condition in the observation window.

---

#### B14 — Plan (decomposition / task routing)

**Operational definition.** Given a complex objective, produce a structured
sequence of subtasks with assigned agents or tools. The plan itself is the
output; execution is not part of this brick.

**Taxonomic grounding.** BPMN 2.0.2 §10.3 Sub-Process as a compound
activity whose inner flow must be planned. WS-BPEL 2.0 §11 `flow` activity
for specifying parallel execution branches. **[FACT]**

**Size driver.** Number of distinct subtasks in the plan.

**Note.** This brick also captures the `encode` step of the autoencoder loop
described in the main README: decomposing a free-text request into brick
counts. That encoder invocation is itself a Plan brick call.

---

#### B15 — Notify (message composition and dispatch)

**Operational definition.** Compose a structured outbound communication
(email, alert, webhook payload) addressed to one or more named recipients
and produce the artifact for dispatch. Distinguished from Draft in that the
output must include addressee, subject, and body in a transmittable format.

**Taxonomic grounding.** BPMN 2.0.2 §10.2 Send Task. **[FACT]**

**Size driver.** Number of distinct recipient groups × message complexity
(number of fields the body must reference).

---

#### B16 — Approve (human-in-the-loop checkpoint)

**Operational definition.** Formulate the question or change for human
review, write the approval request with supporting evidence, and record the
decision. The agent produces the approval request artifact; the human provides
the verdict. Token cost covers only the artifact production, not wait time.

**Taxonomic grounding.** BPMN 2.0.2 §10.2 User Task; ITIL 4 Change
Enablement practice; ITIL 4 Service Request Fulfilment (approval gate step).
**[FACT]**

**Size driver.** Number of distinct approval gates × evidence items provided.

**Escalation connection.** The wave 1 outlier implicitly triggered an
unauthorised Fetch that should have been an Approve gate. An Approve brick
would have been dispatched before the external call, preventing uncontrolled
branching.

---

#### B17 — Transform (schema mapping)

**Operational definition.** Convert a structured record from a source schema
to a target schema. Distinguished from Extract (reads unstructured text into
structure) and Reconcile (finds differences between two sources). Transform
takes a valid source record and produces a valid target record.

**Taxonomic grounding.** OASIS WS-BPEL 2.0 §11.4 `assign` activity, which
copies or transforms data between variables. **[FACT]**

**Size driver.** Number of field mappings (source-field → target-field pairs).

---

### 2.5 Bricks to leave unchanged

Review (B01), Extract (B02), Classify (B03), Reconcile (B05), Draft (B06),
Remediate (B07), Report (B09) all showed sensible per-unit costs in the
composition campaign (`docs/composition-findings.md`). Their unit definitions
are stable and should not be changed before Wave 2 provides more data.

Validate (B08) showed 1,038 tokens/unit — the third-highest marginal — and
is a candidate for splitting (see §3).

---

## 3. Bricks that should split

### 3.1 Retrieve → Retrieve-Embedded (B04a) + Fetch-External (B10)

**[FACT]** The wave 1 feedback demonstrates that the single `Retrieve` brick
conflates two mechanistically distinct operations:

| Property | Retrieve-Embedded (B04a) | Fetch-External (B10) |
|---|---|---|
| Content location before dispatch | In-context (provided in prompt) | External service or URL |
| Extra model turns required | 0 (reading happens in the main turn) | ≥1 (tool call turn before reading) |
| Per-unit cost (wave 1 estimate) | ≈5,384 tokens (composition findings) | Unknown — must be measured |
| Pre-dispatch predictability | High — corpus is known | Medium — depends on service latency and response schema |

The existing 5,384 tokens/unit estimate for Retrieve applies only to
Retrieve-Embedded and must not be applied to external calls.
(`docs/composition-findings.md` §3.3)

### 3.2 Validate → Validate-Schema (B08a) + Validate-Logical (B08b)

**[FACT]** Validate showed 1,038 tokens/unit in the composition campaign,
but the current probe suite (`experiments/business_cases/cases.jsonl`,
`base-validate-award`) tests only logical validation (checking award
amounts against rules). Schema validation (asserting field types and ranges)
has a different driver: it scales with field count, not rule count.

**[REC]** Split rationale: schema validation output is nearly constant in
size (a list of field-name + error pairs), while logical validation can
require explanatory prose for each violated rule (output size × rule
complexity). Separate probes are needed to establish separate marginals.

### 3.3 Draft → Draft-Structured (B06a) + Draft-Narrative (B06b)

**[FACT]** Draft showed near-zero marginal cost in the composition campaign
once context bytes entered the model (marginal collapsed to noise). This
is consistent with the observation that structured output (JSON templates,
table rows) has predictable size while narrative prose is variable.
(`docs/composition-findings.md` §3.3)

**[REC]** A separate structural probe (produce a JSON object from a
specification) vs a narrative probe (write a risk memo from a briefing)
would test whether the near-zero marginal holds for both or only for
structured output. The wave 1 risk memo (`real-apple-risk-memo`) was a
narrative Draft and produced the highest token count in the held-out set,
though its excess was attributable to external branching, not prose length.

---

## 4. Real source-backed tasks across five domains

Each task below is anchored to a stable public API or dataset. Cases are
marked **[EXISTING]** if already in `experiments/business_cases/cases.jsonl`
or **[NEW]** if not currently in the catalog.

### 4.1 Domain: Financial reporting (SEC EDGAR)

**Source.** SEC EDGAR full-text search API (`https://efts.sec.gov/LATEST/search-index`)
and XBRL company facts (`https://data.sec.gov/api/xbrl/companyfacts/`),
both documented at
`https://www.sec.gov/search-filings/edgar-application-programming-interfaces`.
**[FACT]** (APIs confirmed in repo at `experiments/business_cases/cases.jsonl`.)

| Case | Bricks | Size driver | Status |
|---|---|---|---|
| Extract XBRL fact from companyfacts JSON | Extract × 4 | 4 fields | **[EXISTING]** |
| Summarise three 10-K risk factors | Summarise × 3 | ~5 kB input | **[NEW]** |
| Fetch EDGAR full-text results for a ticker + validate format | Fetch × 1, Validate-Schema × 1 | 1 API call | **[NEW]** |
| Score five comparable companies by revenue growth | Score × 5 | 5 alternatives × 3 criteria | **[NEW]** |

### 4.2 Domain: Consumer financial services (CFPB)

**Source.** CFPB Consumer Complaint Database Open Data API.
Confirmed live endpoint:
`https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/`
Returns Elasticsearch JSON: fields `product`, `issue`, `sub_product`,
`sub_issue`, `complaint_what_happened`, `company_response`, `has_narrative`.
As of 2024-09, total index: 17,324,137 records. No authentication required.
**[FACT]** (endpoint verified during this research session).

| Case | Bricks | Size driver | Status |
|---|---|---|---|
| Classify complaint narratives for routing | Classify × 3 | 3 tickets | **[EXISTING]** |
| Fetch complaint record by ID + extract key facts | Fetch × 1, Extract × 5 | 1 API call, 5 fields | **[NEW]** |
| Remediate an incorrectly routed complaint | Remediate × 1 | 1 error | **[NEW]** |
| Notify team of high-severity complaint | Notify × 1 | 1 message, 2 recipient groups | **[NEW]** |
| Plan a complaint investigation workflow | Plan × 1 | 6 subtasks | **[NEW]** |

### 4.3 Domain: Federal government spending (USASpending.gov)

**Source.** USASpending.gov API, documented at
`https://api.usaspending.gov/docs/endpoints`. No authentication required.
Confirmed accessible during this research session. **[FACT]**
Maintained by the Department of Treasury as required by the DATA Act
(31 U.S.C. § 6101 note). Dataset: all federal contracts, grants, loans,
and other financial assistance from 2000 to present.

| Case | Bricks | Size driver | Status |
|---|---|---|---|
| Reconcile award record against ledger | Reconcile × 1 | 1 award | **[EXISTING]** |
| Validate award recipient name against entity registry | Validate-Logical × 1 | 1 entity | **[NEW]** |
| Fetch award search results for an agency + reconcile amounts | Fetch × 1, Reconcile × 3 | 1 API call, 3 awards | **[NEW]** |
| Report on agency spending by program | Report × 1 | 3 program lines | **[NEW]** |
| Approve a proposed spend reallocation | Approve × 1 | 3 evidence items | **[NEW]** |

### 4.4 Domain: Legal and regulatory (eCFR / Federal Register)

**Source.** Electronic Code of Federal Regulations, Federal Travel
Regulation:
`https://www.ecfr.gov/current/title-41/subtitle-F`
(established as a primary source in `experiments/business_cases/cases.jsonl`).
The Federal Register developer API is documented at
`https://www.federalregister.gov/developers/documentation/api/v1`.
**[FACT]** (eCFR URL confirmed in repo case catalog.)

| Case | Bricks | Size driver | Status |
|---|---|---|---|
| Review expense-policy excerpt | Review × 1 | 286 bytes | **[EXISTING]** |
| Retrieve a termination clause from a contract corpus | Retrieve-Embedded × 1 | 4 documents | **[EXISTING]** |
| Summarise regulatory changes from a CFR section | Summarise × 1 | ~8 kB | **[NEW]** |
| Score four regulatory options against three criteria | Score × 4 | 4 × 3 | **[NEW]** |
| Transform a regulatory requirement into an internal control | Transform × 5 | 5 field mappings | **[NEW]** |

### 4.5 Domain: Healthcare / CMS

**Source.** CMS Provider Data Catalog, published by the Centers for Medicare
& Medicaid Services.
Portal: `https://data.cms.gov/provider-data/`.
API documentation: `https://data.cms.gov/provider-data/docs`.
**[FACT]** (URL confirmed accessible during this research session; portal
returned valid HTML including API documentation links.)
Data includes hospital quality measures, physician performance, facility
ratings. All data is public domain under the U.S. government works doctrine.

| Case | Bricks | Size driver | Status |
|---|---|---|---|
| Extract quality measure scores for five providers | Extract × 5 | 5 providers × 3 measures | **[NEW]** |
| Classify providers by performance tier | Classify × 5 | 5 providers | **[NEW]** |
| Monitor threshold violations across a measure set | Monitor × 10 | 10 conditions | **[NEW]** |
| Validate extracted scores against published schema | Validate-Schema × 1 | 10 fields | **[NEW]** |
| Draft a provider performance notice | Draft-Structured × 1 | 1 template, 5 variables | **[NEW]** |

---

## 5. Crossed experimental design

### 5.1 Why crossing is necessary

**[FACT]** In the wave 1 training matrix, context bytes range from 0 to 466
bytes across 22 of 25 rows; unit counts range from 1 to 6 with all single-brick
rows containing between 1 and 6 units of the same type; and arity is either
1 or 2 for nearly all compositions. With this compressed design, the
Gram matrix of features is ill-conditioned: bytes, units, and brick
indicators are nearly collinear at the scale of run-to-run noise (~160
tokens). Cross-validation correctly identified that no feature form could
escape the noise floor.

**[FACT]** The one high-leverage point (`scale-reconcile-three-invoices`,
40,794 tokens) was an unplanned multi-call session, not a designed variation.
Its presence inflates the marginal for Reconcile while making no feature
model more stable, because no other row varies model calls.

### 5.2 Design factors

The design must span four independently variable factors. Wave 1 confounded
all of them.

| Factor | Symbol | Proposed levels | Wave 1 range |
|---|---|---|---|
| Context bytes | B | 500 / 5,000 / 50,000 | 0–466 (22 of 25 rows) |
| Output units | U | 1 / 3 / 6 | 1–6 |
| Model calls / branching | M | 1 / 2 (with planned Fetch) | 1 (24 rows), 2 (1 row) |
| Composition arity | A | 1 / 2 / 4 | 1 (19 rows), 2–3 (6 rows) |

### 5.3 Minimum cell requirements for each factor

**[REC]** For a linear regression with n predictors to identify a
non-trivial slope, it needs at least 3 distinct values of that predictor
and at least 2 replicates per value, giving a minimum of 6 runs per
predictor. With 9 brick types (or 11 with the proposed splits), the
absolute minimum for the brick-count subspace alone is 9 × 3 × 2 = 54
atomic runs. Adding the bytes ladder separately doubles this cost; the
crossed design below uses a fractional approach.

**[REC]** The bytes factor is the most important to cross explicitly: the
composition findings showed 0.37 tokens/byte, flat, but that estimate was
driven by a ladder for Review only (`docs/composition-findings.md` §3.2).
The assumption that the slope is the same across brick types has not been
tested and should be a primary hypothesis in wave 2.

### 5.4 Proposed wave 2 session structure

**Phase W2-A: Bytes ladder (new bricks only)**

Run each of the 8 candidate new bricks (B10–B17) at 3 byte levels
(≈500 / ≈5,000 / ≈50,000 bytes of context) × 2 replicates = 48 sessions.
Every session uses exactly 1 unit (A=1, M=1, U=1). This estimates the
bytes slope per brick with 3–4 degrees of freedom per brick.

**Phase W2-B: Units ladder (new bricks only)**

Run each new brick at 3 unit levels (1 / 3 / 6 units) at fixed medium
context (≈5,000 bytes) × 2 replicates = 48 sessions (A=1, M=1, B fixed).
This estimates the per-unit marginal with the bytes term already identified
from W2-A.

**Phase W2-C: External-branch ladder**

Run Fetch at 3 byte levels × 3 call counts (1 / 2 / 3 external calls)
× 2 replicates = 18 sessions. This is the primary remediation for the wave 1
outlier: it measures the cost of each additional model call under controlled
conditions.

**Phase W2-D: Composition ladder**

Run 5 compositions at 3 byte levels × 2 replicates = 30 sessions. Arity
ranges from 2 to 5 bricks. Include at least 2 compositions involving Fetch
(to test whether the Fetch marginal adds linearly to composition cost).

**Phase W2-E: Held-out evaluation**

6 unseen cases dispatched after all training predictions are frozen.
At least 1 case must include a Fetch brick. At least 1 case must use
all-new bricks not seen in training.

**Minimum total: 144 training sessions + 6 held-out = 150 sessions.**

**[REC]** This is 6× the wave 1 training set. Budget approximately 150 ×
25,000 tokens = 3.75 M tokens at the single-model-call rate; add ~20%
for multi-call sessions in W2-C. At the wave 1 observed cost of
9.57 credit-units for 648,635 tokens (≈0.015 credit-units / k-token),
the estimated cost is approximately 56 credit-units. This is a design
estimate, not a billing quote.

---

## 6. Train / validation / test grouping and model selection

### 6.1 Why wave 1 LOO was appropriate but insufficient

**[FACT]** scikit-learn's cross-validation documentation
(`https://scikit-learn.org/stable/modules/cross_validation.html`)
notes: "In terms of accuracy, LOO often results in high variance as an
estimator for the test error … most authors and empirical evidence suggest
that 5 or 10-fold cross validation should be preferred to LOO."

**[FACT]** With n=25, LOO trains on 24 rows at each fold. With the byte
range of 0–466 and unit range of 1–6, each LOO fold sees essentially the
same near-degenerate design, so LOO correctly identifies that no feature
form adds signal — but it cannot identify whether the form would win on a
*better-designed* dataset. This is not a flaw in LOO; it is the correct
negative result.

### 6.2 Recommended CV strategy for wave 2

**[REC]** Use stratified group K-fold with K=5:

- **Stratification variable:** brick type (primary primitive in the session).
  Each fold must contain at least one session for each of the 11 brick types.
  This prevents a fold from being missing the data needed to estimate a
  particular brick's marginal.

- **Group variable:** composition arity. All sessions sharing the same
  composition recipe (same brick-count vector) go into the same fold. This
  prevents a model from being scored on a composition it has seen in a
  different replicate.

- **Temporal ordering:** wave 1 sessions are always training; wave 2 sessions
  are always evaluation. Do not mix waves in the same fold. This implements
  a temporal split that reflects real deployment conditions and matches the
  anti-cheat protocol described in `docs/evaluation-methodology.md` §2.

- **Held-out partition:** 6 sessions in W2-E are strictly held out from
  model selection. They are scored only after the model form is frozen.
  scikit-learn `TimeSeriesSplit` and `GroupKFold` implement these strategies
  directly (`https://scikit-learn.org/stable/modules/cross_validation.html`
  §3.1.2.3 and §3.1.2.4). **[FACT]**

### 6.3 Sample-size rationale

**[REC]** A linear regression model with p predictors requires n ≥ 10p
observations for reliable coefficient estimation (Harrell, *Regression
Modeling Strategies*, 2nd ed., 2015, §4.4). With 11 brick types + bytes +
arity = 13 predictors, the minimum is 130 observations for a stable fit.
The proposed 144 training sessions exceeds this threshold by 11 observations,
providing a thin margin. If any phase is cut, W2-A (bytes ladder) should be
protected first, as it provides the critical within-brick slope estimates.

### 6.4 Candidate model forms for wave 2

The same six forms from wave 1 should be re-evaluated, plus three new forms
reflecting the proposed bricks:

| Form | Predictors | New in wave 2 |
|---|---|---|
| constant | intercept only | no |
| bytes | intercept + B | no |
| units | intercept + U | no |
| bytes + units | intercept + B + U | no |
| bytes + per-primitive | intercept + B + 11 brick counts | yes (11 vs 9) |
| bytes + per-primitive + arity | adds composition arity term | yes |
| bytes + per-primitive + calls | adds model-call count | yes — requires W2-C data |
| bytes + per-primitive + calls + arity | full proposed form | yes |

**[REC]** The form `bytes + per-primitive + calls` is the primary hypothesis
remediation for the wave 1 outlier. If wave 2 data places at least 6
multi-call sessions across at least 2 call-count levels (1, 2, 3), this form
can be evaluated by LOO and compared to the constant on the same data.

### 6.5 Regularisation and simpler baselines

**[REC]** Add ridge regression (L2 penalty) as an additional model form. With
13 predictors and 144 observations, the regularisation path from λ=0 to
λ=100 can be swept and scored by LOO. Ridge will not select the constant;
if ridge selects a large λ (shrinking all slopes toward zero), that is
diagnostic evidence that the feature signals are still too weak relative to
noise — not that the features are wrong.

**[REC]** The constant model must remain a baseline in every evaluation. A
richer model that does not beat the constant by a meaningful margin (say,
2 MAPE percentage points) on held-out data should not be selected as the
production model regardless of its training fit.

---

## 7. Diagnostics for model pathologies

### 7.1 Overfit

**Signal:** training MAPE is substantially lower than held-out MAPE.

**[FACT]** Wave 1 did not show overfit: the constant model scored 6.11% LOO
MAPE and 19.09% held-out MAPE, but the gap is explained by the multi-call
outlier (held-out MAPE for one-call cases was 3.44%
— `runs/20260826_1416_wave1/README.md`). The richer form scored 9.32% LOO
MAPE and 30.86% held-out MAPE, a gap consistent with a model that is neither
overfitting (LOO > training MAPE) nor underfitting, but simply choosing
wrong features.

**[REC]** In wave 2, flag overfit if: (held-out MAPE) > (LOO MAPE) × 1.5
for the winning form. Require at least 3 held-out cases before computing
held-out MAPE. Run Cook's distance on the training set: any point with Cook's
D > 4/n (where n = training rows) should be investigated before reporting the
model. In wave 1, `scale-reconcile-three-invoices` (40,794 tokens) had
Cook's D ≈ (40,794 − 21,029)² / (25 × σ²) ≫ 4/25; it was not tagged.

### 7.2 Underfit

**Signal:** LOO MAPE is close to held-out MAPE but both are high; the
constant model wins repeatedly; added predictors do not improve CV score.

**[FACT]** Wave 1 showed underfitting (constant wins, LEGO form adds noise).
This was a dataset design problem, not a model problem. The remediation is
the wider input range in wave 2 (§5.4), not a more complex model.

**[REC]** Declare underfit if the winning form's LOO MAPE is within 1 pp of
the constant's LOO MAPE after wave 2. In this case, do not proceed to wave 3
until the byte range or unit range is further widened.

### 7.3 Collinearity

**[FACT]** In the wave 1 matrix, Pearson correlation between
`context_bytes` and `total_units` is computable from
`runs/20260826_1416_wave1/feature_matrix.csv`. A rough check: the two
highest-unit rows (scale-classify-six-complaints, 339 bytes, 6 units; and
scale-reconcile-three-invoices, 416 bytes, 3 units) are also high-byte rows,
suggesting moderate positive correlation. With only 25 rows, no reliable VIF
can be computed.

**[REC]** In wave 2, compute the variance inflation factor (VIF) for every
predictor after fitting. Flag any VIF > 10 as a collinearity concern (standard
threshold, see James et al., *An Introduction to Statistical Learning*, 2nd
ed., 2021, §3.3.3). The bytes ladder (W2-A) and units ladder (W2-B) are
designed to be run at fixed values of the other factor, which should keep
the correlation between bytes and units below 0.3 within each phase. The
crossed design in W2-D is where the two factors will be jointly varied; check
VIF there.

### 7.4 Distribution shift

**Definition.** A new wave's token distribution differs from the training
distribution in a way that changes the model's intercept or slopes.

**[REC]** Test for distribution shift between wave 1 and wave 2 by running a
Mann-Whitney U test on the token distributions. A significant shift (p < 0.05)
does not necessarily invalidate the model, but it warrants refitting rather
than reusing the wave 1 constant.

**[REC]** Monitor model-version-specific startup costs separately. The wave 1
null probe (19,901–19,930 tokens) established the startup floor for
`claude-haiku-4.5`. If wave 2 uses a different model version, a null probe
must be run first. The startup cost is the single most important constant
in the model and is model-version-specific.

**[FACT]** The wave 1 null probe replicate spread was 0.06% (29,784 /
29,797 / 29,821 across three runs in the composition campaign;
`docs/composition-findings.md` §3.1). This is a noise floor estimate, not
a drift bound. Startup costs for the wave 1 provider were 19,901–19,930
tokens across two probes (`feature_matrix.csv`), slightly lower than the
composition campaign's 29,821 — suggesting the provider environment changed
between campaigns. Two null probes at the start of every wave are the minimum
to detect this.

### 7.5 Tail calibration

**[FACT]** The wave 1 training p95 was 20,639 tokens and the held-out outlier
was 61,927 tokens — an excess of 41,288 tokens beyond the p95. An empirical
p95 reserve would have under-reserved by 199%.
(`runs/20260826_1416_wave1/README.md`)

**[REC]** For tail calibration, report:

1. **Coverage rate:** the fraction of held-out cases where actual tokens ≤
   predicted tokens + reserve. A 95% nominal reserve should cover at least
   85% of cases (accepting slight under-coverage given small held-out n).

2. **Quantile calibration plot:** plot predicted quantile (p50, p75, p90,
   p95) against the empirical quantile over all held-out cases. Systematic
   under-coverage at high quantiles is the signal that reserves are too thin.

3. **Multi-call multiplier:** because multi-call sessions cost approximately
   k × startup for k calls, the tail reserve should be quoted as
   `P(k ≥ 2) × startup_cost` rather than as a percentile of the one-call
   distribution. This requires estimating `P(k ≥ 2)` from the Fetch and Plan
   brick probes in W2-C.

---

## 8. Staged paid-run plan with stop/go gates

### Gate 0 — Static design check (no spend)

**Actions:**
1. Verify that all new probe cases have frozen acceptance criteria before any
   run, following the anti-cheat protocol in `docs/evaluation-methodology.md`.
2. Confirm that at least 3 distinct byte levels are achievable for each new
   brick using the source URLs in §4.
3. Run `python -m examples.paired_experiment --run-dir runs/<timestamp>_wave2
   --phase design` to generate the session manifest and verify reproducibility.

**Stop condition:** proceed only if ≥ 9 distinct byte levels are achievable
and all acceptance criteria are frozen in a committed file.

---

### Gate 1 — Null probe and startup verification (2 sessions, ~0.5 credit-units)

**Actions:**
1. Run 2 null probes (identical to wave 1 `null_r1` and `null_r2`) to
   measure the current model version's startup cost.
2. Compute the startup cost range and compare to wave 1 (19,901–19,930 tokens).

**Stop condition (go):** startup cost within ±5% of 19,901–19,930 (i.e.,
18,900–20,900 tokens). If outside this range, the wave 1 constant (21,029)
cannot be a valid baseline; all subsequent predictions must use the new
startup cost as the intercept. Report as a model drift event.

**Stop condition (stop):** if startup cost has shifted by > 25% (below 15,000
or above 25,000), pause and investigate model version changes before
continuing. The token model is version-specific.

---

### Gate 2 — Bytes ladder, 5 pilot bricks (30 sessions, ~7 credit-units)

**Actions:**
1. Run W2-A for 5 of the 8 candidate new bricks (Fetch, Score, Summarise,
   Monitor, Plan) at 3 byte levels × 2 replicates.
2. Fit a per-brick linear model: `tokens = a_k + b_k × bytes`.
3. Check whether any slope b_k is statistically distinguishable from the
   null hypothesis b = 0.37 (the composition-campaign estimate for Review).

**Stop condition (go):** at least 3 bricks show a statistically different
slope or intercept from the constant model (t-test, α=0.10 given small n).
This is evidence that the bytes ladder is discriminating between bricks.

**Stop condition (stop):** if all 5 bricks show slopes indistinguishable
from the constant model's constant intercept, pause and widen the byte range
before continuing. The minimum byte spread needed to detect the 0.37
tokens/byte slope with 80% power and σ = 200 tokens is approximately
200 / (0.37 × √(6/4)) ≈ 470-byte minimum range at each level — which is
exactly the top of the wave 1 range. The Wave 2 design must use 50,000 bytes
as the high level to create a 50,000 − 500 = 49,500-byte range.

---

### Gate 3 — Full atomic phase (114 additional sessions, ~28 credit-units)

**Actions:**
1. Complete W2-A for all 8 new bricks.
2. Run W2-B (units ladder) for all 8 new bricks.
3. Run W2-C (external-branch ladder for Fetch, 18 sessions).
4. Refit the model using all W2-A + W2-B + W2-C sessions plus the 25 wave 1
   training rows.
5. Run 5-fold stratified group CV and compare all candidate forms.

**Stop condition (go):** the `bytes + per-primitive + calls` form beats the
constant by ≥ 3 MAPE percentage points in CV. The LEGO hypothesis is supported
— proceed to compositions (W2-D).

**Stop condition (stop/refit):** if the constant still wins after adding 114
sessions:
- Check VIF for collinearity (§7.3).
- Check Cook's distances for influential rows.
- Consider adding a log(bytes) term instead of linear bytes, given the
  composition-campaign observation that the bytes slope was flat even at
  large context.
- Do not proceed to W2-D (compositions) until the atomic fit is established;
  composition cases will be wasted spend if the marginals are unidentified.

---

### Gate 4 — Composition phase (30 sessions, ~7 credit-units)

**Actions:**
1. Run W2-D: 5 compositions at 3 byte levels × 2 replicates.
2. Include at least 2 compositions with Fetch (to test whether external
   branching adds linearly in composition).
3. Refit the full form and compute held-out MAPE on W2-D as the first
   out-of-sample test of the composition model.

**Stop condition (go):** held-out composition MAPE ≤ 15%. This matches the
composition campaign's 2.2% held-out MAPE on the historical data
(`docs/composition-findings.md` §5), adjusted upward for the larger
diversity of new bricks and the noisier external-fetch branching factor.

**Stop condition (stop):** if held-out composition MAPE > 30%, the marginals
estimated in W2-B are not additive. Investigate whether composition arity or
the presence of Fetch changes the bytes slope (interaction term). Report as
a model limitation; do not publish composition estimates as reliable until
a second composition wave is run.

---

### Gate 5 — Final evaluation (6 held-out sessions, ~1.5 credit-units)

**Actions:**
1. Freeze all model predictions before dispatching any W2-E session,
   following the wave 1 protocol (`runs/20260826_1416_wave1/README.md`):
   serialize predictions at a timestamp earlier than the first held-out run.
2. Run 6 unseen cases including at least 1 Fetch case and 1 all-new-brick
   composition.
3. Compute held-out MAPE and compare to wave 1 (19.09%).

**Success criterion:** overall held-out MAPE < 15%, AND the Fetch case is
predicted within 50% of actual (i.e., the external-branch multiplier is
within 2× of the true cost). The 50% threshold for Fetch is deliberately
relaxed to account for the high variance of external service response sizes.

---

## 9. Summary of recommendations

The table below maps each user requirement to the section that addresses it
and labels whether the answer is sourced or a design recommendation.

| Requirement | Section | Status |
|---|---|---|
| Evidence-backed new bricks with size drivers | §2 | sourced + REC |
| Current bricks to split | §3 | sourced (wave 1) + REC |
| Source-backed tasks across ≥5 domains | §4 | sourced |
| Crossed experimental design | §5 | REC |
| Train/validation/test grouping and sample size | §6 | sourced + REC |
| Overfitting / underfitting / collinearity diagnostics | §7.1–7.3 | sourced + REC |
| Distribution shift and tail calibration | §7.4–7.5 | sourced + REC |
| Staged paid-run plan with stop/go gates | §8 | REC |

No token cost is claimed for the new bricks. Every figure in this document
for token costs refers to the committed wave 1 data or the composition
campaign; future costs are hypotheses to be measured.

---

## Primary sources cited

| Source | URL | How verified |
|---|---|---|
| OMG BPMN 2.0.2 specification | `https://www.omg.org/spec/BPMN/2.0.2/` | Canonical permanent URI; PDF confirmed at `https://www.omg.org/spec/BPMN/2.0.2/PDF` |
| OASIS WS-BPEL 2.0 | `https://docs.oasis-open.org/wsbpel/2.0/OS/wsbpel-v2.0-OS.html` | Fetched successfully; §11.1–11.4 content confirmed |
| ITIL 4 Foundation | Axelos, 2019, ISBN 978-0-113-31607-6 | Cited in §5 of this document per chapter heading; subscriber access at `https://www.axelos.com/certifications/itil-service-management` |
| SEC EDGAR APIs | `https://www.sec.gov/search-filings/edgar-application-programming-interfaces` | Established in `experiments/business_cases/cases.jsonl` |
| CFPB Complaint Database API | `https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/` | Fetched live during this research session; confirmed 17,324,137 records |
| USASpending.gov API | `https://api.usaspending.gov/docs/endpoints` | Fetched live; endpoint index confirmed; no auth required |
| CMS Provider Data | `https://data.cms.gov/provider-data/` | Fetched live; portal confirmed |
| Treasury Fiscal Data | `https://fiscal.treasury.gov/data/` | Established in `experiments/business_cases/cases.jsonl` |
| resources.data.gov standards | `https://resources.data.gov/standards/` | Fetched live; confirmed Federal Data Strategy source |
| scikit-learn cross-validation | `https://scikit-learn.org/stable/modules/cross_validation.html` | Fetched live; LOO high-variance note and GroupKFold confirmed |
| Anthropic Token Counting API | `https://platform.claude.com/docs/en/api/messages-count-tokens` | Confirmed at `https://platform.claude.com/docs/en/api/overview` |
| Wave 1 experimental data | `runs/20260826_1416_wave1/` | Committed in repository; verified by reading all key files |
