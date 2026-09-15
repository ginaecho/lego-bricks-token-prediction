## Wave-3 taxonomy and industry design

**Scope.** This document proposes a preregistered wave-3 taxonomy expansion and experimental design for `ginaecho/lego-bricks-token-prediction`. It uses only primary sources or committed repository artifacts. Sentences marked **[FACT]** report verified source material; sentences marked **[REC]** are design recommendations motivated by those sources and by the recorded wave-2 pilot outcome (`docs/foundry-wave2-pilot.md`, `runs/20260826_1627_wave2/README.md`, `runs/20260826_1627_wave2/analysis.json`).

**Inherited constraints.** **[FACT]** Wave 2 settled USD 5.05074 under the conservative safety ledger (not reconciled provider billing), recorded one incomplete 100 KB Summarise attempt with missing usage telemetry, accepted only 29/30 attempts after semantic regrade, left Fetch within-shape MAD unidentified because each arm/shape had only one row, omitted live Fetch response-size as a pre-dispatch feature, made model-call and tool-call targets mechanically easy because the live branch was declared before dispatch, under-covered the 90% tail, and nevertheless produced a grouped ridge model with 69.1% grouped-CV MAE improvement, 95% bootstrap CI 48.7% to 89.5%, matrix rank 6/6, and condition number 2.88 (`docs/foundry-wave2-pilot.md`; `runs/20260826_1627_wave2/README.md`; `runs/20260826_1627_wave2/analysis.json`).

## Part 1. Candidate wave-3 bricks

### Selection rule

**[REC]** A new brick should be added only when the task has a distinct start/end boundary, a pre-dispatch size driver, and an acceptance oracle that can be checked without rerunning the work. **[REC]** Under that rule, the strongest additions are `evaluate`, `triage`, `correlate`, `attest`, `dispatch`, `merge`, `contain`, and `provision`; `settle` remains a plausible finance-specific addition but should not enter paid testing until an ISO 20022 snapshot is frozen from the owning catalogue (`https://www.omg.org/spec/DMN/1.5/`, `https://hl7.org/fhir/task.html`, `https://docs.oasis-open.org/ebxml-bp/2.0.4/ebbp-2.0.4.xsd`, `https://www.w3.org/TR/prov-dm/`, `https://docs.oasis-open.org/tosca/TOSCA/v2.0/os/TOSCA-v2.0-os.html`, `https://www.iso20022.org/catalogue-messages`).

### `evaluate`

**[FACT]** DMN 1.5 exists to standardize business decision design and implementation rather than only taxonomy assignment, validation, or numeric scoring (`https://www.omg.org/spec/DMN/1.5/`). **[REC]** Define `evaluate` as: apply a frozen decision table, FEEL expression, or equivalent rule set to a typed input tuple and return the computed output tuple. **[REC]** It starts when the input facts and rule artifact are fixed and ends when the output fields are emitted; it excludes downstream approval or notification.

**[REC] Boundary from existing bricks.** Not `classify`, because output need not be a label; not `score`, because output need not be scalar; not `validate`, because the purpose is deriving new values rather than checking conformance.

**[REC] Pre-dispatch size driver.** Count `decision_rows × input_fields × output_fields`, plus a binary `multi_output` flag.

**[REC] Likely call/tool branching.** Usually 1 model call, 0 tools for snapshot arms; 2 calls only when the rule artifact is fetched live.

**[REC] Deterministic acceptance oracle.** Output JSON validates against a frozen schema; every required output key is present; every enumerated field is in the allowed set; every numeric field parses; if the rule artifact encodes a closed output vocabulary, no out-of-vocabulary value is allowed.

**[REC] Task templates.**
- Financial services: evaluate whether an SEC filer breaches a simple leverage or liquidity threshold using `https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json`, with required output keys such as `passes_rule`, `rule_id`, and `supporting_facts` (`https://www.sec.gov/search-filings/edgar-application-programming-interfaces`).
- Insurance: evaluate whether a claim packet is complete enough for straight-through processing using ACORD data-standard field requirements and NAIC reporting expectations (`https://www.acord.org/standards-architecture/acord-data-standards`, `https://content.naic.org/sites/default/files/publications-sta-zu-statistical-handbook.pdf`).
- Legal/regulatory: evaluate whether a Federal Register document opens a comment window by checking `type`, `publication_date`, and `comments_close_on` from `https://www.federalregister.gov/api/v1/documents/2026-00001.json` (`https://www.federalregister.gov/developers/documentation/api/v1`).

### `triage`

**[FACT]** NIST SP 800-61 Rev.2 treats incident handling as analysis followed by choice of response, and FHIR Task explicitly models claimed, accepted, rejected, ready, on-hold, failed, and completed states for work items (`https://csrc.nist.gov/pubs/sp/800/61/r2/final`, `https://hl7.org/fhir/task.html`, `https://hl7.org/fhir/valueset-task-status.html`). **[REC]** Define `triage` as: rank or bucket inbound items by urgency and assign each item to the next queue. **[REC]** It starts with an unordered intake set and ends when every item has both a priority and a route.

**[REC] Boundary from existing bricks.** Not `monitor`, because it acts on a bounded intake rather than watching a stream indefinitely; not `classify`, because it also chooses queue and urgency; not `score`, because the outcome is operational routing, not only a number.

**[REC] Pre-dispatch size driver.** Count `items × triage_fields × routing_options`.

**[REC] Likely call/tool branching.** Usually 1 model call, 0 tools on frozen inputs; 2 calls when the intake is first fetched from an API.

**[REC] Deterministic acceptance oracle.** Every input item appears exactly once in output; every output row has `item_id`, `priority`, and `route`; `priority` matches a frozen enum; `route` matches an allowed queue identifier.

**[REC] Task templates.**
- Customer operations: triage one page of CFPB complaints from `https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?size=1&from=0` into queues such as billing, fraud, credit reporting, or servicing, using `product`, `issue`, and `submitted_via` (`https://www.consumerfinance.gov/data-research/consumer-complaints/`).
- Healthcare: triage FHIR Tasks by state and readiness using the 12-value task-status code system and Task queue semantics (`https://hl7.org/fhir/valueset-task-status.html`, `https://hl7.org/fhir/task.html`).
- Cybersecurity: triage CISA KEV entries by due date and ransomware-use flag using `cveID`, `dateAdded`, `dueDate`, and `knownRansomwareCampaignUse` from `https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json` (`https://www.cisa.gov/known-exploited-vulnerabilities-catalog`).

### `correlate`

**[FACT]** ebXML BPSS exposes explicit business transactions, collaborations, signals, and timing/non-repudiation attributes, and STIX 2.1 exists to express threat and observable information plus the overall structure that links those objects (`https://docs.oasis-open.org/ebxml-bp/2.0.4/ebbp-2.0.4.xsd`, `https://docs.oasis-open.org/cti/stix/v2.1/os/stix-v2.1-os.html`). **[REC]** Define `correlate` as: connect multiple records, messages, or observations to the same case, incident, or transaction by matching on frozen keys. **[REC]** It starts with at least two candidate records and ends when the output groups and residual ungrouped records are emitted.

**[REC] Boundary from existing bricks.** Not `reconcile`, because correlation links records before any discrepancy resolution; not `extract`, because the work is across records; not `retrieve-embedded`, because the matching rule is explicit rather than vector similarity.

**[REC] Pre-dispatch size driver.** Count `records × key_fields × pair_candidates`; for fetch arms also record `declared_fetch_bytes_kib`.

**[REC] Likely call/tool branching.** 1/2/3 model-call branches are natural here: one snapshot record set already present; one live fetch plus join; or two live fetches plus join.

**[REC] Deterministic acceptance oracle.** Output JSON validates; every input record appears in exactly one output bucket or `unmatched`; each matched bucket includes the key values that justified linkage; no bucket violates the frozen uniqueness policy.

**[REC] Task templates.**
- Procurement/government spending: correlate USAspending budget-function or agency outputs with eCFR agency/title metadata using `budget_function_code`, `budget_function_title`, agency names, and title references from `https://api.usaspending.gov/api/v2/budget_functions/list_budget_functions/` and `https://www.ecfr.gov/api/admin/v1/agencies.json` (`https://api.usaspending.gov/docs/endpoints`, `https://www.ecfr.gov/developers/documentation/api/v1`).
- Cybersecurity: correlate STIX bundles or ATT&CK-aligned objects by `type`, `id`, and observable overlap from the official STIX standard or the MITRE ATT&CK enterprise bundle (`https://docs.oasis-open.org/cti/stix/v2.1/os/stix-v2.1-os.html`, `https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json`).
- Customer operations: correlate duplicate support requests by requester, submitter, and first-comment metadata in Zendesk ticket JSON (`https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/`).

### `attest`

**[FACT]** PROV-DM defines provenance as a record about entities, activities, and agents, and explicitly includes responsibility and attribution relations (`https://www.w3.org/TR/prov-dm/`). **[REC]** Define `attest` as: create a provenance or accountability statement that binds an agent to an artifact or outcome. **[REC]** It starts when the subject artifact and responsible agent are fixed and ends when a verifiable attestation record is emitted.

**[REC] Boundary from existing bricks.** Not `approve`, because approval authorizes a future action while attestation records accountability for an existing artifact; not `report`, because the output is provenance metadata rather than audience-facing prose; not `validate`, because it records who stands behind the artifact rather than whether the artifact is correct.

**[REC] Pre-dispatch size driver.** Count `artifacts × asserted_fields × provenance_edges`.

**[REC] Likely call/tool branching.** Usually 1 model call, 0 tools on frozen inputs; 2 calls if external authority metadata must be fetched.

**[REC] Deterministic acceptance oracle.** Output contains `entity_id`, `agent_id`, `generated_at`, `attribution_relation`, and `source_hash`; all IDs are non-empty strings; timestamp parses; source hash matches the frozen input hash.

**[REC] Task templates.**
- Legal/regulatory: attest a summary of `https://www.federalregister.gov/api/v1/documents/2026-00001.json` by emitting PROV-style fields linking the summary artifact to its source document number and generation timestamp (`https://www.w3.org/TR/prov-dm/`, `https://www.federalregister.gov/developers/documentation/api/v1`).
- Supply chain/manufacturing: attest a product-identifier normalization using GS1 Application Identifier definitions and the public JSON-LD dataset (`https://ref.gs1.org/ai/`, `https://ref.gs1.org/ai/GS1_Application_Identifiers.jsonld`).
- Insurance: attest a structured extraction prepared for downstream NAIC or ACORD processing so the output carries explicit source-hash and preparer identity fields (`https://www.acord.org/standards-architecture/acord-data-standards`, `https://content.naic.org/sites/default/files/publications-sta-zu-statistical-handbook.pdf`).

### `dispatch`

**[FACT]** FHIR Task says tasks may be claimed, accepted, rejected, delegated, or reassigned, and the request-intent value set separates `proposal`, `plan`, `order`, `original-order`, `reflex-order`, `filler-order`, and `instance-order` (`https://hl7.org/fhir/task.html`, `https://hl7.org/fhir/valueset-task-status.html`, `https://hl7.org/fhir/valueset-request-intent.html`). **[REC]** Define `dispatch` as: assign a fully specified work item to a performer and transition it from plan or request into active work ownership.

**[REC] Boundary from existing bricks.** Not `plan`, because planning need not name a performer; not `notify`, because notification does not create work ownership; not `approve`, because approval need not route execution to a specific actor.

**[REC] Pre-dispatch size driver.** Count `tasks × assignee_options × required_inputs`.

**[REC] Likely call/tool branching.** 1 model call on snapshot tasks; 2 calls when a live queue is fetched first.

**[REC] Deterministic acceptance oracle.** Every output row has `task_id`, `owner`, and `handoff_status`; `owner` is in a frozen assignee set; `handoff_status` is one of `requested|received|accepted|ready`.

**[REC] Task templates.**
- Healthcare: dispatch radiology or referral tasks using FHIR Task ownership and status fields (`https://hl7.org/fhir/task.html`).
- Customer operations: dispatch Zendesk tickets to groups or assignees using `group_id`, `assignee_id`, `requester_id`, and `custom_status_id` semantics from the Tickets API (`https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/`).
- Procurement/government spending: dispatch a FAR clause review to a contracting queue using eCFR Title 48 structures and agency references (`https://www.ecfr.gov/api/versioner/v1/structure/2026-08-24/title-48.json`, `https://www.ecfr.gov/developers/documentation/api/v1`).

### `merge`

**[FACT]** FHIR defines `$merge` on Patient with explicit source and target records, replaces/replaced-by links, optional preview, and Task tracking for asynchronous processing (`https://hl7.org/fhir/patient-operation-merge.html`). **[FACT]** FHIR Task also lists “merging a set of records” as a workflow step example (`https://hl7.org/fhir/task.html`). **[REC]** Define `merge` as: collapse duplicate records into one survivor while preserving a trace to absorbed IDs.

**[REC] Boundary from existing bricks.** Not `reconcile`, because merge changes identity cardinality rather than only values; not `transform`, because it is not format conversion; not `classify`, because output is a survivor record plus redirects.

**[REC] Pre-dispatch size driver.** Count `records_to_merge × populated_fields × inbound_references`.

**[REC] Likely call/tool branching.** 1 model call for preview logic; 2 calls when a preview plus final patch artifact are separated.

**[REC] Deterministic acceptance oracle.** Exactly one survivor ID remains active; every absorbed ID is listed in redirects or replacement links; output preserves all frozen mandatory fields.

**[REC] Task templates.**
- Healthcare: preview or document a Patient merge using `[base]/Patient/$merge` semantics (`https://hl7.org/fhir/patient-operation-merge.html`).
- Customer operations: merge duplicate support cases that share requester or submitter data (`https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/`).
- Legal/regulatory: merge duplicate cited authorities into one canonical reference set when building a structured register from eCFR title/section metadata (`https://www.ecfr.gov/api/versioner/v1/structure/2026-08-24/title-12.json`).

### `contain`

**[FACT]** NIST SP 800-61 Rev.2 defines containment as a distinct phase before eradication and recovery (`https://csrc.nist.gov/pubs/sp/800/61/r2/final`). **[REC]** Define `contain` as: isolate a compromised or unsafe entity quickly enough to stop propagation while preserving state for later remediation.

**[REC] Boundary from existing bricks.** Not `remediate`, because the root cause may remain; not `monitor`, because containment is an active intervention; not `validate`, because the goal is risk reduction, not conformance checking.

**[REC] Pre-dispatch size driver.** Count `assets × isolation_actions × exception_rules`.

**[REC] Likely call/tool branching.** Usually 1 model call to synthesize the action list from structured evidence; 0 or 1 tools if a live asset inventory is fetched.

**[REC] Deterministic acceptance oracle.** Output includes one action per named asset; every action is from a frozen allowlist such as `disable_account`, `block_hash`, or `isolate_host`; no asset is omitted.

**[REC] Task templates.**
- Cybersecurity: contain KEV-listed exposure by producing a bounded action list from `https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json` (`https://www.cisa.gov/known-exploited-vulnerabilities-catalog`).
- IT operations: contain a misconfigured deployment by halting access until a TOSCA-defined lifecycle operation can repair it (`https://docs.oasis-open.org/tosca/TOSCA/v2.0/os/TOSCA-v2.0-os.html`).
- Financial services: contain a suspicious complaint cluster after triage, for example by pausing a product workflow pending review, using CFPB complaint feeds as intake (`https://www.consumerfinance.gov/data-research/consumer-complaints/`).

### `provision`

**[FACT]** TOSCA 2.0 describes service topology plus lifecycle-management procedures for creation or modification of services through orchestration processes (`https://docs.oasis-open.org/tosca/TOSCA/v2.0/os/TOSCA-v2.0-os.html`). **[REC]** Define `provision` as: instantiate a resource or service from a declared template so that the output is a newly existing runtime object, not only a plan.

**[REC] Boundary from existing bricks.** Not `plan`, because provision changes the external world; not `fetch-external`, because it creates resources rather than reading them; not `transform`, because it does not merely reshape existing data.

**[REC] Pre-dispatch size driver.** Count `node_templates × dependency_edges × lifecycle_steps`.

**[REC] Likely call/tool branching.** Usually 2 to 3 model/tool turns because templates, dependencies, and health checks create naturally staged work.

**[REC] Deterministic acceptance oracle.** Output JSON contains one record per declared node with `node_id`, `intended_type`, and `status`; `status` must be an allowed running/created state; node count must equal template count.

**[REC] Task templates.**
- Cybersecurity/IT operations: provision a small service topology from a TOSCA template and return node inventory (`https://docs.oasis-open.org/tosca/TOSCA/v2.0/os/TOSCA-v2.0-os.html`).
- Customer operations: provision workflow-support artifacts, such as queue descriptors or service connectors, from a frozen topology specification before ticket dispatch (`https://docs.oasis-open.org/tosca/TOSCA/v2.0/os/TOSCA-v2.0-os.html`, `https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/`).
- Supply chain/manufacturing: provision a digital-traceability service that expects GS1 identifier fields defined in public AI references (`https://ref.gs1.org/ai/`).

### Provisional finance-specific candidate: `settle`

**[FACT]** ISO 20022 maintains the authoritative message catalogue and business-area repository for financial messages, including payments and securities (`https://www.iso20022.org/catalogue-messages`, `https://www.iso20022.org/financial-repository`). **[REC]** A distinct `settle` brick is likely warranted for final transfer-of-value work because execution of a payment or securities movement is not the same as approval, reporting, or reconciliation. **[REC]** However, wave-3 paid testing should defer this brick until an exact ISO 20022 catalogue snapshot is frozen and hashed, because this review did not capture the full message-definition pages directly from the ISO host.

## Part 2. Stable public sources for frozen and controlled-live arms

**[REC]** The source set below is restricted to public, no-auth, single-GET endpoints or datasets whose response shape can be frozen and whose live-arm size can be proxied before dispatch. **[REC]** “Declared fetch bytes” should always be computed from the frozen snapshot of the exact endpoint pattern that will be used in the live arm.

| Source | Endpoint URL pattern | Key fields for pre-dispatch proxy | Rate limits / stability | Wave-2 use |
|---|---|---|---|---|
| openFDA drug labels | `https://api.fda.gov/drug/label.json?limit={N}` | `meta.results.limit`, `meta.results.total`, and per-record `openfda.brand_name`, `openfda.generic_name`, `openfda.route`, `openfda.spl_id` | **[FACT]** No-key access is limited to 240 requests/minute and 1,000/day per IP; openFDA returns JSON and documents label coverage from June 2009 to present (`https://open.fda.gov/apis/authentication/`, `https://open.fda.gov/apis/drug/label/`) | No |
| openFDA drug enforcement | `https://api.fda.gov/drug/enforcement.json?limit={N}` | `meta.results.limit`, `meta.results.total`, and per-record `recall_number`, `classification`, `status`, `product_description`, `reason_for_recall` | **[FACT]** Same published limits as above; enforcement data covers publicly releasable recalls from 2004-present and updates weekly (`https://open.fda.gov/apis/authentication/`, `https://open.fda.gov/apis/drug/enforcement/`) | No |
| SEC submissions | `https://data.sec.gov/submissions/CIK##########.json` | top-level issuer metadata plus `filings.recent` arrays such as `accessionNumber`, `filingDate`, `form`, and `size` | **[FACT]** SEC says the APIs require no authentication or API keys and are updated in real time during the day, with bulk ZIP files republished nightly; fair-access guidance applies (`https://www.sec.gov/search-filings/edgar-application-programming-interfaces`, `https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data`) | No |
| SEC company facts | `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` | `cik`, `entityName`, and `facts` namespaces such as `dei` and `us-gaap` | **[FACT]** Same SEC public/no-key and real-time update statements as above (`https://www.sec.gov/search-filings/edgar-application-programming-interfaces`) | No |
| SEC XBRL frames | `https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{uom}/{period}.json` | top-level `taxonomy`, `tag`, `uom`, `pts`, and per-row `accn`, `cik`, `entityName`, `end`, `val` | **[FACT]** Same SEC public/no-key and real-time update statements as above (`https://www.sec.gov/search-filings/edgar-application-programming-interfaces`) | No |
| USAspending budget functions | `https://api.usaspending.gov/api/v2/budget_functions/list_budget_functions/` | `results` array length plus per-row `budget_function_code` and `budget_function_title` | **[FACT]** USAspending states endpoints do not currently require authorization (`https://api.usaspending.gov/docs/endpoints`) | No |
| CMS provider enrollment open data | `https://data.cms.gov/data-api/v1/dataset/2457ea29-fc82-48b0-86ec-3b0755de7515/data?size={N}&offset={K}` | row count from `size`, and per-row fields such as `NPI`, `ENRLMT_ID`, `PROVIDER_TYPE_DESC`, `STATE_CD` | **[FACT]** CMS says its public API returns JSON/JSON:API, supports paging with `size` and `offset`, and caps page size at 5000 rows (`https://data.cms.gov/api-docs`) | No |
| CFPB complaints API | `https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?size={N}&from={K}` | `hits.total`, requested `size`, and per-hit `_source.product`, `issue`, `company`, `date_received` | **[FACT]** CFPB says the complaint database generally updates daily and exposes an Open Data API for controlled downloads (`https://www.consumerfinance.gov/data-research/consumer-complaints/`) | No |
| eCFR agencies | `https://www.ecfr.gov/api/admin/v1/agencies.json` | array length plus per-agency `name`, `short_name`, `slug`, `cfr_references` | **[FACT]** eCFR serves JSON from public GET endpoints; the developer-documentation URL is the owning documentation location even though anti-bot access controls may interpose a request-access page (`https://www.ecfr.gov/api/admin/v1/agencies.json`, `https://www.ecfr.gov/developers/documentation/api/v1`) | No |
| eCFR titles / structure | `https://www.ecfr.gov/api/versioner/v1/titles.json` and `https://www.ecfr.gov/api/versioner/v1/structure/{date}/title-{n}.json` | `number`, `name`, `latest_issue_date`, `size`, child counts, and section sizes | **[FACT]** The versioner endpoints return explicit `size` fields and dated structures suitable for frozen snapshots (`https://www.ecfr.gov/api/versioner/v1/titles.json`, `https://www.ecfr.gov/api/versioner/v1/structure/2026-08-24/title-12.json`) | No |
| Federal Register document JSON | `https://www.federalregister.gov/api/v1/documents/{document_number}.json` | `document_number`, `publication_date`, `type`, `agencies`, `abstract`, `page_length` | **[FACT]** The API returns exact JSON documents by document number; the developer-doc URL is official though anti-bot protections may require separate access for the HTML docs (`https://www.federalregister.gov/api/v1/documents/2026-00001.json`, `https://www.federalregister.gov/developers/documentation/api/v1`) | Yes, but not the three wave-2 documents |
| CISA KEV catalog | `https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json` | top-level `catalogVersion`, `count`, and per-entry `cveID`, `dateAdded`, `dueDate`, `knownRansomwareCampaignUse` | **[FACT]** CISA publishes JSON and CSV catalogue downloads from the official catalog page (`https://www.cisa.gov/known-exploited-vulnerabilities-catalog`, `https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json`) | No |
| GS1 Application Identifiers | `https://ref.gs1.org/ai/GS1_Application_Identifiers.jsonld` | count of AI definitions and per-AI prefix/description fields | **[FACT]** GS1 states that Application Identifiers are prefixes used in barcodes and EPC/RFID tags and publishes the current dataset as JSON-LD (`https://ref.gs1.org/ai/`) | No |

## Part 3. Wave-3 experimental design

### Frozen preregistration rules

**[REC]** The wave-3 preregistration should freeze: `replicates_per_shape: 3`, `minimum_valid_sessions_per_shape: 2`, `telemetry_complete` at the raw HTTP recording layer, `declared_fetch_bytes_kib`, `declared_model_calls`, `declared_tool_calls`, effect-coded brick indicators, the semantic-oracle JSON format for language-output shapes, and the nested grouped-CV procedure.

**[REC]** The semantic oracle format for `summarise`, `report`, `notify`, and `draft` should be:

```json
{
  "required_facts": [
    {"field": "agency_name", "pattern": "(?i)federal reserve|board of governors"},
    {"field": "publication_date", "pattern": "(?i)2026-08-26"},
    {"field": "document_number", "pattern": "(?i)2026-17477"}
  ],
  "pass_condition": "all_present"
}
```

**[REC]** The normalized field values, not the raw prose strings, should be tested against those regexes; exact title equality is prohibited because wave 2 showed literal matching was over-rejective (`docs/foundry-wave2-pilot.md`, `runs/20260826_1627_wave2/README.md`).

### Session structure and budget

**[REC]** Use three paid phases: 162 atomic sessions, 36 composition sessions, and a 24-session blind reserve, for 222 total sessions.

| Phase | Purpose | Cells | Sessions |
|---|---|---:|---:|
| Atomic non-fetch | `evaluate`, `triage`, `dispatch` crossed by 4 context levels × 3 output-unit levels × 3 replicates | 36 per brick | 108 |
| Atomic fetch-class | `correlate` crossed by 3 branch levels × 3 document-size levels × 2 arms × 3 replicates | 54 total | 54 |
| Composition | Winning bricks from atomic phase crossed by arity 1/2/4 × 4 context levels × 3 replicates | 36 total | 36 |
| Blind reserve | Frozen holdout documents and unseen templates, opened once after model freeze | 24 total | 24 |

**[REC]** Context levels should be approximately 500 B, 5 KiB, 50 KiB, and 200 KiB to widen the wave-2 range beyond 1,024 to 102,400 bytes (`docs/taxonomy-expansion-and-wave2-design.md`, `docs/foundry-wave2-pilot.md`). **[REC]** Output-unit levels should be 1, 3, and 6. **[REC]** Fetch-class branching should be 1, 2, and 3 declared model-call/tool-call stages, operationalized for `correlate` as: one preloaded snapshot; one live fetch plus join; or two live fetches plus join.

**[REC]** A conservative planning average of 19,000 input tokens and 1,000 output tokens per session yields about USD 0.58/session at the safety rates used in wave 2, or about USD 128.76 for 222 sessions; this leaves material room below the USD 200 hard cap for incomplete attempts whose full reservations must remain in the ledger (`docs/foundry-wave2-pilot.md`, `runs/20260826_1627_wave2/README.md`). **[REC]** Operational stop should remain at USD 170 so the blind reserve is protected from earlier drift.

### Explicit fixes for the seven wave-2 issues

**Issue 1 — missing telemetry incomplete response.** **[REC]** Recording must first capture raw HTTP status, provider usage block, and response body hash, then derive acceptance. **[REC]** A session with missing usage is stored as `telemetry_complete=false`, counted as a failed attempt, and keeps its full reservation in the safety ledger exactly as wave 2 did. **[REC]** Because every shape has 3 replicates, one incomplete attempt still leaves up to 2 valid sessions and does not destroy identifiability.

**Issue 2 — over-literal summary oracle.** **[REC]** All prose-output shapes use the semantic-oracle JSON schema above; the gate is field-level presence on normalized values, never exact whole-string equality. **[REC]** This applies to new-wave prose tasks that compose with `summarise`, `report`, `notify`, or `draft`.

**Issue 3 — Fetch one-row-per-arm shapes.** **[REC]** Every fetch-class shape uses at least 3 documents per document-size level and 3 replicates per arm. **[REC]** For the minimum pilot, `correlate` alone contributes 3 size levels × 2 arms × 3 replicates = 18 sessions even before branching is crossed, satisfying the requested lower bound.

**Issue 4 — live Fetch response-size omission.** **[REC]** For every live fetch session, compute `declared_fetch_bytes_kib = len(snapshot_response_bytes) / 1024`, rounded to 3 decimals, and freeze it in preregistration. **[REC]** This feature is stored alongside `context_kib`, so the live-arm coefficient is not forced to absorb response-size variation as it did in wave 2 (`docs/foundry-wave2-pilot.md`, `runs/20260826_1627_wave2/README.md`).

**Issue 5 — mechanically predictable targets.** **[REC]** Primary target = output tokens; secondary targets = input tokens and total tokens; model-call and tool-call counts remain recorded but are excluded from promotion gates. **[REC]** Nested grouped CV uses outer folds by shape/source and inner selection by replicate groups inside the training fold only; replicates from the same shape never cross outer folds.

**Issue 6 — tail calibration.** **[REC]** Pre-register `p90_coverage >= 0.80` and `p95_coverage >= 0.85`. **[REC]** Calibration radius is the 0.90 or 0.95 quantile of out-of-fold residuals from training only, computed separately for `live` and `non-live` arms because wave 2 showed live-arm MAE 4,728.9 versus no-arm MAE 174.5 (`runs/20260826_1627_wave2/analysis.json`).

**Issue 7 — dummy-variable collinearity.** **[REC]** Use effect coding for brick indicators so coefficients sum to zero rather than creating intercept-plus-dummy collinearity. **[REC]** Fit ridge regression over the full predictor set and choose alpha by grouped CV inside the training fold only. **[REC]** Before any paid session runs, freeze a design-matrix gate `rank(X) == ncols(X)`; if that fails, do not start the wave.

### Phase gates

**[REC] Atomic stop/go gate.** Advance from atomic to composition only if: (a) every planned source snapshot hash exists, (b) every shape has at least 2 valid sessions out of 3, (c) semantic-acceptance rate is at least 2/3 in every shape, (d) the preflight design matrix is full rank, and (e) at least one of the primary target models beats the constant on grouped-CV MAE in the atomic-only data.

**[REC] Composition stop/go gate.** Advance to blind reserve only if: (a) the additive-plus-interaction candidate selected inside training beats the constant by at least 15% grouped-CV MAE on output tokens, (b) calibration thresholds are met separately for live and non-live arms, and (c) no shape violates the `minimum_valid_sessions_per_shape` rule.

**[REC] Blind reserve gate.** Open the 24 holdouts once, without retuning, only after model form, alpha rule, calibration rule, and acceptance oracles are frozen. **[REC]** Promotion requires the blind output-token MAE to remain below the training constant baseline and the preregistered tail-coverage thresholds to remain satisfied.

### Tested bricks by phase

**[REC]** The paid wave-3 brick set should be limited to `evaluate`, `triage`, `dispatch`, and fetch-class `correlate`. **[REC]** `attest`, `merge`, `contain`, `provision`, and provisional `settle` should be added to the taxonomy now but deferred from paid testing until either an easier no-auth live source or a tighter deterministic oracle is frozen. This keeps the experiment crossed, analyzable, and well below the spend cap while still adding genuinely new mechanisms.

## Part 4. Primary sources cited

| Source | Official URL | Grounded claim |
|---|---|---|
| OMG DMN 1.5 | https://www.omg.org/spec/DMN/1.5/ | Decision logic is a separate standard concern, motivating `evaluate` |
| OMG CMMN 1.1 | https://www.omg.org/spec/CMMN/1.1/ | Case-management vocabulary remains a source of dispatch/escalation gaps |
| OASIS ebXML BPSS 2.0.4 schema | https://docs.oasis-open.org/ebxml-bp/2.0.4/ebbp-2.0.4.xsd | Business transactions, collaborations, signals, timers, and non-repudiation attributes motivate `correlate` |
| ISO 20022 catalogue | https://www.iso20022.org/catalogue-messages | Official message catalogue owning the provisional `settle` brick |
| ISO 20022 repository | https://www.iso20022.org/financial-repository | Official repository for ISO 20022 financial message artifacts |
| HL7 FHIR workflow | https://hl7.org/fhir/workflow.html | Requests, events, and definitions motivate dispatchable work patterns |
| HL7 FHIR Task | https://hl7.org/fhir/task.html | Task queue semantics, reassignment, examples including merge |
| HL7 FHIR task-status value set | https://hl7.org/fhir/valueset-task-status.html | Twelve task states and deterministic status enums |
| HL7 FHIR request-intent value set | https://hl7.org/fhir/valueset-request-intent.html | Separation of proposal, plan, and order for `dispatch` boundaries |
| HL7 FHIR Patient `$merge` | https://hl7.org/fhir/patient-operation-merge.html | Source/target merge semantics and preview/Task tracking |
| NIST SP 800-61 Rev.2 | https://csrc.nist.gov/pubs/sp/800/61/r2/final | Incident analysis, containment before eradication, triage motivation |
| OASIS TOSCA 2.0 | https://docs.oasis-open.org/tosca/TOSCA/v2.0/os/TOSCA-v2.0-os.html | Service topology plus lifecycle management, motivating `provision` |
| W3C PROV-DM | https://www.w3.org/TR/prov-dm/ | Provenance, agents, and attribution motivating `attest` |
| STIX 2.1 | https://docs.oasis-open.org/cti/stix/v2.1/os/stix-v2.1-os.html | Threat/observable object structure motivating `correlate` |
| MITRE ATT&CK enterprise bundle | https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json | Public STIX-form bundle for cyber correlation tasks |
| openFDA authentication | https://open.fda.gov/apis/authentication/ | Published no-key rate limits |
| openFDA drug labels | https://open.fda.gov/apis/drug/label/ | Label coverage and schema context |
| openFDA drug enforcement | https://open.fda.gov/apis/drug/enforcement/ | Recall coverage and weekly updates |
| SEC EDGAR APIs | https://www.sec.gov/search-filings/edgar-application-programming-interfaces | Public, no-key SEC JSON APIs and endpoint patterns |
| SEC EDGAR access guidance | https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data | Fair-access guidance for controlled live arms |
| USAspending API docs | https://api.usaspending.gov/docs/endpoints | No-auth endpoint status and endpoint ownership |
| CMS Data API docs | https://data.cms.gov/api-docs | Public JSON API, filtering, and paging limits |
| CFPB complaints data | https://www.consumerfinance.gov/data-research/consumer-complaints/ | Daily complaint updates and Open Data API availability |
| eCFR API docs | https://www.ecfr.gov/developers/documentation/api/v1 | Official developer-doc location for eCFR endpoints |
| Federal Register API docs | https://www.federalregister.gov/developers/documentation/api/v1 | Official developer-doc location for Federal Register endpoints |
| CISA KEV catalog | https://www.cisa.gov/known-exploited-vulnerabilities-catalog | Official JSON/CSV vulnerability catalogue publication |
| GS1 Application Identifiers | https://ref.gs1.org/ai/ | Public AI definitions and JSON-LD dataset |
| ACORD Data Standards | https://www.acord.org/standards-architecture/acord-data-standards | Insurance straight-through-processing field expectations |
| NAIC Statistical Handbook | https://content.naic.org/sites/default/files/publications-sta-zu-statistical-handbook.pdf | Insurance regulatory-reporting field context |
| Wave-2 pilot report | docs/foundry-wave2-pilot.md | Spend, telemetry failure, literal oracle failure, live-fetch omission |
| Wave-2 run README | runs/20260826_1627_wave2/README.md | Grouped ridge result, interval coverage, budget ledger |
| Wave-2 analysis artifact | runs/20260826_1627_wave2/analysis.json | Coefficient intervals, matrix rank, condition number, live-arm MAE |

---

## Part 5. Big-T Notation analysis

### 5.1 Source description and epistemic status

**[FACT]** Big-T Notation is an industry framework authored by Dan Neff (Adobe) and published by the Tokenomics Foundation, a Linux Foundation project, at `https://www.tokeneconomics.com/projects/big-t-notation/`. A companion paper is available at `https://www.tokeneconomics.com/docs/projects/big-t/big-t-notation-paper/`. The project page was last modified 2026-08-26 per its `og:updated_time` metadata. The framework is described by its author as "deliberately informal" with "no proofs and no master theorem" and as "a thinking tool … a shared vocabulary" (`https://www.tokeneconomics.com/docs/projects/big-t/big-t-notation-paper/` §2.3). **[FACT]** This characterisation is accurate: the document is a primary source for its own claims but is not a peer-reviewed scientific work and should be cited accordingly.

**[FACT]** The framework's sole explicit primary-source citation is to Big-O notation via the Wikipedia article on Big-O (`https://en.wikipedia.org/wiki/Big_O_notation`), tracing the concept to Bachmann (1894) and Knuth's *The Art of Computer Programming* (`https://www.tokeneconomics.com/docs/projects/big-t/big-t-notation-paper/` Abstract). The quantitative claims in the paper — "SQL views shrinking clinical data from ~240,000 tokens to 19,000", "retrieval pipelines eliminating better than 99.9 percent of a million-token corpus", "roughly 80 percent fewer tokens on batch operations", and "compact tabular formats can consume roughly half the tokens of verbose ones" — are described as "published results" without specific citations to identifiable sources. **[REC]** These claims should be treated as illustrative benchmarks requiring independent verification rather than as grounded experimental evidence.

### 5.2 The Big-T dimensions

**[FACT]** Big-T defines a three-variable formula **T(n · k · a)** and a six-rung complexity ladder (`https://www.tokeneconomics.com/docs/projects/big-t/big-t-notation-paper/` §2.1–2.2):

| Symbol | Definition in Big-T | Complexity rungs that depend on it |
|---|---|---|
| **n** | Input size or request count; "two things that often move together" | T(log n), T(n), T(n·k), T(n·k·a) |
| **k** | Model calls per request; includes reasoning steps, multi-turn chains, tool calls that replay context; "usually invisible" | T(n·k), T(n·k·a) |
| **a** | Agent depth; sub-agents calling sub-agents | T(n·k·a) |

The six rungs are: **T(1)** constant (cache hit, static lookup); **T(log n)** sublinear (deterministic pre-filtering); **T(n)** linear (one model call, cost ∝ input); **T(n·k)** multiplicative (k calls per request, k invisible); **T(n·k·a)** agent-multiplicative (orchestrator spawning workers); **T(∞)** unbounded (retry loop without termination).

### 5.3 Tracing Big-T claims to prior repo evidence

**[FACT]** The repo independently discovered the same phenomena Big-T names:

| Big-T observation | Repo evidence | Source |
|---|---|---|
| Large fixed per-call overhead ("startup cost") dominates over task work at small n | Wave-1: constant model won with intercept 21,029 tokens; single-call token range 19,901–20,649; CV ≈ 0.8% | `runs/20260826_1416_wave1/model.json`; `docs/calibration-findings.md` §3.1 |
| "Hidden k": context-replaying tool calls multiply cost; k is invisible in the request | Wave-1 outlier: `real-apple-risk-memo` triggered an unplanned external fetch, expanded to 3 model calls, cost ~41 k tokens above the constant | `runs/20260826_1416_wave1/feedback.jsonl` |
| k=2 (live Fetch) vs k=1 (snapshot) creates a multiplicative regime | Wave-2 live-arm MAE 4,728.9 tokens vs no-arm MAE 174.5 tokens; live_fetch CI95 [409, 8,237] (wide; absorbs both continuation overhead and source size) | `runs/20260826_1627_wave2/analysis.json` |
| Input-size (n) coefficient ≈ 0.37–0.40 tokens/byte for T(n) bricks | Calibration campaign: `tokens = 36,044 + 0.4174 × bytes` across 3 repos; wave-2 context_kib CI95 [154, 312] tokens/KiB ≈ 0.15–0.30 tokens/byte | `docs/calibration-findings.md` §3.4; `runs/20260826_1627_wave2/analysis.json` |
| T(log n): retrieve-embedded pre-filters corpus before inference | Retrieve-Embedded (B04) calibrated at ≈5,384 tokens/unit (reading pre-indexed corpus); no model call multiplier | `docs/composition-findings.md` §3.3 |

**[FACT]** Big-T names the "serialization" lever: "compact tabular formats can consume roughly half the tokens of verbose ones" (`https://www.tokeneconomics.com/docs/projects/big-t/big-t-notation-paper/` §4). The repo's Transform brick (structured field mapping) produced the lowest within-family MAE in wave-2 at 26.0 tokens (`runs/20260826_1627_wave2/analysis.json`), consistent with the bounded-output property of structured transformation tasks.

### 5.4 Mapping Big-T classes to 17 current bricks and wave-3 candidates

**[REC]** Each brick is classified by the Big-T rung most likely to govern its single-session cost at the atomic level (A=1, no composition). The declared-k values are pre-dispatch observable; reasoning-token k is always post-run.

| Brick | Big-T class (atomic) | Declared k | Pre-dispatch n proxy | Notes |
|---|---|---|---|---|
| review | T(n) | 1 | context_kib | Reading cost ∝ input; confirmed by calibration campaign |
| extract | T(n) | 1 | context_kib | Same driver as review; output small and bounded |
| classify | T(n) → T(1) | 1 | context_kib or 0 | Approaches T(1) when input is pre-filtered to small label set |
| **retrieve-embedded** | **T(log n)** | 1 | corpus_kib × retrieval_k | Pre-filtered embedding search; Big-T "RAG done right" |
| reconcile | T(n) | 1 | context_kib | Cost ∝ two-source input size |
| draft | T(n) | 1 | context_kib | Output-bounded by template structure |
| remediate | T(n) | 1 | context_kib | Correction proportional to error-document size |
| validate | T(n) → T(1) | 1 | rule_count × field_count | Schema validation is T(1); logical validation is T(n) |
| report | T(n) | 1 | context_kib | Audience-targeted output; output size ∝ schema |
| **fetch-external** | **T(n·k)** | 2 (live) / 1 (snapshot) | context_kib + declared_fetch_kib | k=2: tool call + reading call; this is the wave-2 live Fetch pattern |
| score | T(n) | 1 | alternatives × criteria | Product of countable quantities; bounded output |
| summarise | T(n) | 1 | context_kib | Output ∝ input at compression ratio; bounded template lowers constant |
| monitor | T(n) | 1 | conditions × time_points | Event list bounded by condition count |
| plan | T(n) or T(n·k) | 1–2 | subtask_count | May require live resource lookup → k=2 |
| notify | T(n) | 1 | recipient_groups × fields | Bounded message format |
| approve | T(n) | 1 | evidence_items | Artifact production only; wait time excluded |
| transform | T(n) | 1 | field_mapping_count | Wave-2 confirmed: lowest MAE family; bounded output |
| **evaluate** (wave-3) | T(n) or T(n·k) | 1 (snapshot) / 2 (live) | decision_rows × fields | Live rule fetch → k=2 |
| **triage** (wave-3) | T(n) | 1–2 | items × routing_options | Live intake fetch → k=2 |
| **correlate** (wave-3) | **T(n·k)** | 1/2/3 declared | context_kib + declared_fetch_kib | Primary wave-3 test of k multiplier with 3 declared levels |
| **attest** (wave-3) | T(n) | 1 | artifacts × provenance_edges | Bounded metadata output |
| **dispatch** (wave-3) | T(n) | 1–2 | tasks × assignees | Live queue fetch → k=2 |
| **merge** (wave-3) | T(n) | 1–2 | records × fields | Preview + patch may require k=2 |
| **contain** (wave-3) | T(n) | 1 | assets × isolation_actions | Bounded action-list output |
| **provision** (wave-3) | **T(n·k)** | 2–3 declared | node_templates × lifecycle_steps | Multi-stage template/dependency/health-check pattern |

### 5.5 Pre-run observable features vs. post-run leakage

**[REC]** Every feature in the design matrix must be computable from information available before the first token is dispatched. The Big-T dimensions introduce the following leakage risks:

| Feature | Status | Leakage mode if misused |
|---|---|---|
| `context_kib` (= Big-T **n**, input side) | **Pre-run ✓** | None; computed from prompt bytes |
| `declared_fetch_kib` (= Big-T **n**, fetch side) | **Pre-run ✓** | Must be derived from frozen snapshot, not from live response; wave-2 omitted this and it leaked via `live_fetch` coefficient |
| `declared_model_calls` (= Big-T **k**, declared) | **Pre-run ✓** | Only if call count is fixed by design (Fetch, correlate); must not be inferred from task complexity estimate |
| Reasoning tokens (a form of **k**) | **Post-run ✗** | Wave-2 records show `reasoning: 0` for all gpt-5-mini sessions; non-zero reasoning tokens would be entirely post-run; never use as a feature |
| `agent_depth` (= Big-T **a**) | **Pre-run ✓** (= 1 always for atomic sessions) | Zero variance across all single-agent sessions; drop from design matrix |
| **T-class label** (T(1)/T(log n)/T(n)/T(n·k)/…) | **Derived variable ✗** | = f(declared_model_calls, agent_depth); perfectly collinear with a subset of existing features; must not be added as an independent predictor |
| Complexity-class dummy `is_multiplicative` | **Derived variable ✗** | = `I(declared_model_calls > 1)`; collinear with `declared_model_calls` when k ∈ {1, 2}; adds no independent information |
| `declared_output_bound_tokens` | **Pre-run ✓ (new)** | None if set from prompt template before dispatch; not collinear with input features |

**[FACT]** The wave-2 analysis shows that reasoning tokens were recorded as zero for all 30 sessions (`runs/20260826_1627_wave2/README.md`: "reasoning (subset of output): 0"). Reasoning tokens would be pure post-run leakage if non-zero and must never enter the design matrix as a pre-dispatch feature.

### 5.6 Overlaps and collinearity risks from Big-T features

**[REC]** The table below maps each potential Big-T-derived feature against the existing and recommended wave-3 predictors and assesses the collinearity risk:

| Big-T feature | Equivalent or alias in wave-3 design | Collinearity risk | Action |
|---|---|---|---|
| n (input size) | `context_kib` | **Identical** | Do not add; already present |
| n (fetch response size) | `declared_fetch_kib` | **Identical** | Already in wave-3 recommendation (Part 3, Issue 4) |
| k (model calls) | `declared_model_calls` | **Identical** | Already in wave-3 recommendation (Part 3, Issue 5) |
| T-class categorical | Derived from `declared_model_calls` and `agent_depth` | **Perfect multicollinearity** with existing features | Drop; never add as predictor |
| `is_multiplicative` dummy | = `I(declared_model_calls > 1)` | **Perfect collinearity** with `declared_model_calls` when k ∈ {1, 2, 3} | Drop |
| a (agent depth) | Constant = 1 for all sessions | **Zero variance** | Drop; cannot be estimated |
| `declared_output_bound_tokens` | Not currently in wave-3 design | **Orthogonal** to context_kib and declared_model_calls | **Add** (see §5.7) |
| Serialization format (tabular vs. prose) | Partially captured by brick identity | Low risk but partially redundant with brick effects | Record as metadata; test as auxiliary feature in a separate pass |

**[REC]** The risk of adding Big-T class labels as dummy variables is identical to the wave-2 collinearity concern that was already identified: if `is_multiplicative = I(declared_model_calls > 1)`, and `declared_model_calls ∈ {1, 2, 3}`, then the two predictors carry the same information and their VIFs will be very large. Effect-coded brick indicators already partially capture the T-class structure (fetch-class bricks are systematically k > 1), so adding a T-class dummy would introduce near-perfect collinearity with the fetch brick indicator.

### 5.7 Recommended minimal orthogonal feature grammar

**[REC]** The Big-T analysis motivates adding exactly **one genuinely new pre-dispatch feature** that is not already in the wave-3 design: **declared output bound**.

**Declared output bound** (`declared_max_output_tokens`): the maximum number of output tokens explicitly specified in the prompt template before dispatch. For structured bricks with a fixed output schema (transform, validate-schema, evaluate, triage), this is computable from the schema size. For open-ended bricks (summarise, draft, review), it is either the explicit `max_tokens` parameter or a declared word-limit in the instruction. It is distinct from `context_kib` (input side) and `declared_model_calls` (call multiplier), so it is a genuinely orthogonal predictor.

**[FACT]** The repo's calibration campaign independently observed that structured output tasks (code_write, test_write) selected a `constant` model form while comprehension tasks selected `affine@bytes`, suggesting that output-bounded tasks have a different cost structure from input-proportional tasks (`docs/calibration-findings.md` §4). Big-T's "serialization" and "bounded template" concepts name the same phenomenon. Wave-2 Transform brick had within-family MAE of 26.0 tokens (lowest of any family), consistent with tightly bounded output (`runs/20260826_1627_wave2/analysis.json`).

**[REC]** The minimal orthogonal feature grammar for wave-3, incorporating the one new Big-T-motivated feature, is:

```
output_tokens = α                               # startup cost; model-version specific; T(1) floor
              + β₁ × context_kib               # Big-T n, input side; T(n) slope
              + β₂ × declared_fetch_kib        # Big-T n, fetch side; orthogonal when fetch_kib > 0
              + β₃ × declared_model_calls      # Big-T k; T(n·k) multiplier; declared pre-dispatch
              + β₄ × declared_max_output_tokens# Big-T "serialization"; output-bound floor [NEW]
              + γⱼ × brick_effect_coded[j]     # mechanism identity; sum-to-zero coded
              + δ  × composition_arity         # brick count in session
              + ε
```

**[REC]** This grammar has 6 + J predictors (where J = number of brick types − 1 with effect coding). For 8 wave-3 brick types it yields 6 + 7 = 13 predictors, unchanged from the prior wave-3 estimate. Adding `declared_max_output_tokens` replaces no existing predictor; it occupies an orthogonal slot that was previously unoccupied. VIF must be checked as before.

**[REC]** The T-class label itself should be recorded as **metadata only** (not as a design-matrix predictor): it provides useful language for describing sessions to stakeholders and for stratifying the held-out blind reserve, but it does not add statistical information beyond the numeric features already in the grammar.

### 5.8 Researcher recommendation: when Big-T adds value

**[REC]** Adopt the Big-T vocabulary for three purposes:

1. **Stratification and labeling**: classify each session's T-class before dispatch and record it as metadata; use T-class as the primary stratification variable for ensuring the blind reserve includes at least one T(n·k) session.

2. **`declared_max_output_tokens` as a new pre-dispatch feature**: preregister the formula `declared_max_output_tokens = min(prompt_template_schema_tokens, max_tokens_param)` before any wave-3 session runs; freeze the value alongside `context_kib` and `declared_fetch_kib`; add it to the design matrix.

3. **Hidden-k instrumentation**: use Big-T's "hidden k" concept as a reminder to record `reasoning_tokens` as a separate post-run diagnostic channel even on gpt-5-mini where it is currently zero; if a model change produces non-zero reasoning tokens, they will appear as post-run leakage in a predictor that previously had none.

**[REC]** Do not adopt Big-T's complexity-class dummies as design-matrix predictors; do not add a `is_multiplicative` binary; do not add `agent_depth` (= 1 for all sessions); do not re-encode `declared_model_calls` as a Big-T-class label.

### 5.9 Critic assessment: does adopting Big-T improve identifiability?

**Net verdict: marginally, through one new feature; not through class labels.**

**[REC]** The wave-2 identifiability failures — missing telemetry, over-literal oracle, one-row-per-arm Fetch shapes, omitted fetch-size feature, mechanically predictable tool-call targets, under-calibrated tails, dummy-variable collinearity — are design-execution failures. Big-T does not address any of them. The seven fixes in Part 3 of this document address them directly; Big-T does not change those fixes.

**[REC]** On the question of additional pre-dispatch features, the honest count is: Big-T motivates **one new orthogonal predictor** (`declared_max_output_tokens`) and provides useful vocabulary for communicating experiment design to non-statistician stakeholders. The other Big-T dimensions (n, k) are already in the wave-3 design under different names; a (agent depth) is zero-variance across all single-agent sessions.

**[REC]** The Big-T paper explicitly says "Big-T deliberately ignores constants" (`https://www.tokeneconomics.com/docs/projects/big-t/big-t-notation-paper/` §2.3). The repo's dominant finding across waves 1 and 2 is that the constant (startup cost, approximately 20,000 tokens for gpt-5-mini) dominates token variance more than any other term. A framework that ignores constants is useful for architecture classification but not for the quantitative regression problem this repo addresses. Adopting Big-T as the primary feature-engineering framework would have led wave-1 to the same near-constant result, because T-class labels do not vary across single-call atomic sessions.

**[REC]** The specific claim "adopting Big-T improves identifiability" is **true only if** the new `declared_max_output_tokens` feature is added and contributes signal above the noise floor. That is a testable hypothesis; it should be preregistered and tested in wave-3 as a secondary promotion gate: the feature is retained only if its ridge coefficient's 95% bootstrap CI excludes zero.

| Addition from Big-T | Improves identifiability? | Reason |
|---|---|---|
| `declared_max_output_tokens` | **Yes, possibly** — testable | Orthogonal to input features; explains bounded-output vs. open-ended variance; wave-2 Transform MAE 26 vs. Fetch MAE 3,087 suggests output-side predictor is missing |
| T-class categorical dummy | **No** | Collinear with `declared_model_calls`; adds zero independent information |
| `is_multiplicative` binary | **No** | Subset of `declared_model_calls`; perfect collinearity risk |
| `agent_depth` | **No** | Zero variance for all single-agent sessions |
| Big-T vocabulary labels (metadata) | **Neutral** | Useful for communication; not a statistical predictor |

**[REC]** The correct citation for this document is: Dan Neff, "Big-T Notation," Tokenomics Foundation (Linux Foundation project), published 2026-08-01, last modified 2026-08-26, `https://www.tokeneconomics.com/projects/big-t-notation/`; companion paper at `https://www.tokeneconomics.com/docs/projects/big-t/big-t-notation-paper/`. The framework should be cited as an industry framework document, not a peer-reviewed source, and its empirical claims (92%, 99.9%, 80% reductions) should be noted as uncited benchmarks pending independent verification.

| Source | Official URL | Grounded claim |
|---|---|---|
| Big-T Notation project page | https://www.tokeneconomics.com/projects/big-t-notation/ | T(n·k·a) formula; complexity ladder; five levers; worked examples |
| Big-T Notation companion paper | https://www.tokeneconomics.com/docs/projects/big-t/big-t-notation-paper/ | Full ladder definitions; hidden-k concept; serialization lever; "no proofs" epistemic status |
| Wikipedia Big-O notation | https://en.wikipedia.org/wiki/Big_O_notation | Sole primary-source citation within Big-T for the algorithmic complexity analogy |

---

## Part 6. Settled decisions, revised preregistration, and binding design

**Scope of this part.** The following settled decisions and critic constraints supersede the Part 3 design wherever they conflict. Parts 1, 2, and 5 remain as taxonomy and analysis background. The binding preregistration for wave-3 paid sessions is this part alone.

### 6.1 Binding constraints

**[REC]** The following eleven constraints are settled; no session may be dispatched unless all of them are satisfied by the frozen preregistration.

| ID | Constraint |
|---|---|
| C-01 | **Quote-time features only.** Every predictor in the design matrix must be computable from information available at the moment the API request object is constructed, before any response byte is received. Reasoning tokens, observed branch count, response size, and post-call metadata are prohibited predictors. |
| C-02 | **Required null = exact tokenizer/harness baseline.** The fixed overhead (system prompt + tool schema catalog + instruction prefix) must be counted via the Responses API `/v1/responses/input_tokens` endpoint with a zero-content task payload, called before any paid generation call. **[FACT]** tiktoken's `o200k_base` encoding (the correct encoding for `gpt-5-mini`, per `openai/tiktoken` `model.py` prefix rule `"gpt-5" → "o200k_base"`, SHA `166db06`) can count plain-text tokens offline, but the OpenAI token-counting guide explicitly states that "tools and schemas add tokens that are hard to count locally" and that "formatting tokens used to represent request structure … might not appear in the text or fields you tokenize locally" (`https://developers.openai.com/api/docs/guides/token-counting`). Therefore offline tiktoken counting of the full harness (including tool schemas) yields only a lower-bound estimate; the API endpoint is required for exactness. The counted value is stored as `fixed_overhead_tokens` with a SHA-256 hash of the harness components. This value is the fixed intercept of the cost model, not a fitted parameter. Two confirmation null probes (trivial task, same harness) verify the count against actual billed input. |
| C-03 | **Primary uncertain targets are output tokens and k_free.** `output_tokens` is the count of model-generated response tokens excluding reasoning. `k_free = observed_model_calls − k_declared` is unforced branching. Model-call count and tool-call count are excluded from promotion gates because they are mechanically predictable for architecturally-fixed bricks. |
| C-04 | **Admit new bricks only for mechanisms not reducible to {task_payload_tokens, output_bound_tokens, output_spec_units, k_declared}.** A candidate brick that produces a cost profile fully explained by those four quote-time features under any linear combination is a size-or-schema variant, not a new mechanism. The admission test is: does the brick's mechanism produce a non-trivial distribution of `k_free > 0` as a structural property of the mechanism type? |
| C-05 | **Hierarchical preregistered interactions.** Interaction terms are tested level by level; a higher level is tested only if the preceding level's gate passes. All interaction terms are frozen in the registry before any session runs. |
| C-06 | **Machine-readable registry.** The full preregistration is a committed JSON file (§6.6 below specifies its schema). The analysis script must be able to reconstruct the design matrix from the registry without manual intervention. |
| C-07 | **Cross minimal/low vs medium/medium.** `reasoning.effort` and `text.verbosity` are treated as a joint factor with two levels: `{effort=minimal, verbosity=low}` (wave-2 default) and `{effort=medium, verbosity=medium}` (new). This factor is crossed with brick type in the Phase 3 block; it is never confounded with payload level. |
| C-08 | **Randomized cold/warm cache.** Within each shape-replicate block, cache state is assigned randomly: cold (unique instruction prefix suffix so the provider cannot cache the harness) versus warm (standard prefix repeated). `cache_warm` is a binary quote-time feature. |
| C-09 | **Independent sources; disjoint blind pools.** Every source document or endpoint used in training sessions must be from one of the three designated training-eligible industries. Every blind-reserve session must draw from one of the three designated blind industries. No source document, API endpoint, or industry vertical may appear in both pools. |
| C-10 | **Tail claims deferred until ≥ 30 independent shape groups.** Wave-2 completed with 9 groups; wave-3 is designed to reach 30+ groups across all training phases, at which point p90/p95 coverage claims may be computed. Until that threshold is reached, only signal gates and main-effect CI95 results are reported. |
| C-11 | **Repair-then-pause protocol.** A shape that fails its acceptance gate may undergo one repair pass (oracle pattern fix or prompt-wording fix within preregistered bounds; no gate-threshold change; no added signal rows). If the repair block also fails the gate, execution pauses and requires new approval before any further phase. |

### 6.2 Revised critic feature grammar

**[REC]** The six critic predictors replace the five wave-2 features (`context_kib`, `units`, `transform`, `fetch`, `live_fetch`) with a grammar that separates the harness constant from payload, splits input from output dimensions, and isolates declared branching from free branching.

| Predictor | Symbol | Quote-time source | Wave-2 equivalent | Collinearity status |
|---|---|---|---|---|
| Task payload tokens | `task_payload_tokens` | `len(prompt_document_bytes) ÷ 4` (UTF-8 rough estimate) or tiktoken `o200k_base` for plain-text lower bound; exact harness total (including tool schemas) requires `POST /v1/responses/input_tokens` pre-dispatch API call | Subset of `context_kib` (conflated with harness) | Orthogonal to harness after separation |
| Fixed overhead tokens | `fixed_overhead_tokens` | Offline tokenization of harness; stored as constant per deployment config | Conflated into fitted intercept in wave-2 | **Fixed constant, not a predictor;** absorbed into intercept by design |
| Output bound tokens | `output_bound_tokens` | `min(prompt_max_tokens_param, schema_field_count × avg_field_tokens)` | Not in wave-2 | Orthogonal to task_payload_tokens and k_declared |
| Output spec units | `output_spec_units` | Count of discrete output items/records/fields declared in the prompt template | Partially captured by `units` in wave-2 | Possible moderate correlation with task_payload_tokens; check VIF |
| Declared model calls | `k_declared` | Integer from the task specification; 1 for snapshot arms, 2–3 for multi-call arms | `live_fetch` binary (conflated call count with arm identity) | Must not be encoded as a binary; use integer |
| Effort level | `effort_level` | 0 = `{minimal, low}`, 1 = `{medium, medium}`; from deployment configuration | Not in wave-2 | Orthogonal by design (crossed factor) |
| Cache warm | `cache_warm` | 0 = cold (unique prefix suffix), 1 = warm (repeated standard prefix); randomized per session | Implicit in some wave-2 sessions but not recorded | Orthogonal by design (randomized) |
| Brick effect code | `brick_j` | Sum-to-zero effect-coded indicator for brick type j; J bricks → J−1 predictors | `transform`, `fetch` dummies in wave-2 (reference-coded; collinearity risk) | Sum-to-zero coding eliminates intercept-dummy collinearity |
| Composition arity | `composition_arity` | Count of brick types in session | Not in wave-2 | May correlate with task_payload_tokens in composition sessions; monitor VIF |

**[REC]** `fixed_overhead_tokens` is the required null baseline (C-02). It is not entered as a predictor column; it is subtracted from measured input tokens before fitting to isolate the payload signal. The model fitted is therefore:

```
output_tokens = α (fitted residual above fixed overhead)
              + β₁ × task_payload_tokens
              + β₂ × output_bound_tokens
              + β₃ × output_spec_units
              + β₄ × k_declared
              + β₅ × effort_level
              + β₆ × cache_warm
              + γⱼ × brick_j           (J−1 effect-coded terms)
              + δ  × composition_arity
              + ε
```

**[REC]** Rejected predictors and their rejection reasons:

| Rejected predictor | Reason |
|---|---|
| `k_observed` | Post-run: observed branch count is available only after response; using it as a predictor leaks the target into the feature |
| `a_observed` | Post-run and zero-variance (a=1 for all single-agent sessions) |
| Big-T class label (T(n), T(n·k), …) | Derived variable: = f(k_declared, a_observed); collinear with `k_declared`; adds no independent information |
| `is_multiplicative` | = I(k_declared > 1); collinear subset of `k_declared` |
| `reasoning_tokens` | Post-run; zero for gpt-5-mini at minimal effort (`runs/20260826_1627_wave2/README.md`); must not be used even if non-zero in future |
| `response_size_observed` | Post-run; using prior-session response size to predict current session output violates the quote-time constraint except for `declared_fetch_kib` (derived from frozen snapshot, not observed response) |

### 6.3 Strict brick admission test applied to Part 1 candidates

**[REC]** Each Part 1 candidate is tested against C-04: "does the mechanism produce a non-trivial distribution of k_free > 0 as a structural property?" Bricks that fail are retained in the taxonomy (Part 1) but deferred from paid testing.

| Brick | k_free structural property? | Reasoning | Admission result |
|---|---|---|---|
| evaluate | No | Applies a frozen rule set to a typed input; output fields and bound are known before dispatch; k_free ≈ 0 by design | **Defer** — reducible to task_payload_tokens + output_bound_tokens |
| triage | No | Maps N items to N priority/route pairs; output_spec_units = N; k_free ≈ 0 | **Defer** — reducible to output_spec_units |
| correlate | Yes | Multi-arm design exposes k_declared = 1/2/3; additionally, grouping decisions over ambiguous keys can require iterative refinement not declared before dispatch → k_free > 0 possible | **ADMIT for paid testing** |
| attest | No | Produces fixed provenance fields; output schema known before dispatch; k_free ≈ 0 | **Defer** — reducible to output_bound_tokens |
| dispatch | No | Assigns ownership from a fixed assignee set; output_spec_units known before dispatch | **Defer** — reducible to output_spec_units |
| merge | Marginal | FHIR $merge has a distinct preview phase (k=1) followed by apply phase (k=2) and optional Task-tracking call (k=3); however, the three phases can be declared before dispatch as k_declared = 2 or 3 → k_free may remain ≈ 0 if phases are architecturally fixed | **Defer** pending design that isolates k_free from k_declared |
| contain | No | Produces a bounded action list; output_bound_tokens × output_spec_units cover the cost driver | **Defer** — reducible to output_spec_units + output_bound_tokens |
| provision | Yes | TOSCA lifecycle operations have declared stages (k_declared = 2–3), but health-check failures create additional unforced tool calls → k_free > 0 from retry logic | **ADMIT as secondary arm** (small scale; 12 sessions) |

**New brick added under C-04:**

**`diagnose`** — **ADMIT for paid testing** as the primary uncertain-k mechanism:
- **[FACT]** NIST SP 800-61 Rev.2 §3.2.3 defines an explicit incident analysis phase that precedes containment, describing the process of identifying the attack vector, scope, and indicators of compromise from evidence; it does not prescribe a fixed number of analysis steps (`https://csrc.nist.gov/pubs/sp/800/61/r2/final`). **[FACT]** ITIL 4 defines Problem Management as a distinct practice in which root-cause identification is an iterative process that may require multiple investigation passes (`https://www.axelos.com/certifications/itil-service-management`).
- **Operational definition.** Given a structured evidence set (logs, alerts, or anomaly records) and a declared symptom list, identify the root cause, contributing factors, and confidence level. The task ends when a structured diagnosis is emitted. Remediation is excluded.
- **Boundary from existing bricks.** Not `monitor` (monitor observes thresholds on live streams; diagnose receives a bounded historical evidence set and reasons to a conclusion); not `remediate` (remediate assumes a known issue and applies a fix; diagnose discovers the cause); not `review` (review reads and assesses; diagnose reasons iteratively from evidence to hypothesis).
- **k_free driver.** The model may decide, based on initial evidence, that additional clarification passes are warranted before committing to a root cause. This branching is not architecturally declared; it arises from the task content. k_free ~ Bernoulli(p) where p is estimated from the data.
- **Pre-dispatch size driver.** `output_spec_units` = `symptom_count × evidence_items`; `output_bound_tokens` = schema size of the structured diagnosis record.
- **Deterministic acceptance oracle.** Output JSON validates against frozen schema with required keys `root_cause` (non-empty string), `evidence_cited` (array of IDs all present in input), `confidence` (enum: `low|medium|high|definitive`), `contributing_factors` (array); all `evidence_cited` items must resolve to input evidence items; no hallucinated sources allowed (checked by set-intersection of cited IDs against input IDs).
- **Task templates.** (1) Cybersecurity: diagnose which CVE from a CISA KEV subset triggered a simulated alert, using `cveID`, `dateAdded`, and `knownRansomwareCampaignUse` fields from `https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json`. (2) IT operations: diagnose a root cause from a structured event log derived from a TOSCA service topology, citing node and operation IDs from `https://docs.oasis-open.org/tosca/TOSCA/v2.0/os/TOSCA-v2.0-os.html`. (3) Healthcare safety: diagnose a drug enforcement recall pattern from openFDA enforcement records, citing `recall_number` and `product_description` from `https://api.fda.gov/drug/enforcement.json`.

**[REC]** Final paid-testing brick list: `correlate` (primary), `diagnose` (uncertain-k), `provision` (secondary, 12 sessions). `summarise` and `transform` are retained as reference bricks because their wave-2 signal passed; they appear in the crossed-factors phase only.

### 6.4 Source pools and blind-industry specification

**[REC]** The three training-eligible industries and three blind-only industries below are designated before any session runs and cannot be changed without new approval. A source document or endpoint used in any training session is prohibited from the blind pool regardless of its industry designation.

**Training-eligible industries and primary endpoints:**

| Industry | Primary endpoints | Session types |
|---|---|---|
| Financial services | `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`; `https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{uom}/{period}.json`; `https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?size={N}` | correlate, diagnose |
| Procurement / government spending | `https://api.usaspending.gov/api/v2/budget_functions/list_budget_functions/`; `https://www.ecfr.gov/api/versioner/v1/structure/{date}/title-{n}.json`; `https://www.ecfr.gov/api/admin/v1/agencies.json` | correlate, compositions |
| Healthcare | `https://api.fda.gov/drug/enforcement.json?limit={N}`; `https://api.fda.gov/drug/label.json?limit={N}`; `https://data.cms.gov/data-api/v1/dataset/2457ea29-fc82-48b0-86ec-3b0755de7515/data?size={N}` | diagnose, provision |

Federal Register documents other than the three wave-2 snapshots may be used in training for the legal/regulatory vertical, provided the specific document numbers are registered before use. **[FACT]** The three wave-2 documents (`2019-24499`, `2024-06550`, `2025-01358`) are reserved as permanent exclusions from all new training sessions (`experiments/foundry_wave2/api_manifests.json`).

**Blind-only industries (no session from these sources in training):**

| Industry | Primary endpoints | Rationale for blind designation |
|---|---|---|
| Cybersecurity / IT operations | `https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json`; `https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json` | No training session uses these; CISA KEV and ATT&CK bundles are large and structurally distinct |
| Supply chain / manufacturing | `https://ref.gs1.org/ai/GS1_Application_Identifiers.jsonld`; UN/CEFACT EDIFACT message directory at `https://unece.org/trade/uncefact/introducing-unedifact` (static reference files) | No training session uses these |
| Legal / international regulatory | EUR-Lex web-services SOAP or REST endpoint for a specific EUR-Lex document: `https://eur-lex.europa.eu/legal-content/EN/TXT/JSON/?uri=CELEX:{celex_id}` (EUR-Lex CELLAR REST, documented at `https://eur-lex.europa.eu/content/tools/webservices/`) | Industry not represented in any training source; non-US regulatory text tests generalization beyond US government sources |

**[REC]** Each blind session must use a document or endpoint from one of the three blind-only industries. All blind documents must be frozen as SHA-256-verified snapshots before the blind reserve is opened.

### 6.5 Machine-readable registry fragment

**[REC]** The full wave-3 preregistration must be committed as a JSON file at `experiments/foundry_wave3/preregistration.json` before any paid session. The schema below is the required structure; values marked `[TBD]` are filled immediately before first dispatch.

```json
{
  "schema_version": "wave3-v1",
  "wave": 3,
  "status": "frozen-before-dispatch",
  "cumulative_hard_cap_usd": 200.0,
  "prior_waves_settled_usd": 5.050739,
  "wave3_operational_stop_cumulative_usd": 185.0,
  "runtime": {
    "provider": "microsoft-foundry-responses-api",
    "deployment": "gpt-5-mini",
    "effort_levels": ["minimal", "medium"],
    "verbosity_levels": ["low", "medium"]
  },
  "required_null": {
    "type": "exact-tokenizer-harness-measurement",
    "components": ["system_prompt", "tool_schemas", "instruction_prefix"],
    "method": "responses-api-input-tokens-count-endpoint",
    "note": "POST /v1/responses/input_tokens with zero-content task; tiktoken o200k_base covers plain-text tokens offline only; tool-schema formatting overhead requires this API call (source: https://developers.openai.com/api/docs/guides/token-counting)",
    "stores_as": "fixed_overhead_tokens",
    "harness_sha256": "[TBD]",
    "confirmation_null_probes": 2,
    "required_before_phase": "phase_1"
  },
  "feature_grammar": {
    "quote_time_only": true,
    "predictors": [
      "task_payload_tokens",
      "output_bound_tokens",
      "output_spec_units",
      "k_declared",
      "effort_level",
      "cache_warm",
      "brick_effect_coded",
      "composition_arity"
    ],
    "fixed_constant_not_predictor": "fixed_overhead_tokens",
    "rejected_predictors": [
      "k_observed", "a_observed", "big_t_class_label",
      "reasoning_tokens", "response_size_observed", "is_multiplicative"
    ]
  },
  "primary_targets": ["output_tokens", "k_free"],
  "secondary_targets": ["input_tokens", "total_tokens"],
  "excluded_from_promotion_gates": ["model_call_count", "tool_call_count"],
  "interaction_hierarchy": [
    {
      "level": 1,
      "terms": ["main_effects"],
      "gate": "ridge_ci95_excludes_zero_for_at_least_one_non_brick_predictor"
    },
    {
      "level": 2,
      "terms": ["task_payload_tokens:brick_type", "k_declared:output_bound_tokens"],
      "precondition": "level_1_gate_passed"
    },
    {
      "level": 3,
      "terms": ["task_payload_tokens:effort_level", "cache_warm:k_declared"],
      "precondition": "level_2_gate_passed"
    },
    {
      "level": 4,
      "terms": ["three_way_interactions_if_any_level3_ci95_excludes_zero"],
      "precondition": "level_3_gate_passed"
    }
  ],
  "acceptance_oracle": {
    "prose_shapes": {
      "format": "semantic_fact_checklist",
      "schema_key": "required_facts[{field, pattern}]",
      "pass_condition": "all_present",
      "prohibited": ["exact_string_equality", "whole_title_match"]
    },
    "structured_shapes": {
      "format": "json_schema_validation",
      "pass_condition": "schema_valid AND all_required_keys_present AND all_enums_in_vocabulary AND all_cited_ids_in_input"
    }
  },
  "tail_claims": {
    "deferred_until_independent_groups": 30,
    "wave2_groups": 9,
    "permitted_before_threshold": ["signal_gates", "main_effect_ci95", "grouped_cv_mae"]
  },
  "repair_protocol": {
    "trigger": "shape_acceptance_rate_below_2_over_3",
    "permitted_repairs": [
      "oracle_pattern_fix_within_semantic_not_literal_constraint",
      "prompt_wording_fix_within_instruction_template_bounds"
    ],
    "prohibited_repairs": [
      "adding_rows_to_signal_gate",
      "changing_gate_threshold",
      "changing_acceptance_criterion_type"
    ],
    "repair_sessions_per_failed_shape": 3,
    "max_repair_shapes": 4,
    "post_repair_failure_action": "pause_and_require_approval"
  },
  "grouped_cv": {
    "outer_grouping_variable": "shape_source",
    "inner_grouping_variable": "replicate_within_shape",
    "replicates_per_shape": 3,
    "minimum_valid_sessions_per_shape": 2,
    "telemetry_complete_required": true
  },
  "blind_reserve": {
    "session_count": 24,
    "open_condition": "after_model_form_alpha_oracle_all_frozen",
    "open_once": true,
    "industries": ["cybersecurity_it_operations", "supply_chain_manufacturing", "legal_international_regulatory"],
    "source_snapshots_sha256": "[TBD_before_opening]"
  },
  "training_sessions": "[TBD_randomized_order_before_phase_1]"
}
```

### 6.6 Session allocation, repair-block protocol, and budget

**[REC]** The session structure below replaces the Part 3 session count estimate. Phase gates are defined before any session in that phase runs; the repair block is pre-authorized but conditional; the blind reserve is opened exactly once after all model-form decisions are frozen.

**Phase 0: Harness baseline and null probes (2 sessions)**

- Step 0a: Offline tokenization of harness components. Zero API cost. Stores `fixed_overhead_tokens` and `harness_sha256`.
- Step 0b: 2 null probes (same harness, trivial payload, `cache_warm = 0`). Verifies tokenizer estimate against billed input.
- **Stop/go gate:** `|billed_input − (fixed_overhead_tokens + trivial_payload_tokens)| ≤ 2%`. If outside 2%, investigate harness tokenization before proceeding.

**Phase 1: Atomic correlate — primary mechanism (54 sessions)**

- 3 payload levels (≈ 500 B, 5 KB, 50 KB) × 3 k_declared levels (1, 2, 3) × 2 arms (snapshot, live) × 3 replicates = 54 sessions.
- `effort_level = 0` (minimal/low), `cache_warm` randomized within each replicate block.
- `declared_fetch_kib` computed from frozen snapshot before each live session.
- **Stop/go gate:** at least 2/3 replicates per shape have `telemetry_complete = true`; semantic acceptance ≥ 2/3 per shape; within-shape MAD identifiable (at least 3 valid sessions per arm).

**Phase 2: Atomic diagnose — uncertain-k mechanism (36 sessions)**

- 3 payload levels (≈ 500 B, 5 KB, 50 KB) × 2 effort levels × 3 replicates = 18 sessions per effort level; 36 total.
- `cache_warm` randomized within each replicate block.
- Primary outcome: distribution of `k_free = observed_calls − k_declared`; `k_declared = 1` for all diagnose sessions.
- **Stop/go gate:** ≥ 1 shape shows `k_free > 0` in at least 2 sessions (confirms uncertain-k mechanism is non-degenerate).

**Phase 3: Atomic provision — secondary k_free mechanism (12 sessions)**

- 2 payload levels (≈ 5 KB, 50 KB) × `k_declared ∈ {2, 3}` × 3 replicates = 12 sessions.
- `effort_level = 0`, `cache_warm = 0` (cold only; small scale does not require the full crossing).
- **Stop/go gate:** ≥ 1/3 sessions per shape show `k_free > 0`; if none, provision is reclassified as fixed-k and dropped from the k_free estimation.

**Phase 4: Crossed effort × cache factors (36 sessions)**

- 2 effort levels × 2 cache states × 3 reference bricks (`correlate` at medium payload, `summarise` at medium payload, `transform` at 8-field) × 3 replicates = 36 sessions.
- This phase estimates the `effort_level` and `cache_warm` coefficients and their interaction with brick type.
- **Stop/go gate:** `effort_level` or `cache_warm` ridge CI95 excludes zero for at least one brick (otherwise the crossing added no information and the gate passes as negative result, which is also valid).

**Phase 5: Composition training (24 sessions)**

- 6 composition recipes × 2 payload levels × 2 replicates = 24 sessions.
- Recipes must include: (a) correlate + summarise, (b) diagnose + report, (c) fetch-external + evaluate, (d) plan + correlate, (e) transform + validate, (f) a four-brick composition including at least one k_declared > 1 brick.
- **Stop/go gate:** additive composition model beat constant on grouped-CV MAE by ≥ 15%.

**Repair block: conditional, pre-authorized (up to 12 sessions)**

- Triggered by Phase 1, 2, or 3 gate failure on acceptance (C-11).
- 3 sessions per failed shape; maximum 4 shapes = 12 sessions.
- Repair actions limited to oracle pattern fix or prompt wording fix (C-11 prohibited list applies).
- If the repaired gate still fails: pause and require approval before Phase 4.

**Blind reserve (24 sessions)**

- Opened once, after all of the following are frozen: model form, interaction level, alpha selection rule, acceptance oracle patterns, calibration method, arm-specific calibration rule.
- 8 sessions per blind industry (cybersecurity, supply chain, international regulatory).
- Each session uses a document frozen as a SHA-256-verified snapshot before opening.

**Session summary and budget:**

| Phase | Sessions | Cost range per session (USD at safety rates) | Estimated phase cost (USD) |
|---|---|---|---|
| Phase 0 — Null probes | 2 | 0.004–0.010 | 0.02 |
| Phase 1 — Atomic correlate | 54 | 0.04–0.60 (mixed payload) | 16.00 |
| Phase 2 — Atomic diagnose | 36 | 0.04–0.30 | 7.20 |
| Phase 3 — Atomic provision | 12 | 0.10–0.50 | 3.60 |
| Phase 4 — Crossed effort × cache | 36 | 0.08–0.60 | 9.00 |
| Phase 5 — Compositions | 24 | 0.15–1.20 | 10.80 |
| Repair block (conditional) | 0–12 | 0.04–0.60 | 0–3.60 |
| Blind reserve | 24 | 0.10–0.80 | 9.60 |
| **Wave-3 total** | **188–200** | | **56.22–59.82** |
| **Cumulative (wave-2 + wave-3)** | | | **61.27–64.87** |

**[FACT]** Per-session cost estimates are derived from the actual wave-2 settled ledger: control sessions USD 0.00402; transform-8 USD 0.0205; summarise-10240 USD 0.107–0.115; fetch-live (small document) USD 0.055–0.275 (`runs/20260826_1627_wave2/budget.json`). The one incomplete 102 KB Summarise session was reserved at USD 2.356 under the pre-request reservation rule; excluding that outlier, the largest completed session cost USD 0.597. Safety rates are USD 20/M input and USD 200/M output, not provider billing quotes.

**[REC]** The operational stop is set at cumulative USD 190, leaving a USD 10 buffer to the hard cap for delayed metering. These amounts use the conservative safety rate card and are not provider-price or billing claims. The constraint is not budget but the quality gates in C-03 and C-10.

### 6.7 Hierarchical interaction preregistration

**[REC]** The four-level interaction hierarchy in the machine-readable registry (§6.5) is expanded here with exact term definitions and promotion criteria.

**Level 1 — Main effects (base model):**

```
output_tokens ~ α + β₁·task_payload_tokens + β₂·output_bound_tokens
              + β₃·output_spec_units + β₄·k_declared + β₅·effort_level
              + β₆·cache_warm + Σγⱼ·brick_j + δ·composition_arity
```

Promotion gate: grouped-CV MAE improvement over constant ≥ 20%; ridge CI95 for at least one of {β₁, β₂, β₃, β₄} excludes zero (these are the size-driver predictors; brick effects are allowed to shrink to zero).

**Level 2 — Size × mechanism and call × output interactions:**

Add `task_payload_tokens × brick_type` (slope of input-size effect may differ by brick) and `k_declared × output_bound_tokens` (call count interacts with declared output cap — a multi-call session with a tight output bound may show different scaling than one with a loose bound).

Promotion gate: at least one Level 2 interaction term's ridge CI95 excludes zero in grouped CV.

**Level 3 — Effort × size and cache × call interactions:**

Add `task_payload_tokens × effort_level` (reasoning depth may amplify the payload slope) and `cache_warm × k_declared` (cache benefit is largest for multi-call sessions where the harness is re-read multiple times).

Promotion gate: same as Level 2.

**Level 4 — Three-way interactions (if Level 3 passed):**

The only pre-authorized three-way term is `task_payload_tokens × k_declared × effort_level`. No other three-way terms are permitted. This term captures the case where a multi-call, high-effort session with a large payload is disproportionately expensive.

**[REC]** Interaction terms are added cumulatively; a lower-level term is never dropped when a higher-level term is added. The ridge alpha is re-selected by grouped CV after each level is added.

### 6.8 Tail-claim deferral protocol

**[REC]** Tail coverage claims require at least 30 independent shape groups, as specified in C-10. The following table tracks the group count through the wave-3 phases; tail claims are unlocked only after the blind reserve reaches the threshold.

| After phase | Cumulative independent groups | Tail claims permitted? |
|---|---|---|
| Wave-2 complete | 9 | No |
| Phase 1 complete | 9 + 18 = 27 (3 payload × 3 k_declared × 2 arms = 18 new groups) | No |
| Phase 2 complete | 27 + 6 = 33 (3 payload × 2 effort = 6 new diagnose groups) | **Yes, after Phase 2** |
| Phase 3 complete | 33 + 4 = 37 (2 payload × 2 k_declared = 4 provision groups) | Yes |
| Phase 5 complete | 37 + 6 = 43 (6 composition groups) | Yes |
| Blind reserve opened | 43 + 3 = 46 (3 blind-industry groups, each treated as one group regardless of session count within) | Yes |

**[REC]** When the 30-group threshold is reached (after Phase 2), tail calibration is computed as follows: calibration radius = q(0.90) or q(0.95) of out-of-fold residuals from training sessions only; computed separately for live-arm sessions (`k_declared > 1` or `k_free > 0`) and no-arm sessions; cross-arm calibration is prohibited. Coverage is reported as the fraction of held-out sessions where actual output_tokens ≤ predicted + calibration_radius. Pre-registered thresholds: `p90_coverage ≥ 0.80`; `p95_coverage ≥ 0.85`.

**[REC]** Any tail claim reported before the 30-group threshold must be explicitly labeled "pre-threshold estimate subject to revision" and must include the current group count. The wave-2 analysis artifact's interval coverage figures (87.0% at both p90 and p95) are pre-threshold estimates from 9 groups and must not be reported as validated calibration results (`runs/20260826_1627_wave2/analysis.json`).

| Source | Official URL | Grounded claim |
|---|---|---|
| NIST SP 800-61 Rev.2 §3.2.3 | https://csrc.nist.gov/pubs/sp/800/61/r2/final | Incident analysis phase as iterative root-cause identification; grounds `diagnose` brick |
| ITIL 4 Problem Management | https://www.axelos.com/certifications/itil-service-management | Iterative root-cause investigation practice distinct from monitoring and remediation |
| OASIS TOSCA 2.0 lifecycle | https://docs.oasis-open.org/tosca/TOSCA/v2.0/os/TOSCA-v2.0-os.html | Multi-stage lifecycle with health checks; grounds `provision` k_free |
| CISA KEV catalog (blind pool) | https://www.cisa.gov/known-exploited-vulnerabilities-catalog | Blind-pool cybersecurity source |
| MITRE ATT&CK bundle (blind pool) | https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json | Blind-pool STIX-form cybersecurity source |
| EUR-Lex web services (blind pool) | https://eur-lex.europa.eu/content/tools/webservices/ | Blind-pool international regulatory source |
| GS1 Application Identifiers (blind pool) | https://ref.gs1.org/ai/GS1_Application_Identifiers.jsonld | Blind-pool supply-chain source |
| Wave-2 budget ledger | runs/20260826_1627_wave2/budget.json | Per-session conservative safety amounts used to calibrate Phase 6.6 reservations; not provider billing |

## Part 7. Completed repair gate

The v2 repair block is frozen at
`runs/20260827_1152_wave3/`. The earlier
`runs/20260827_1145_wave3/` canary is retained as an aborted instrumentation
and prompt-validation record; its safety amount is carried into the cumulative
ledger rather than erased.

The completed block stopped at the preregistered 36 logical sessions. It
recorded 84,622 provider-reported tokens: 77,481 input, 7,141 output, 3,648
reasoning (a subset of output), and 13,184 cached input tokens. Structural
acceptance was 30/36 and semantic/overall acceptance was 24/36.

Three repair findings block expansion:

1. Azure returned `This model is not supported by Responses API` from
   `/responses/input_tokens` for the fixed `gpt-5-mini` deployment. All rows
   retain local `o200k_base` payload counts and are marked ineligible for an
   exact tokenizer/harness-baseline fit; no provider count was imputed.
2. Six medium/medium requests were provider-incomplete at the frozen output
   cap: four Summarise cases and two null controls. Their usage was preserved,
   and content acceptance remains unset rather than failed.
3. All six SEC Fetch outputs were structurally valid but miscounted the 117
   frozen USD observations. This is a substantive task-quality failure, not an
   oracle-format failure.

Fetch replication repaired the wave-2 identifiability gap. Within each
source/arm, three total-token observations had MAD 0. Snapshot/live median
differences were 513 tokens for SEC, 313 for USAspending, and 272 for CMS.
Assigned cache warmth did not guarantee a cache hit: only 4/18 warm-assigned
rows reported cached input.

The cumulative conservative safety ledger is USD 14.43254, below the USD 190
operational stop and USD 200 hard cap. This is not an Azure billing claim.
Per C-03, the 144-session training campaign and 48-session untouched blind
reserve are **not authorized** from this repair result.
