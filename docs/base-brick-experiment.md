# Base-brick business-case experiment

This is the first step from a document-only token model toward the **AI Unit
Economics Challenge**: connect spend to accepted work, quote a credible range
before dispatch, reserve for expensive tails, and allocate actual cost after
dispatch.

It does not replace the existing SEC campaign. It adds the missing experiment
contract around it: sourced business cases, frozen acceptance criteria,
auditable execution records, and outcome-based economics.

## What was built

[`experiments/business_cases/cases.jsonl`](../experiments/business_cases/cases.jsonl)
contains 12 reproducible cases:

| tier | cases | purpose |
|---|---:|---|
| atomic | 9 | one case for every base brick |
| composite | 3 | two- and three-way combinations |
| held out | 1 | a composition marked unavailable to fitting |

The catalog covers six domains: finance operations, financial reporting,
customer support, legal operations, management reporting, and procurement.
Every case records its primitive counts, size driver, context bytes, prompt,
acceptance rules, and official source URLs.

## Why these are real business shapes

The fixtures are deterministic and created locally, but the workflows and data
shapes are anchored to primary sources:

- SEC filing search and XBRL APIs:
  [EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
  and [company facts](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json)
- CFPB complaint routing and debt-collection work:
  [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
  and [debt-collection resources](https://www.consumerfinance.gov/compliance/compliance-resources/other-applicable-requirements/debt-collection/)
- federal travel and expense controls:
  [Federal Travel Regulation in eCFR](https://www.ecfr.gov/current/title-41/subtitle-F)
- federal award records:
  [official USAspending API](https://github.com/fedspendingtransparency/usaspending-api)
- public financial data and quality standards:
  [Treasury Fiscal Data](https://fiscal.treasury.gov/data/) and
  [Data.gov standards](https://resources.data.gov/standards/)

The repository does **not** claim that these organizations ran the exact local
prompts. The sources establish external validity; embedded facts and frozen
oracles establish reproducibility.

## Acceptance proof

Three independent agents executed four cases each. Their exact outputs and
verdicts are committed in
[`proof_results.jsonl`](../experiments/business_cases/proof_results.jsonl).

| result | count |
|---|---:|
| accepted | 11 |
| rejected | 1 |
| acceptance rate | 91.7% |

The rejected remediation case is retained. It calculated the correct percentage
but omitted one fact required by the frozen oracle. Keeping the miss prevents
post-hoc criteria from turning the experiment into self-grading.

These are **acceptance-only proofs**. The agent runtime did not expose provider
token counters, so every record deliberately omits token usage. They prove that
the catalog can dispatch and independently score work; they do not measure
token cost and cannot enter model fitting or unit-economics calculations.

## Measurement and economics contract

[`token_yield/runner.py`](../token_yield/runner.py) provides the live boundary.
A provider adapter supplies a fresh agent result with measured input, output,
and total tokens. The runner binds that spend to the acceptance verdict and
records:

- model and model version;
- timestamp and duration;
- prompt and source-manifest SHA-256 hashes;
- tool uses, context bytes, and primitive counts; and
- whether the case is held out.

[`token_yield/business_cases.py`](../token_yield/business_cases.py) converts a
measured result directly to the existing composition model's `Run` type.
Rejected attempts remain in the spend cohort: filtering them out would
systematically underquote the cost of accepted work.

[`token_yield/economics.py`](../token_yield/economics.py) then reports:

- acceptance rate and attempts per accepted outcome;
- tokens and cost per accepted outcome, including failed attempts;
- empirical p50, p90, p95, and p99 attempt sizes;
- per-attempt tail reserves and accepted-outcome reserves under an explicitly
  labeled IID geometric retry model; and
- actual-cost chargeback by cost center.

No monetary result is produced unless the caller supplies the actual billing
rate. The accepted-outcome percentile assumes independent attempts with a
stable acceptance probability; production workflows should replace that
assumption with observed retry sequences when available.

## A stricter reading of existing accuracy

The existing 35 fitted SEC runs still select the brick model over the candidate
pre-run forms:

| diagnostic | result |
|---|---:|
| total-token leave-one-out error | 2.55% |
| error relative to work above agent startup | 18.73% |
| skill over a constant model on that variable work | 64.47% |
| post-run `bytes + units + tool uses` rival | 3.19% total-token error |

The 2.55% number is valid but dominated by the roughly 30k-token startup
constant. The excess-work metric is the more demanding test of whether the
LEGO features explain variable cost. Tool-use forms are now reported as
diagnostic rivals but cannot be selected for a pre-dispatch quote because tool
uses are unknown at scoping time.

## What remains to prove

The next paid measurement wave should execute the 12 cases through a provider
adapter with at least two replicates per atomic shape and should add:

1. crossed context-by-unit ladders so bytes and unit counts are identifiable;
2. at least 15 held-out compositions;
3. real split-arm runs comparing one batched agent with one agent per brick;
4. correct-versus-random-versus-adversarial decomposition tests; and
5. a time-ordered production holdout for the feedback loop.

Until the split arm is run, the existing 64-71% batching figure is a
**model-based estimate**, not a measured saving.

## Run it

```bash
python -m examples.business_case_demo
python -m pytest tests/test_business_cases.py tests/test_business_case_demo.py -q
```

The demo will print the catalog and acceptance proofs, then stop before token
economics because the committed proofs intentionally contain no usage data.
