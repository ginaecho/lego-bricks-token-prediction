# Measured software-build simulations

Here, **simulation means an agent actually constructs a runnable functionality
or integrated combination in an isolated build session**. A data point is not
a model-generated token guess. It is not the number of tokens in the resulting
source code, and it is not the token consumption of operating the built feature.

The pre-existing Studio model measures **workload execution**. This corpus
measures **software construction**. Their targets must never be pooled or
relabeled as one another.

## Files

| Path | Contents |
| --- | --- |
| `index.html` | Read-only browser view of all records and input features; no server or provider calls |
| `functionality_registry.json` | Seven independently defined basic functions and 16 types |
| `feature_dictionary.json` | Input feature names, units, availability, labels and reward definition |
| `combination_design.json` | 16 singles, 109 pairs, 410 triples and 920 four-function memberships; measured coverage is explicit |
| `data_points/` | One JSON record per selected build or real-use-case reference |
| `data_points.csv` | Readable overview: staffing, duration, measured tokens, estimated cost and pseudo scores |
| `builds/<point>/` | Actual implementation, tests, example input and build manifest |
| `campaign.json` | Exact builder agent IDs, measurement protocol and selected wave |
| `azure_rate_cards.json` | Versioned, sourced reference price assumptions |
| `real_use_cases.json` | Fourteen user-supplied case references and unverified post-hoc mappings |
| `coverage.json` | Measured versus unmeasured coverage; no hidden claims of exhaustive training |
| `TODO.md` | Remaining measurement, training, evaluation, Studio integration and feedback-learning plan |

The first wave covers **every standalone type** and a fresh two-, three- and
four-function integrated build. Other design entries remain unmeasured; the
combination design is not a table of invented token labels. Memberships use
distinct basic functionalities. Multiple types from the same family and every
possible execution-order permutation are not claimed as exhaustively measured.

The saved wave contains **19 measured builds and 2,921,703 consumed tokens**.
Its split is 14 training, zero validation and five test records; no
construction-token model has been trained or promoted. See the
[remaining-work plan](TODO.md) for dependencies and acceptance requirements.
Per-request usage traces are embedded in each measured record's
`build_token_usage.events`; verification outputs and hashes are in
`build_evidence`. These exports are not full conversations or the raw session
database. Git preserves the exact bytes of `builds/` artifacts so their recorded
hashes remain reproducible across platforms.

## Input features

| Group | Fields / meaning |
| --- | --- |
| Identity | `basic_functionality_ids`, `basic_functionalities`, ordered `types`, `functionality_count` |
| Staffing | `staff_data_scientist`, `staff_architect`, `staff_consultant`, `staff_software_engineer`, in **FTE** |
| Duration | `estimated_months_to_finish`, a declared human-delivery planning assumption |
| Integration | `integration_edge_count`, `shared_schema_count`, `shared_validation_layer_count`, plus the ordered handoff graph |
| Scope | `planned_acceptance_case_count`, `artifact_kind`, `llm_integration` |
| Catalog encoding | Sixteen `has_type_*` indicators and seven `planned_operation_*` counts |

Staffing and delivery months are scenario assumptions, **not observed staffing,
agent wall-clock duration, or demonstrated causal drivers of token use**. The
builders did not receive randomized staffing interventions. Feature encoding
describes the requested build scope; this first wave is not a prospectively
controlled staffing experiment.

Actual test counts, final file sizes, observed token usage, costs and outcome
scores are recorded after construction and must not leak into pre-build token
predictor inputs. Repeated or reordered builds of the same type membership stay
in one split group. The initial wave is too small for reliable held-out
generalization claims; no construction-token predictor is trained or promoted
merely because the files exist.

## What is actually measured

Each builder starts from an empty implementation and may share schemas,
validation, retrieval and execution paths **inside its integrated build**.
It does not copy standalone implementations or sum standalone token labels.
The bounded scope is a working Python CLI reference implementation with
validation, executable tests and optional injected model interfaces.

These are not claims of complete enterprise deployment. Deterministic search,
ranking or extraction baselines and fixture-tested LLM interfaces are identified
in each manifest. A deployed external model, production infrastructure, customer
connectors and clinical/compliance certification were not measured.

The collector reads authoritative Copilot `assistant_usage_events` for the
**exact parent session and builder agent ID**. It never reads prompt content,
credentials or other sessions. It checks reported counters against the runtime's
token-type breakdown and stores the event identifiers and counts.

```text
AIC_input  = sum(reported input_tokens across the isolated build session)
AIC_output = sum(reported output_tokens across the isolated build session)
AIC_total  = AIC_input + AIC_output
```

Input already includes cache reads and cache writes: they are not added twice.
Reasoning is auxiliary output detail, not an additional token sum. The measured
scope includes builder instructions/context, tool interactions, coding,
tests/repairs and the builder's final report. Parent orchestration and the future
deployed workload are excluded. Consequently, these are aggregate consumed
tokens, not the number of unique words or tokens in a code file.

The data collector independently reruns each implementation's tests and CLI and
records artifact SHA-256 hashes before accepting its measured point. A failed
or missing build is not silently replaced with a numerical prediction.

## Cost is a separate, counterfactual calculation

The observed builder model is recorded from telemetry. The build wave uses
authenticated subscription subagents; it does **not** pretend they ran GPT-5.4,
DeepSeek, or the Foundry endpoint when they did not.

For an Azure target model, the collector prices each observed model request at
that model's reference input/output rates. Context tiers apply per request,
not to the sum of all requests in a build. It reports both:

1. A no-cache scenario.
2. A scenario preserving the observed cache-read profile, where a cached rate
   has been verified.

These hold token volume constant. Different model tokenizers, reasoning,
quality, cache eligibility and repair counts can change actual consumption.
The values are **not actual Azure bills or measured usage of the target model**.
Actual subscription USD cost remains unknown, not zero.

GPT-5.4 and GPT-5.4 Pro reference rates cite the Microsoft-hosted pricing answer
in `azure_rate_cards.json`. They are published reference rates, not a live tenant
quote. No DeepSeek price or API usage is invented.

## Pseudo client outcomes and reinforcement-learning reward

All generated outcomes have `is_pseudo: true`. They are clearly fictional,
including when attached to a real-use-case reference; they do not assert a
customer endorsement or measured financial return.

Scores are **0-5**: unacceptable, weak, partial, baseline, strong, exceptional.

| Evaluation dimension | Weight |
| --- | ---: |
| Client satisfaction | 30% |
| Recognized impact | 25% |
| ROI satisfaction | 20% |
| Future impact potential | 10% |
| Delivery quality | 15% |

`reward = sum(weight * score / 5)`, yielding a value from 0 to 1.
`roi_satisfaction` is an ordinal rating; it is not financial ROI percentage.
Pseudo AI value, net impact and ROI percentage have separate assumed benefit
and cost fields. Verified client ROI remains `null`.

These versioned scores can support **synthetic contextual-bandit experiments**,
not foundation-model reinforcement fine-tuning or production ROI claims.
No action propensities were prospectively logged for this enumerated build wave,
so it cannot establish off-policy improvement. It does not automatically promote
a real recommendation policy or silently convert the older feedback interface's
1-5 scales to this 0-5 schema.

## Real use cases are separate evidence

Case references cover finance, healthcare and manufacturing. They were supplied
by the user; the referenced slide decks were not independently retrieved here.
Catalog mappings are **post-hoc hypotheses**, never the source of the basic
functionality taxonomy or simulation combinations. Capability gaps, such as
forecasting, Text2SQL or medical imaging, are explicit rather than forced into
the available types.

No construction-token measurements, actual staffing or verified client ROI
were supplied for those organizations. Those fields remain unknown and the
records are excluded from build-token training. Their clearly labeled pseudo
outcome scenarios remain available for feedback-interface experiments.

## Reproduce and extend

From the repository root:

```powershell
python -B -m examples.build_simulation_corpus initialize
python -B -m examples.build_simulation_corpus import `
  --copilot-db <path-to-your-local-session-store.db> `
  --ids single_interests
python -B -m examples.build_simulation_corpus summarize
python -B -m pytest tests\test_build_simulations.py -q
```

Import only **completed** builder sessions with exact IDs recorded in the
campaign. Re-importing the same record refreshes its evidence, not a second
independent sample. If a builder receives a repair or clarification turn,
its additional recorded usage belongs to that same build. A missing usage
record is an error; do not substitute an LLM guess, text-length estimate,
tokenizer count of visible output, or unrelated session total.
