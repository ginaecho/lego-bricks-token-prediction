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
| `waves/wave2.json` | Frozen wave-2 manifest: selected trials, instruction fingerprints, builders, failures, protocol deviations and evaluation gates |
| `models/` | Construction-token model artifact and its validation/test report |
| `azure_rate_cards.json` | Versioned, sourced reference price assumptions |
| `real_use_cases.json` | Fourteen user-supplied case references and unverified post-hoc mappings |
| `coverage.json` | Measured versus unmeasured coverage; no hidden claims of exhaustive training |
| `TODO.md` | Remaining measurement, training, evaluation, Studio integration and feedback-learning plan |

The first wave covers **every standalone type** and a fresh two-, three- and
four-function integrated build. Other design entries remain unmeasured; the
combination design is not a table of invented token labels. Memberships use
distinct basic functionalities. Multiple types from the same family and every
possible execution-order permutation are not claimed as exhaustively measured.

The first saved wave contains **19 measured builds and 2,921,703 consumed tokens**.
Its split is 14 training, zero validation and five test records; wave 1 alone
did not train or promote a construction-token model. See the
[remaining-work plan](TODO.md) for dependencies and acceptance requirements.
Per-request usage traces are embedded in each measured record's
`build_token_usage.events`; verification outputs and hashes are in
`build_evidence`. These exports are not full conversations or the raw session
database. Git preserves the exact bytes of `builds/` artifacts so their recorded
hashes remain reproducible across platforms.

## Wave 2 and the first construction-token model

Wave 2 froze **120 trials** before dispatch (seed 20260923, builder model
`gpt-6-astra`, instructions `builder-instructions-v2`): 16 singles, 33 pairs,
38 triples and 33 quadruples, including six repeated memberships. All 120
builders passed independent runtime verification of their tests and CLI; none
failed. They consumed **18,247,910 tokens** (mean 152,066 per build).

| Functionalities | Builds | Mean tokens | Median | Range |
| ---: | ---: | ---: | ---: | --- |
| 1 | 16 | 125,407 | 121,160 | 114,902-160,476 |
| 2 | 33 | 133,225 | 133,170 | 123,748-157,764 |
| 3 | 38 | 157,479 | 146,649 | 132,202-243,236 |
| 4 | 33 | 177,599 | 172,329 | 117,187-231,283 |

Repeated memberships varied by a mean coefficient of variation of 5.6%. In 19
trials the builder also wrote an auxiliary `docs\My_prompt.txt` in its staging
directory; these are recorded under `protocol_deviations` and were not imported
into `builds/`. Staging copies are kept in `runs/20260923_1100_wave2/`.

`examples/train_construction_model.py` fits ridge regressions on log input and
log output tokens from pre-build features only (staffing and months excluded),
selecting the penalty by grouped cross-validation over membership groups.
Combined with wave 1: 92 training, 20 validation and 27 test records.

| Split | n | MAE (tokens) | MAPE | Training-mean baseline MAE | Interval coverage (80% target) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 20 | 19,961 | 11.1% | 25,597 | 90.0% |
| Test (evaluated once) | 27 | 16,567 | 11.7% | 18,032 | 92.6% |

The validation gates (MAPE at most 20% and beating the baseline MAE) passed.
Test MAPE by size: 8.4% (1), 10.6% (2), 9.2% (3), 16.9% (4). The test gain over
the training mean is modest (about 8% lower MAE) and the model overpredicted on
average (+10,188 tokens). The selected penalty (alpha 100) is the largest value
in the fixed grid, indicating weak per-feature signal. These results hold only
for bounded Python CLI builds by one builder model.

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
python -B -m examples.train_construction_model --wave wave2
python -B -m pytest tests\test_build_simulations.py -q
```

Import only **completed** builder sessions with exact IDs recorded in the
campaign. Re-importing the same record refreshes its evidence, not a second
independent sample. If a builder receives a repair or clarification turn,
its additional recorded usage belongs to that same build. A missing usage
record is an error; do not substitute an LLM guess, text-length estimate,
tokenizer count of visible output, or unrelated session total.
