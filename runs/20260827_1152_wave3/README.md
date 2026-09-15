# Wave 3 repair gate

**Status:** complete and paused. This run contains exactly 36 logical repair
sessions and is calibration/instrumentation evidence only. It is prohibited
from final model selection.

## Frozen allocation

| family | sessions |
|---|---:|
| Tokenizer/null confirmation | 2 |
| Summarise | 12 |
| Fetch (snapshot/live) | 18 |
| Cache/drift controls | 4 |

The source pool spans SEC and CFPB-adjacent financial evidence, eCFR,
openFDA, USAspending, and CMS. Inputs, source hashes, case order, feature
registry, oracle hashes, effort/verbosity regimes, and output caps are preserved
under `frozen_inputs/`.

## Gate result

| result | value |
|---|---:|
| Logical sessions | 36 |
| Provider-reported total tokens | 84,622 |
| Input tokens | 77,481 |
| Cached input subset | 13,184 |
| Output tokens | 7,141 |
| Reasoning subset | 3,648 |
| Structural acceptance | 30/36 |
| Semantic/overall acceptance | 24/36 |
| Provider-incomplete responses | 6/36 |
| Conservative cumulative safety ledger | USD 14.43254 |

The selected Azure `gpt-5-mini` deployment rejected
`/responses/input_tokens` as unsupported. No exact provider count was imputed:
all rows retain local `o200k_base` diagnostics, have null exact
`provider_input_tokens` and `fixed_overhead_tokens`, and are ineligible for
exact-baseline fitting.

All six medium-effort incomplete responses reached the frozen output cap:
four Summarise cases and two controls. All six SEC Fetch cases were
structurally valid but failed the semantic oracle by miscounting the 117 frozen
USD records.

Three repetitions per Fetch source/arm made within-shape variation
identifiable: every total-token MAD was 0. Live minus snapshot median totals
were 513 tokens for SEC, 313 for USAspending, and 272 for CMS. Only 4/18
warm-assigned cases showed cached input, so assigned warmth is not a reliable
proxy for observed cache behavior.

The safety ledger uses conservative authorization rates, not Azure prices or
reconciled billing. The 144-training/48-blind campaign remains blocked and was
not started.
