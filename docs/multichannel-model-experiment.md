# Multichannel quote model — staged preregistration

**Status:** Stage 1 implemented; no promotion claim is made by this document.
**Scope:** the LEGO objective remains unchanged: decompose a project into
reusable task bricks at quote time, predict its resource range before dispatch,
and reconcile the quote to measured, independently accepted work afterwards.
This protocol expands the quoted outcome from one total-token number into a
coherent multichannel distribution.

The completed Wave 3 repair block at
`runs/20260827_1152_wave3/` is **calibration and instrumentation evidence
only**. Its 36 records are prohibited from final model selection, tail
calibration, and campaign authorization, including after any reanalysis. In
particular, it found that assigned cache warmth was not an observed cache hit
(only 4 of 18 warm assignments were cached), the provider input-count endpoint
was unavailable, six medium responses were output-capped and incomplete, and
six SEC Fetch outputs failed the semantic oracle. These are design inputs, not
training wins.

## 1. Confirmatory question and estimand

For a request represented solely by quote-time data \(X\), estimate the
conditional distribution of resource use and work outcomes:

\[
 p(I,C,N,R,L,Z,B,Q,A \mid X).
\]

Here `I` is actual input tokens, `C` cached input tokens, `N` non-reasoning
output tokens, `R` reasoning tokens, `L` end-to-end latency, `Z` response
bytes, `B` unplanned within-attempt branching, `Q` retry dispatched under the
frozen retry policy, and `A` semantic acceptance. All are observed **after**
dispatch, hence are targets and never columns of \(X\). `observed_model_calls`
and `observed_tool_calls` are likewise outcomes, not substitutes for
`k_declared`.

Every quote reports `p50`, `p90`, and `p95` for each non-negative amount and
for derived tokens and safety dollars, plus event probabilities
`P(cache_hit)`, `P(reasoning > 0)`, `P(B)`, `P(Q)`, and `P(A)`. A probability
is reported with its grouped out-of-fold calibration uncertainty, rather than
rounded into a deterministic promise.

### Target identities

The following identities are validation invariants and are reconstructed on
every simulated draw, not only at the forecast summary:

| Quantity | Definition and treatment |
|---|---|
| Cache hit | \(H_C=\mathbb{1}[C>0]\); report `P(H_C=1)` and `C \| H_C=1`. Cache assignment is a quote-time treatment, whereas the hit and amount are targets. |
| Input | \(0\le C\le I\). `C` is an input-detail subset, so `I` is **not** increased by `C`. |
| Output | \(O=N+R\), where \(N=O-R\ge0\). Reasoning is a subset of reported output; it is never added a second time to token totals. |
| Total tokens | \(T=I+O=I+N+R\). A direct total-token model is diagnostic only and cannot replace the channel reconstruction for quote publication. |
| Branching | \(B=\mathbb{1}[\texttt{observed_model_calls}>\texttt{k_declared}\ \lor\ \text{unplanned bounded tool continuation}]\). Observed calls remain in the outcome ledger only. |
| Retry | `Q=1` when a further logical attempt is actually dispatched after this attempt; it is distinct from an in-attempt continuation. The frozen `retry_policy`, maximum attempts, and planned `attempt_index` are quote-time inputs. |
| Acceptance | `A=1` only for a completed output passing the frozen independent semantic oracle. A missing/indeterminate oracle label is missing, not a failure imputed as zero. |

`provider_incomplete` is a separate censoring/instrumentation target. A
response ending at its output cap has a known incurred cost but an unknown
uncensored output amount. It may contribute to all-attempt ledger cost and
cap-risk diagnostics, but not to an uncensored output-amount selection fit.

## 2. Public interfaces and seams

This is an interface-level preregistration, not a code change. The following
existing seams define the implementation boundary:

| Seam | Responsibility in this protocol |
|---|---|
| `token_yield.wave3_features.extract_quote_time_features` and `validate_quote_features` | Create and reject-leakage-check the immutable quote row before the first response byte. Extend the registry rather than permitting ad-hoc columns. |
| `FoundryDispatcher.initial_payload`, `FoundryInputTokenCounter`, and `InputTokenCount` | Freeze the exact request, count it before generation where supported, and bind the count to request and harness hashes. The counter is never called after observing usage to create a predictor. |
| `TokenUsage`, `ResponseCall`, `DispatchResult`, and `FetchRecord` | Persist raw post-run telemetry: all usage channels, calls, latency, response bytes, and hashes. `parse_usage` already enforces `total=input+output`, `cached≤input`, and `reasoning≤output`. |
| `evaluate_oracle` / frozen case oracles | Produce structural, semantic, and overall acceptance labels independently of the cost model. |
| `HardBudget` and `economics.Pricing` | Reserve worst-case safety cost before every paid dispatch and reconcile all attempts, including failed work, without representing safety rates as provider billing. |
| `robust.grouped_kfold`, `nested_grouped_cv`, grouped bootstrap, and `quantile` | Supply the group-respecting split, nested tuning, uncertainty, and empirical quantile primitives. |
| `MultiChannelModel` | Be the forecast facade only after it accepts the complete target ledger and a frozen split plan; it must reconstruct identities rather than independently publishing incompatible channel quantiles. |

The implementation contract should expose three immutable records:

1. `QuoteTimeFeatures`: the validated feature registry row, its
   `request_sha256`, `harness_sha256`, source-snapshot hash, template version,
   recipe ID, and pre-dispatch `retry_policy`.
2. `PostRunObservation`: usage, per-call data, observed calls, latency,
   response bytes, completion/censoring state, oracle labels, and actual
   attempts. It cannot be passed to feature construction.
3. `LogicalWorkLedger`: all attempts belonging to one quoted job, their
   safety/billing rates and acceptance terminal state. It derives
   accepted-work cost; it must not accept a hand-entered `accepted_work_cost`
   predictor or silently discard failed attempts.

The fit boundary accepts `(QuoteTimeFeatures[], PostRunObservation[],
SplitPlan)` and rejects a feature name or value from the post-run ledger.
The forecast boundary returns a coherent joint draw distribution, event
probabilities, and p50/p90/p95 summaries. It must retain the model,
calibration, split-plan, and data hashes used for the quote.

## 3. Row eligibility and data firewall

### Selection-eligible row

A row is eligible for *prospective final selection* only when all conditions
hold:

1. The complete quote row, request/harness/template/source hashes, output
   cap, and retry policy were frozen before dispatch; no source outside the
   frozen training allowlist is opened.
2. The runtime stratum (endpoint, deployment/model, API version, harness
   hash, tool schema hash, effort/verbosity configuration) is recorded and is
   not pooled with another stratum.
3. Provider usage is complete and passes all target identities. The exact
   pre-dispatch provider input count and zero-payload fixed-harness count are
   available for an exact tokenizer/harness comparison; local `o200k_base`
   counts remain diagnostics only.
4. The response is complete for an uncensored amount target. Capped/incomplete
   responses are retained for cost and censoring reporting but excluded from
   that target's uncensored magnitude fit.
5. The source, template, and recipe IDs can be assigned to a split cluster;
   semantic acceptance is independently gradeable for the acceptance target.
6. The record is not from a repair, canary, test fixture, or previously opened
   blind reserve. Repeated attempts stay together in the logical-work ledger.

Eligibility is target-specific after this common firewall. For example, a
complete failed semantic output remains eligible for `A=0` and incurred
attempt-cost targets, but a response with no valid acceptance label is
ineligible for `A`; neither case is removed merely to make token performance
look better.

### Explicit exclusions

The following cannot enter a predictor matrix under any spelling or derived
alias: actual/observed cached tokens or cache hit; output/reasoning/total/input
usage; observed model or tool calls; response bytes; latency; provider status;
acceptance; error text; a retry that occurred; cost; and all response hashes or
content derived after dispatch. `Big-T` class is metadata, not a predictor:
it is determined by already-declared call structure. This follows the Wave 3
quote-time registry and prevents the leakage it was designed to detect.

Wave 3 repair rows remain useful for: validating ingestion and identities,
recording counter unavailability, checking oracle/cap instrumentation,
estimating prospective reservation safety, and demonstrating that cache
assignment cannot be relabeled as a hit. They are never final-selection,
threshold, tail-calibration, or blind-test rows.

## 4. Frozen predictor grammar

The initial main-effects grammar is deliberately small:

* exact preflight `provider_input_tokens` where available, and otherwise
  separately labeled local `task_payload_tokens` diagnostic;
* fixed `fixed_overhead_tokens` bound to the zero-payload harness hash;
* `output_bound_tokens`, `output_spec_units`, and frozen
  `declared_fetch_bytes_kib`;
* `k_declared`, `k_free_allowed`, retry-policy category, maximum attempts,
  planned `attempt_index`, composition mode and arity;
* joint `effort_level`/`verbosity_level`, randomized `cache_warm`
  assignment, and declared brick effect-coded counts;
* source *class* and serialization format only where prespecified and
  sufficiently replicated. Raw source, template, and recipe IDs are split
  keys, not high-cardinality learned identifiers.

`fixed_overhead_tokens` is an externally counted fixed baseline/intercept
component, not a coefficient the model may absorb. A source snapshot is used
to calculate fetch bytes before dispatch; measured live bytes are a target
only. The grammar retains the LEGO representation through brick counts,
composition arity, and composition mode; it does not collapse a project into
an opaque prompt-length-only regressor.

## 5. Channel models and small-data quantiles

### Hurdle structure

Each model conditions only on the frozen grammar above. The pre-authorized
hurdles and amount models are:

| Outcome | Hurdle | Positive/amount model |
|---|---|---|
| Cached input | `P(H_C=1\mid X)` | `C \mid H_C=1,X`, truncated to `[0,I]` |
| Non-reasoning output | optional `P(N>0\mid X)` only if zeros occur | `log1p(N)` |
| Reasoning | `P(R>0\mid X)` | `log1p(R) \mid R>0` |
| Branch | `P(B=1\mid X)` | If needed, `observed_model_calls-k_declared \mid B=1` is descriptive, not a predictor. |
| Retry | `P(Q=1\mid X, retry_policy permits)` | Number of retries conditional on `Q`, bounded by the predeclared policy. |
| Acceptance | `P(A=1\mid X)` among oracle-labeled completed attempts | none |
| Latency / response bytes | optional positive hurdle only if zeros occur | `log1p(L)` and `log1p(Z)` |

Use a ridge-penalized logistic model for binary hurdles, with its penalty
chosen only inside a grouped training fold. If a fold has fewer than ten
positive *or* ten negative independent clusters, do not fit a feature model:
use the training-fold beta-binomial per-regime baseline and flag the outcome
as underpowered. Amount centers use ridge regression on `log1p` scale with a
fixed small grid of penalties; retransformation uses a training-fold
smearing/residual draw rather than exponentiating a mean as though it were a
median. Constant columns, non-identifiable effects, and unapproved
interactions are dropped before fitting and reported.

The first confirmatory model contains only the main effects. The only
pre-authorized additions are the hierarchical Wave 3 interactions:
payload×brick and declared-calls×output-bound after a main-effect pass;
payload×effort and cache-assignment×declared-calls only after the second
level passes; then payload×declared-calls×effort. No search over recipes,
source IDs, target transformations, or interaction lists is allowed.

### Quantile and joint-forecast rule

Direct p90/p95 quantile regression is too unstable for the initial small,
grouped sample. Instead:

1. In every outer training fold, fit each center/hurdle and generate
   out-of-fold residual **vectors** by independent source/template/recipe
   cluster. Preserve vectors across `C,N,R,L,Z` rather than summing marginal
   p95s.
2. For a new quote, simulate a fixed 10,000 deterministic-seed draws:
   sample the binary hurdles, resample a training-only residual vector from
   the applicable predeclared regime (or its frozen coarser backoff), construct
   positive magnitudes, then enforce the identities in Section 1.
3. Take empirical p50/p90/p95 of the reconstructed draw-level `C,N,R,O,T,L,Z`
   and dollars. Thus `p95(O)` is from \(N+R\) on the same draws, not
   `p95(N)+p95(R)`.
4. Apply one-sided split-conformal residual radii to the p90 and p95
   amount/tail quote. With \(m\) training-only independent residual clusters,
   use the order statistic at
   \(\lceil(m+1)p\rceil/m\), widening rather than interpolating beyond the
   available rank. Calibrate fixed-call and branchable/retry-allowed regimes
   separately; never borrow a live/branchable radius for a fixed-call quote.

This is a grouped adaptation of conformalized quantile calibration, chosen
because it makes the train/calibrate/test firewall auditable rather than
claiming universal conditional coverage. Romano, Patterson, and Candès,
*Conformalized Quantile Regression* (NeurIPS 2019), is the primary method
source: <https://proceedings.neurips.cc/paper/2019/hash/f4a9b3f15005c5d1ef3c5e4499c9abc0-Abstract.html>.
Finite-sample conditional validity cannot generally be promised for arbitrary
subgroups; group claims are restricted to the prespecified regimes and reported
with uncertainty (Gibbs, Cherian, and Candès, 2025,
<https://doi.org/10.1093/jrsssb/qkaf008>).

## 6. Split plan, selection, and baselines

### Source/template/recipe-only grouping

The split unit is the connected component induced by shared `source_snapshot`
hash, template version, or recipe ID. If two rows share **any** one of these,
they are in the same component; replicate, attempt, and logical-work IDs are
also unioned into that component. Consequently, no source, prompt template,
composition recipe, or retry of a job can cross train and validation. Random
row splits, temporal splits that break this rule, and grouping solely by
replicate are prohibited.

Outer evaluation is leave-one-component-out when fewer than 12 components
exist, otherwise deterministic grouped K-fold (at most five folds) stratified
by prespecified runtime regime. Inner alpha/model selection repeats the same
component rule using only outer-training data. Grouped bootstrap confidence
intervals resample components, not rows. The untouched blind reserve is
evaluated once after all form, penalty, calibration, and gate decisions are
hashed; its sources remain unopened until then.

### Required comparators

All comparison predictions are fit within each training fold:

1. **Exact tokenizer/harness null:** `Î =` preflight provider input count for
   the exact request, with zero-payload harness count separately audited.
   This is the strong baseline for input; an ML model cannot be promoted by
   beating a weaker local tokenizer. `o200k_base + fixed_harness` is reported
   as a lower-bound diagnostic only, because tools and schemas add framing
   tokens that local text tokenization does not count.
2. **Pooled constant:** channel median/mean or beta-binomial prevalence from
   training components.
3. **Per-regime empirical baseline:** train-fold empirical distribution
   conditioned on the predeclared runtime/harness/effort-verbosity/call-policy
   regime, backing off in the frozen order
   `deployment+harness+effort/verbosity+call-policy → deployment+harness →
   runtime stratum`. It has no brick or size features.
4. **Declared-policy baselines:** `P(B)=0` for fixed-call plans; `P(Q)=0` when
   retries are forbidden; otherwise the corresponding training-fold
   per-regime beta-binomial rate. These expose mechanically declared
   call-count gains rather than mistaking them for predictive skill.
5. **Accepted-work empirical baseline:** simulate the training-fold
   per-regime attempt-cost and acceptance/retry distribution under the same
   frozen retry cap.

The candidate LEGO model must beat the *best eligible* baseline, not merely
the pooled constant. A baseline unavailable in a fold is reported unavailable;
it is not replaced after seeing test outcomes.

## 7. Calibration, accuracy, and tail gates

For positive/continuous channels report grouped out-of-fold MAE, median
absolute error, log-scale RMSE, pinball loss at .50/.90/.95, and observed
one-sided coverage `P(Y ≤ q_p(X))`. For derived totals and dollars, evaluate
the same quantities on the joint draw reconstruction. For binary targets
report Brier score, log loss, calibration intercept/slope, grouped reliability
table, and grouped bootstrap intervals. Latency and response bytes are
exploratory targets with the same reporting, never covariates.

The following gates are frozen:

| Gate | Requirement |
|---|---|
| Instrument integrity | Every selection row has complete hashes/telemetry, an exact input-count/harness baseline, identity checks, and a valid source/template/recipe component. |
| Main-effect usefulness | On the primary `O` and accepted-work-dollar estimands, grouped-CV loss improves at least 20% over the best eligible baseline and its 95% grouped-bootstrap improvement interval excludes zero. Input is separately compared with the exact baseline. |
| Probability usefulness | For `H_C`, `B`, `Q`, and `A`, Brier score does not worsen versus the per-regime baseline; a useful probability needs its 95% grouped-bootstrap improvement interval to exclude zero. |
| Tail publication | At least 30 independent split components overall **and** at least 10 in each reported fixed-call/branchable calibration regime. Then p90 coverage must be at least .80 and p95 coverage at least .85, each with its grouped 95% interval reported. Before this, show p90/p95 as exploratory ranges labeled with the component count, not calibrated tail claims. |
| Blind confirmation | The frozen blind reserve meets the tail and primary-loss gates without any refit, source opening, alpha change, or recalibration. A failure is reported; it cannot be repaired by reopening training selection. |

These thresholds retain Wave 3's existing 30-independent-group tail discipline.
Wave 2's 87.0% total-token coverage was based on the small pilot and is not
validated tail calibration; its output target also failed its own tail gate
(78.3% p90 coverage and −0.6% relative MAE improvement versus its constant).

## 8. Accepted-work economics

The quoted attempt price on each simulated draw is:

\[
 C_{\rm attempt}=r_u(I-C)+r_cC+r_o(N+R),
\]

where rates are attached to the quote and may differ for cached and uncached
input. This maintains the token identity: reasoning is included once through
`O=N+R`. The current campaign's conservative authorization ledger may add a
separately labeled `r_reasoning R` reserve as a *safety* surcharge if its
policy requires it; that surcharge is not provider billing and is never
described as additional provider token usage.

For a logical job, simulation draws `B`, usage/latency/bytes, `A`, and,
when permitted and needed, `Q` sequentially until acceptance or the frozen
attempt cap. It includes every failed/incomplete dispatched attempt. Define

\[
 P(W=1\mid X)=P(\text{an accepted outcome before the cap}\mid X)
\]

and, when this probability is nonzero,

\[
 {\rm expected\ accepted\!-\!work\ cost}(X)
 = E[C_{\rm all\ dispatched\ attempts}\mid X] / P(W=1\mid X).
\]

The published p50/p90/p95 accepted-work budgets are quantiles of the simulated
cost conditional on `W=1`, accompanied by `P(W=1)` and the probability of
cap exhaustion. The unconditional all-attempt spend distribution is published
alongside it. This prevents an accepted-only average from hiding failed work,
and does not use observed acceptance as a feature. The iid retry interpretation
is only a sensitivity analysis: it is allowed only where repeated shapes show
no material dependence; otherwise the fitted bounded transition model and
logical-job bootstrap are used.

The hard cumulative safety cap remains **USD 200**, with the existing USD 190
operational stop. The Wave 3 analysis records USD 14.43254 settled safety
spend, leaving USD 185.56746 to the hard cap and USD 175.56746 to the stop at
that snapshot. All proposed dispatches remain blocked pending approval; any
future run must reserve its full worst case before sending and cannot open a
blind source or paid API merely to fill a model cell.

## 9. Staged execution and ablations

| Stage | Work and data boundary | Completion/gate |
|---|---|---|
| 1 — implementation | Implement schema validation, the feature/outcome firewall, target identities, source/template/recipe split constructor, and read-only historical ingestion. Use no paid API and no blind sources. | End-to-end dry run reconstructs `O` and `T` identities, rejects every prohibited predictor, and reproduces fold assignment from frozen hashes. |
| 2 — baseline/instrumentation | In a separately approved prospective training campaign, run null/preflight counter checks, then collect only frozen training cases. Stop on telemetry, oracle, or safety failure. | Exact input count is within 2% on two null confirmations; target eligibility and all-attempt ledger reconcile. Underpowered binary outcomes remain baseline-only. |
| 3 — model selection | Fit the frozen main-effect LEGO/hurdle models with nested grouped CV, then the authorized interaction hierarchy only after each prior gate. | Primary and probability gates in Section 7 pass against strong baselines; at least 30 components are required before any tail-calibration claim. |
| 4 — ablation and blind confirmation | Run prescribed ablations and evaluate the once-opened blind reserve without retuning. | Each retained component improves or is transparently removed; blind gates pass. Otherwise retain the strongest baseline and report the negative result. |

Prespecified ablations are: (a) LEGO brick effects removed; (b) declared
fetch bytes removed; (c) declared output bound/spec units removed; (d)
`k_declared` and retry-policy inputs removed; (e) cache assignment removed;
(f) hurdle versus single amount model; (g) joint residual-vector reconstruction
versus independent marginal quantiles; and (h) exact input baseline replaced by
local tokenizer diagnostic. The latter is a leakage/measurement sensitivity,
not an avenue to promote a weaker baseline. Compare each ablation with the
same outer components, seeds, rate card, and calibration rule.

The existing Wave 3 144-session training campaign and 48-session untouched
blind reserve remain unstarted and unauthorized because the repair gate
paused. Any revised allocation must maintain the cap, the no-open-blind-source
rule, three or more repeats per source/template/recipe component where
within-component variance is claimed, and enough deliberately varied positive
and negative binary events to satisfy the gates above.

## 10. Historical Wave 2 work permitted now

The historical Wave 2 artifact can be used now, offline, to implement and
test the interfaces in Stage 1:

* ingest its 30 logical records; preserve the one usage-incomplete attempt;
  reproduce the run's target-specific 23-row grouped analysis subset only
  where its frozen eligibility rule requires it;
* recreate its source/shape grouping as a source/template/recipe split
  component plan and run the constant, per-regime, and local-tokenizer
  diagnostics in read-only mode;
* validate `N=output-reasoning` (reasoning was zero in this stratum), cache
  subset and total identities, and the distinction between declared Fetch
  branching and observed calls;
* generate exploratory p50/p90/p95 output, cache, and cost plumbing tests
  using training-fold residuals only.

It cannot support promotion: its completed pilot had missing telemetry,
failed frozen acceptance/mechanism gates, insufficient independent groups for
tail claims, an omitted pre-dispatch live-Fetch byte proxy, and mechanically
declared call targets. Its ridge result for total tokens (69.1% grouped-CV MAE
improvement over the constant) is useful historical signal, not evidence that
the multichannel model, acceptance economics, or tail calibration is promoted.
No Wave 2 model form, alpha, feature selection, coverage result, or repair
record may be carried into a final selection decision.

## 11. Implemented Stage 1 result

The Stage 1 interfaces are implemented in:

- `token_yield/multichannel_data.py`: quote/target separation, token
  identities, runtime strata, target-specific eligibility, and all-attempt
  cost derivation.
- `token_yield/multichannel_model.py`: grouped hurdle/positive-amount models,
  group-level residual calibration, censoring and identification status,
  bounded channel reconstruction, and exploratory p50/p90/p95 output.
- `token_yield/multichannel_workflow.py`: read-only joining of frozen Wave 2
  cases to measured records and an explicitly historical diagnostic.

The historical Wave 2 diagnostic uses 29 metered rows from 10 independent
groups and excludes the one row with unknown usage. It does not use Wave 3
repair or blind records.

| Channel | Grouped model MAE | Grouped baseline MAE | Result |
|---|---:|---:|---|
| Non-reasoning output | 107.996 | 113.547 | 4.9% exploratory improvement; below the 20% gate |
| Attempt safety cost | 0.07448 | 0.11318 | Arithmetic-dominated diagnostic only |
| Acceptance probability | 0.29629 | 0.43623 | 32.1% exploratory MAE improvement; not a probability-gate result |
| Reasoning occurrence/amount | unavailable | unavailable | Unidentified: no positive Wave 2 examples |
| Cache occurrence/amount | unavailable | unavailable | Unidentified: insufficient positive groups |
| Unforced branch/retry | unavailable | unavailable | Unidentified: no positive events |
| Accepted-work distribution | unavailable | unavailable | Joint logical-ledger simulation not yet implemented |

With 10 residual groups, the maximum finite one-sided empirical conformal rank
is 10/11, approximately 90.9%. The reported p90 plumbing is exploratory and a
p95 calibration claim is impossible. Because reasoning is unidentified, the
forecast interface does not reconstruct it as zero: output totals and
rate-sensitive costs remain unknown until that channel is identified.

The completed Wave 3 repair block was also retrained offline through the same
seam, without opening blind data or making paid calls. All 36 rows and 16
independent groups entered a **calibration-only** fit using payload size,
declared fetch bytes, output requirements, declared/free calls, assigned cache
regime, effort, brick effects, and live/snapshot arm.

| Wave 3 channel | Grouped model MAE | Grouped baseline MAE | Result |
|---|---:|---:|---|
| Non-reasoning output | 56.570 | 78.652 | 28.1% exploratory improvement |
| Reasoning occurrence | 0.13250 | 0.30852 | 57.1% exploratory MAE improvement |
| Acceptance probability | 0.14481 | 0.30078 | 51.9% exploratory MAE improvement |
| Attempt safety cost | 0.06231 | 0.07550 | 17.5% exploratory improvement |
| Cache occurrence | 0.19968 | 0.20585 | 3.0% exploratory improvement |
| Positive cached amount | 550.23 | 2922.67 | Only four positive groups; not tail evidence |
| Reasoning amount | unavailable | unavailable | Censored: six provider-incomplete rows |
| Unforced branch/retry | unavailable | unavailable | Unidentified: no positive events |
| Accepted-work distribution | unavailable | unavailable | Joint logical-ledger simulation not implemented |

MAE is only a plumbing diagnostic for binary channels; it does not satisfy the
preregistered Brier/log-loss, calibration, and grouped-bootstrap probability
gate. The repair block also failed its original instrumentation and semantic
gates, so these rows remain prohibited from final model selection.

These results complete only Stage 1 diagnostics. They do not authorize paid
collection, model promotion, or blind evaluation.

## Sources and audit anchors

* Repository objective and limits: `README.md`.
* Wave 2 protocol and observed result: `docs/foundry-wave2-pilot.md` and
  `runs/20260826_1627_wave2/analysis.json`.
* Quote-time registry, repair status, exact-baseline requirement, and tail
  threshold: `docs/wave3-taxonomy-and-industry-design.md`,
  `token_yield/wave3_features.py`, and
  `runs/20260827_1152_wave3/analysis.json`.
* Microsoft Foundry Responses usage response fields (primary API reference):
  <https://learn.microsoft.com/en-us/rest/api/microsoft-foundry/azureopenai/responses>.
* Token counting limitation for structured requests/tools (primary provider
  guide): <https://developers.openai.com/api/docs/guides/token-counting>.
* Group-aware conformal rationale: Romano, Patterson, and Candès (2019), and
  Gibbs, Cherian, and Candès (2025), linked in Section 5.
