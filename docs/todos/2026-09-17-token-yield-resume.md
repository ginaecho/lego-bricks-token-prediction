# Token Yield resume checklist

Paused by user request on 2026-09-17. Implementation tip at handoff:
`02204bcc385390d828ce35096ccf44323aea66a6`.
No commit or local integration was performed for this handoff. The parent plans
local integration only; no remote push will be performed.

> [!WARNING]
> Do not resume paid execution automatically. Independent review, a later
> explicit user authorization, and the original canonical funding store are
> required. Never reset spending, clear unknown telemetry, recover a ledger,
> fork funding into a copied directory, or override dissent to continue.

Use this portable checklist as the repository source of truth for resumption.
The [local Copilot handoff](../../.copilot-tracking/resume/2026-09-17-token-yield.md)
locates uncommitted session evidence, approvals, process identities and state.
Do not copy those private runtime artifacts into source control.

## Completed implementation and review boundary

| Commit | Delivered | Review status at pause |
|---|---|---|
| `6cc0adf` | Additive sales marketplace, event console, original scenarios and capability flow | Initial provenance issue found |
| `c942aae` | Mock/paid provenance through catalog, quotes, approval and export | Independent PASS |
| `677f39a` | Clarify formatting examples versus final substantive contract proposals | Independent PASS |
| `149863e` | Compact timestamped terminal with live waiting, pause/follow and saved replay | Independent PASS |
| `b182b94` | Archive Atelier v2 explicitly fictional source fixtures | Offline independent PASS, not paid permission |
| `fd6119e` | Reward-based measurement selection, real local ridge refits and audited feedback | Independent FAIL M1 |
| `2a0d6b3` | Disabled additive ordinary-v2 scope under existing funding | Independent FAIL S1 |
| `2af2ea4` | Include requested/novel actions; stable identities and explicit exclusions | Independent correction PASS received during handoff |
| `02204bc` | Bind scope to one canonical funding store and shared lock | Independent correction PASS received during handoff |

M1 excluded the requested new retention ledger by taking only the first two
selected contracts. S1 accepted identical copied accounting in another state
directory. Both final re-reviews were pending at the start of this handoff.
The Inspector then published PASS reports for the exact correction commits:
`measurement-policy-correction-inspector-feedback.md` and
`archive-v2-funding-correction-inspector-feedback.md`. Those reports were read
before finalizing these docs. Independent correction review is complete, with
no remaining M1/S1 blockers. Paid permission and the live goal remain outstanding.
Read the reports again when resuming and check for any later feedback rather
than relying on an old status snapshot.

After the pause is lifted, the next execution decision is explicit permission
for a bounded ordinary Archive Atelier v2 attempt against the same canonical
existing USD50/USD48 funding store. Paid bandit execution remains separately
excluded and unapproved. Neither the PASS verdicts nor local Git integration
grant permission to spend.

## Architecture and exact learning meaning

See the [implementation guide](../marketplace-prototype.md),
[agent orchestration](../../token_yield/marketplace_agents.py),
[measurement adapter](../../token_yield/marketplace_measurement.py),
[policy](../../token_yield/measurement_policy.py) and
[funding scope](../../token_yield/marketplace_scope.py).

| Requested flow | What currently executes | Boundary |
|---|---|---|
| Orchestrator | Source-bound request, novelty/reuse resolution and supported contract selection | Genuine unsupported scope or disagreement blocks continuation |
| Agents and human | Three-role contract/project discussion and reconciliation; optional stage stepping | No dedicated human approval gate between final contract review and establishment |
| Feature engineering | Validated feature-builder choices and numeric contract/source features | Not unrestricted generated feature code |
| Training | Measured training rows fit supervised ridge input/output-token estimators | Mock token labels are scripted fixtures; paid labels require real provider telemetry |
| Prediction | Compatible model forecasts feed catalog and project quotes | Provenance must remain visible; old/mock forecasts cannot become measured paid output |
| Feedback | Calibration rewards update measurement-selection action values, followed by more measurements/refits | No production client-outcome feedback ingestion and no LLM reinforcement fine-tuning |

Mock agents run the same orchestration, validation, measurement plumbing and
actual numerical ridge fitting. Their agent responses and token counts come from
fixtures. Mock costs use the simulated rate card, not billed provider usage.
The mock proves plumbing and reproducibility, not live model accuracy.

### Measurement-selection bandit

The policy is a non-contextual epsilon-greedy bandit, not a learned contextual
policy. Decision context is audited. Epsilon is 0.2; there are four adaptive
decisions per run. Cold start samples unseen actions uniformly. Later choices
record exploration, probabilities/propensity, the random draw, a uniform-random
baseline and the predictor version.

The bounded actions are two supported single contracts and their fixed ordered
pair. A resolved requested capability is mandatory, including reuse. Without
one, selected novel contracts are mandatory; more than two requires scope
narrowing rather than silent omission. Remaining slots prefer novelty and then
lexical contract IDs. Selection/exclusion reasons are visible in saved artifacts
and the terminal. Equivalent reorder/reuse preserves policy identity; changed
actions/contracts, feature schema or provenance do not share incompatible scores.

```text
reward = clip(((MAE_before - MAE_after) / max(MAE_before, 1))
              * 0.001 / measured_marginal_cost_usd, -1, 1)
Q(action) <- Q(action) + (reward - Q(action)) / observation_count(action)
```

Reward uses calibration input/output token error, not the final acceptance set.
Regression produces negative reward. Missing/invalid telemetry, nonfinite
metrics, or nonpositive cost cannot manufacture a reward. Feedback is durable,
hash-audited and idempotent; duplicate/conflicting observations and unsettled
decisions have explicit guards. Seed and calibration overhead remain separately
accounted for, not hidden inside marginal reward.

Training uses templates 0, 1 and 2, including a repeat of 0. Calibration uses
template 3. Final acceptance templates 4 and 5 are measured only after final
parameters freeze and never update the policy. Provisional reward fits use
ridge alpha 1; final model regularization follows the existing training-only
selection. Improving one does not guarantee improvement in the other.

### Pairs and empirical limitations

Each supported pair runs two validated calls in a fixed order against the same
original source documents. Usage and costs come from those actual executions.
There is no first-output-to-second-input handoff, dependent workflow, shared
context saving or arbitrary combination support. Marketplace project forecasts
still sum independent brick predictions; they are not measured sales combinations.

The retained deterministic synthetic comparison used 24 measurements and virtual
USD0.024 per policy. Cumulative calibration error was
15.897435897435896 for the bandit versus 13.477639751552799 for uniform sampling:
the bandit was worse. One final synthetic point favored the bandit
(0.4153846153846139 versus 0.6153846153846168), which does not establish a general
accuracy or efficiency advantage. Do not tune frozen acceptance data to claim one.

Staffing, hourly rates, utilization and ROI are business assumptions, not learned
client outcomes. Sales quote approval is not training or establishment approval.
Repeated measured-proxy refitting is supervised learning, not field feedback.

### Predictor compatibility

Old published model `933d1eb284b44833b76666ed99b00ab0` is retained historical
evidence, not a usable current paid predictor. It used `source-proxy-v1`;
current contracts use `source-atoms-v2`, with none of the 16 current base contract
hashes matching. Preserve it, but obtain a freshly measured compatible predictor
before presenting current paid forecasts. Never relabel it or a mock model.

## Paid outcomes and preserved accounting

The original USD25 campaign is separate and unchanged. The additional campaign
has a USD50 total cap and USD48 operational stop, not USD50 of fresh money on
each resume. These are the last verified totals, not a new spending permission.

| Accounting item | Value |
|---|---|
| Paid calls, including stopped attempts | 22 |
| Rated usage cost | USD0.120651 |
| Safety-accounted spending | USD1.49220 |
| Reserved amount | USD0 |
| Remaining to operational stop | USD46.50780 |
| Remaining to hard cap | USD48.50780 |

| Attempt | Saved run ID | Actual outcome |
|---|---|---|
| Original Archive Atelier | `727561a9b1f940b1979731696e728bbb` | Cancelled at contract disagreement; 8 calls, rated USD0.03634, safety USD0.40928 |
| Archive retry after format clarification | `899184db1c7d46ea92077315ed337c49` | Cancelled after substantive source-domain objections; 14 calls, rated USD0.084311, safety USD1.08292 |

Neither attempt completed workload measurement, predictor training or publication.
There are zero successful paid demos. Booking Blocks and Care Crew did not start.
Final atom counts on the retry agreed, but missing retention obligations,
exception taxonomy and supporting evidence remained substantive objections.

The [three original scenarios](../../token_yield/marketplace_scenarios.py) remain
immutable. [Archive Atelier v2](../../token_yield/marketplace_source_fixtures.py)
is an additive, versioned fictional policy demo, not an overwrite of Archive v1,
not Paperless policy, not real legal/security authority and not legal advice.
A v2 success alone would not complete or replace the original three-case goal.

## Priority resume checklist

### Read final reviews and preserve state

* [ ] Locate final Inspector feedback for exactly `2af2ea4` and `02204bc`.
  Confirm the newly received PASS reports and review their limitations. If those
  reports are unavailable or the code changes, require appropriate re-review;
  do not infer PASS from scratch files or earlier terminal/v2 reviews.
* [ ] Read the immutable goal, later scope authorizations and both original M1/S1
  reports through the local handoff. Preserve all historical verdicts.
* [x] Receive final independent PASS for both M1 and S1 corrections.
* [ ] Check current branch/worktree and any parent-created local integration commit.
  Preserve unrelated changes and Inspector-owned scratch.
* [ ] Compare the original approvals and both campaign ledgers with retained
  fingerprints. Reconfirm spent/reserved values read-only. Stop on ambiguity.
* [ ] If later feedback or a code change introduces a new blocker, make only the required offline fix,
  add the exact regression, commit separately and obtain independent re-review.
  Do not change prompts, scenario wording or expected outcomes to force success.

### Obtain distinct permissions

* [ ] Confirm the independent PASS for ordinary-v2 source/scope/single-store
  behavior still applies to the exact code and source proposed for execution.
* [ ] Then ask for explicit ordinary-v2 paid-run permission under the same
  existing campaign, exact versioned brief and canonical original funding store.
  The new store-bound request must remain disabled until that permission.
* [ ] Treat paid bandit execution separately. It is currently hard-blocked outside
  mock mode. Ordinary-v2 consent does not authorize the bandit. Any paid-bandit
  enablement needs a separately scoped offline engineering/review step and later
  explicit run permission covering actions, calibration splits, reward,
  overhead/retries and a bounded draw from the same remaining funds.
* [ ] Never fork, copy or move the funding directory to create a fresh allowance.
  Canonical aliases share one lock; clones/replacements are rejected. No automatic
  migration, telemetry recovery, reset or new allocation is authorized.

### Execute only after those gates

* [ ] Verify the actual current deployment, configuration, authentication,
  canonical store, reviewed scope and cap/stop separation before any paid call.
* [ ] Identify an owned server by its exact process/shell identity. Only after
  permission, replace stale code in that owned process and verify responsiveness.
  Never stop unrelated servers or assume a URL serves the latest Python code.
* [ ] Use new run IDs. Preserve the cancelled attempts and their settlements.
* [ ] Run only the specifically authorized version/case, then wait for its actual
  measurement, features, retraining, fresh holdouts and publication outcome before
  advancing to another case. No broad campaign or automatic paid retry.
* [ ] Stop on substantive dissent, provider/accounting ambiguity, unknown telemetry
  or budget halt. Report a code defect before making changes or more paid calls.
* [ ] Save real event streams, run IDs, new/reused decisions, actual call counts,
  token metrics, rated/safety costs, failed-attempt costs, training reuse,
  calibration/final splits, model versions and holdout metrics.
* [ ] Verify sales and terminal show actual live versus replay state and truthful
  provenance. Preserve the cute-brick marketplace, top ROI layout, details links,
  user pause/follow, errors/dissent and no fake shell commands.
* [ ] Independently inspect paid evidence. Resolve the source blockers and obtain
  explicit scope decisions before any continuation of the original three cases.
  Keep all three incomplete until their unchanged requirements actually succeed
  or the user explicitly versions the goal.

### Separate product gaps

* [ ] If requested later, scope a genuine human approve-before-establishment gate.
  Do not advertise current stage stepping or quote approval as that feature.
* [ ] If requested later, scope real client-outcome feedback ingestion and its
  consent, provenance, deduplication and reward semantics. It does not exist now.
* [ ] Require a new scope decision before dependent output-handoff workflows or
  measured sales-combination forecasts. Current pairs do not implement them.

## Future offline commands

These commands are for resumption, not execution during the pause. Run from the
repository root with the configured Python environment and existing dependencies.
The last local evidence used Python 3.13.15. No tests, servers or dependency
installs were run to create this handoff.

Focused policy and scope regressions:

```powershell
python -m pytest tests\test_measurement_policy.py tests\test_marketplace_measurement.py tests\test_marketplace_scope.py -q --tb=short --basetemp=.feedback-resume-tests
```

Related source, workflow and provenance regressions when a fix warrants them:

```powershell
python -m pytest tests\test_marketplace_source_fixtures.py tests\test_marketplace_feedback.py tests\test_marketplace_agents.py tests\test_marketplace_demo.py -q --tb=short --basetemp=.feedback-resume-regressions
```

Use a confirmed unused port and a separate new mock run directory. Example from
the [actual server CLI](../../examples/marketplace_demo_server.py):

```powershell
python -m examples.marketplace_demo_server --port 8790 --run-dir .feedback-resume-archive-v2-policy --mock-agents --source-fixture archive-exceptions-v2 --measurement-policy
```

The mock CLI derives its state under that run directory as `mock-state`; it does
not use a supplied paid state directory. Omit `--measurement-policy` and choose a
different run directory for ordinary v2. Omit both version/policy flags and use
another directory for the original three-story mock. Do not combine mock mode
with `--enable-foundry`, `--campaign-file` or `--scope-file`.

After authorized offline startup, verify the mock runtime before browser tests:

```powershell
Invoke-RestMethod http://127.0.0.1:8790/api/runtime
```

Require `source=mocked-test-provider` and the intended source fixture/policy.
The example sales URL is <http://127.0.0.1:8790/marketplace-sales-demo.html>.
The terminal is <http://127.0.0.1:8790/marketplace-console.html?run=RUN_ID>;
substitute an actual saved run ID. Detailed operations remain at
<http://127.0.0.1:8790/marketplace-operations-demo.html>.
These URLs are examples, not a claim that port 8790 is running.

With already installed Playwright and Edge, use a separate evidence directory:

```powershell
$env:PLAYWRIGHT_MODULE = (Resolve-Path .feedback-tools\node_modules\playwright).Path
$env:PLAYWRIGHT_CHANNEL = 'msedge'
$env:MARKETPLACE_EVIDENCE = Join-Path $env:TEMP 'token-yield-resume-terminal'
node tests\marketplace_terminal_browser.cjs
node tests\marketplace_provenance_browser.cjs
$env:MARKETPLACE_URL = 'http://127.0.0.1:8790'
$env:MARKETPLACE_EVIDENCE = Join-Path $env:TEMP 'token-yield-resume-source'
node tests\marketplace_source_fixture_browser.cjs
```

Terminal/provenance tests intercept fixture requests. The source-fixture browser
test starts an actual mock run and refuses non-mock provenance first; never point
it at the paid server. Preserve captures outside tracked source.

Before the pause, Builder evidence recorded 27 policy tests, 183 related
regressions, 44 final funding/locking checks and desktop/mobile terminal checks
passing. These runs overlap and are not a unique aggregate test count. Both
original failing regressions passed afterward. The subsequently received
independent correction reports separately record 71 focused tests (27 policy
and 44 funding/locking), a repeated synthetic baseline, four store-mutation
probes, desktop/mobile terminal checks and 16 provenance cases passing.
The Inspector independently verified 16 fresh feedback rewards, propensities
and two durable audit chains. No tests were launched during this documentation
handoff. Existing CI runs pytest and Python
smoke/benchmark commands; no repository Markdown-specific validation task was found.

## Commit and local integration handoff

The parent reports local `main` and `origin/main` at
`993696194cb6d6eecd192aea0ad308e6bcc47e0a`, including video commits that must be
preserved. The demo implementation remains at `02204bcc`. Treat this as the
integration baseline, not a request to replace main with the demo branch.

* [ ] Parent reviews and stages only the two requested handoff documents.
* [ ] Parent creates the documentation commit under repository conventions and
  performs the planned local integration, preserving main's video commits.
* [ ] Recheck both branch tips before integration. Do not reset main to the demo
  tip or discard unrelated main changes. No remote push is planned or authorized.
* [ ] Do not stage live state, approval files, copied evidence, test output or
  unrelated Inspector scratch. Do not assume integrating code grants paid consent.
* [ ] Update pause/review status from actual evidence at resumption without
  overwriting earlier Inspector results or the immutable goal.
