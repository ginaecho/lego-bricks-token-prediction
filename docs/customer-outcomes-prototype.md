# Delivered-project feedback prototype

Start with a delivered marketplace build, not a new project. Its selected
functionalities carry into a short feedback form. Submission saves the response,
extracts signals, and updates candidate learning models in the background.
The client never needs to create scope or operate training controls.

This is a local prototype with persistent SQLite storage and real learning
algorithms. Demonstrations are synthetic, not customer results. It does not
dispatch work, run paid inference, or fine-tune foundation-model weights.

## Run locally

From the repository root, using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m examples.customer_outcomes_server --port 8793
```

Open <http://127.0.0.1:8793>. The scratch database persists at
`.outcomes-prototype/PROTOTYPE-customer-outcomes.sqlite3`, outside version
control. Use `--db` to select a different local database. No additional packages,
credentials, paid model calls, or cloud services are required.

## Customer workflow

1. Open the existing marketplace at
   <http://127.0.0.1:8784/marketplace-sales-demo.html>.
2. From the selected build, choose **Delivered project: give feedback**.
   The handoff includes only those selections, their forecast basis, and the
   planned team. A custom whole-project selection expands its original
   functionality breakdown; aggregate tokens are not invented per function.
3. Rate satisfaction, how much output was kept, and whether it helped, with
   optional comments for each selected functionality. SOW quality and staffing
   fit are optional project ratings. Observed finances and autonomous-delivery
   percentages are optional, collapsed details.
4. Confirm delivery and permission to learn, select demo or actual-client
   evidence, and send. Saved responses survive reload and server restart.

The marketplace has quotes and measurement runs, not authoritative delivery
receipts. Delivery is explicitly attested, never inferred from a completed
measurement run. The same frozen selection reuses its local receipt ID, so a
retry or feedback update does not create another reward. Changed scope requires
a different receipt. Local receipt IDs are not authenticated order IDs.

The storefront lives in the companion `marketplace-sales` worktree, branch
`main`; the feedback service lives in `gc/model-improvement`. The integration
edit is in that worktree's `marketplace-sales-demo.html`. Run the existing
marketplace server and this feedback server together. Keep both changes when
consolidating the branches. The storefront opens a loopback feedback tab using
a URL-fragment handoff; it does not relax either server's same-origin policy.
The fragment contains scope, so treat copied handoff links as customer data.

## What submission trains

[`delivery_feedback.py`](../token_yield/delivery_feedback.py) binds feedback to
the purchased variant/model, not a retrospectively sampled action.
It keeps actual-client and synthetic feedback in separate namespaces.

The versioned `delivered-experience-v1` reward uses reported experience:

```text
rating = (satisfaction - 1) / 4
kept = all: 1, some: 0.5, none: 0
useful = yes: 1, partly: 0.5, no: 0
reward = 2 * (0.4 * rating + 0.3 * kept + 0.3 * useful) - 1
```

Unknown retention or usefulness keeps the reward pending. This categorical
reward is not AES, verified ROI, or measured token yield. Free-text comments
produce transparent topic tags, not fabricated ratings or financial evidence.

Candidate action values are rebuilt from the latest unique training receipts.
Ridge models fit satisfaction, SOW quality, staffing fit, reported ROI, provider
profit, and reported AES separately. They require at least three independent
training receipts per feature/action/target. Financial labels remain
`customer_reported_not_verified`; forecast ROI never becomes an observed label.
Prediction through `/api/delivery-policy/predict` uses only a human-approved
model and abstains outside observed input ranges or without sufficient support.
These are associational estimates, not causal ROI or calibrated confidence.

Optional percentages yield reported AES as `100 * A * K * U`, using fractions
from the same autonomous-work cohort. Actual tokens, including retries and
failures, remain unknown unless reported. Token forecasts are never reused as
actual usage. SOW and staffing ratings train outcome targets but are not part
of the categorical experience reward.

## Closing the recommendation loop

The existing marketplace brick configuration has an optional **Suggest from
past delivery feedback** control. It opens a suggestion tab, logs the selected
variant/model with its complete probability distribution, and lets the person
review it before saving. Human overrides do not inherit false randomized-choice
credit. Frozen measured forecasts are not changed through this control.

The epsilon-greedy contextual bandit learns feature-specific variant/model
action values. Before an approved policy exists it explores uniformly.
After approval it exploits the best values with 20% exploration.
Historical human-selected deliveries can train candidate action values, but
only genuinely logged suggestions can support policy evaluation.

Twenty percent of receipt IDs are deterministically reserved for validation.
Their labels never train the models. An operator calls
`POST /api/delivery-policy/promote` with `source` and a named `approver`.
Promotion requires at least 20 training projects, 20 fresh prospectively logged
validation projects, effective sample size at least 10, and a positive
approximate 95% lower bound for reward improvement over the active policy.
Evaluation cohorts cannot be reused. Promoted training evidence and consumed
validation evidence are locked; exact training snapshots and audit events
persist with model versions.

Submitting customer feedback never promotes a policy. The prototype has no
authenticated operator roles; the local promotion endpoint is an internal
workflow, not a production authorization boundary. Accepted suggestions,
selective responses, and feature-only policy context can bias evaluation.
Do not claim causal improvements without a controlled prospective pilot.

## Visible learning in the Studio

Open <http://127.0.0.1:8793/learning> for the internal learning view.
The customer feedback form remains separate and short.

The learning view shows two uses of feedback:

* Marketplace internal rankings show each brick's historical experience score,
  independent project count, and variant/model evidence. A score is
  `50 * (1 + mean reward)`, not AES, ROI, token efficiency, or a confidence
  percentage. Each project contributes one average per brick or option.
  At least three independent training projects are needed for a rank; an
  unfamiliar brick stays unranked rather than receiving a zero.
* Custom-project recommendations retain every matched requirement and sample
  only compatible variant/model options using the active policy. Learned
  experience prioritizes the matching bricks. It never adds unrelated
  high-scoring functions or silently drops required functionality.

In the Studio, choose **Learning & brick rankings**, inspect the evidence,
then send the rankings back to the original tab. Its optional internal-rank
display changes catalog ordering and badges, not the selected build or prices.
Candidate rankings are explicitly previews, distinct from active policy.

For a customized project, enter its description in the existing Studio and
choose **Recommend matching bricks from feedback**. Review the matching needs
and generate recommendations in the learning view, then send them back.
The Studio requires a second explicit scope approval before adding the bricks,
preserves other selected scope, and rejects stale brief/build responses.
Original logged decision IDs continue through delivered-project feedback.
Frozen measured workflows are not rewritten by this catalog prototype.

Requirement matching reuses the Studio's keyword preview, not a new LLM.
Intent, negation, exclusions, and suitability still require human review.
Explicit variant requirements such as web research remain constraints;
the policy does not replace them with normal research based on popularity.
Across-brick rankings are descriptive and affected by task difficulty and
customer mix. They are not evidence that unrelated functions are interchangeable.
Policy evaluation does not validate every possible new constrained option set.

Use **Load synthetic example** to generate fictitious deliveries through the
actual reward-training path. This supplies separate training receipts and 140
prospectively logged holdout deliveries. It never activates a policy.
A named operator can then evaluate and approve the demonstration candidate;
the real-client namespace remains untouched. Repeating the seed for the same
catalog does not duplicate outcomes.
Generated training examples stay out of the customer's delivered-project list;
they remain available to the synthetic learning and audit views.

The standalone learning page uses
[`marketplace_feedback_catalog.json`](../examples/marketplace_feedback_catalog.json),
a metadata-only snapshot of the existing seven-brick storefront.
Opening from Studio passes its current eligible catalog instead. The snapshot
contains no token rates, no measured forecasts, and no outcome evidence.
Regenerate it from storefront metadata if the baseline manual catalog changes.

## Legacy internal studio

The earlier technical studio remains at <http://127.0.0.1:8793/admin> for
internal demonstrations, not as the customer entry point. Its SOW/staffing
package policy and reviewed-evidence reward are separate from the purchased
variant/model feedback loop above. Its synthetic seed does not populate or
train the delivered-project feedback namespace.

The remaining sections describe that legacy internal workflow:

1. Start with the empty real-customer workspace, or explicitly seed the
   synthetic demonstration. Its 180 training and 360 validation projects remain
   separate from real data.
2. Create a project with a description and explicit functionalities. Copy the
   functionality and estimated tokens from your marketplace proposal. The
   prototype does not infer a trustworthy decomposition from arbitrary prose.
3. Freeze each functionality's work units, complexity, observation window,
   token estimate, customer/provider budgets, and reference benefit.
4. Request a recommendation. The active policy samples an approved package:
   lean, balanced, or assured. These packages describe different SOW detail,
   staffing specialization, and review intensity. A person approves or
   rejects the recommendation before it can receive delivery feedback.
5. Record what happened outside the prototype. The client records satisfaction,
   usefulness, impact, SOW quality, staffing suitability, comments, and financial
   evidence. The project manager attests to autonomous completion, retention,
   actual token consumption, and gates. Approval alone is not delivery.
6. Review the evidence explicitly. Only approved reviews enter training.
7. Train the outcome predictors and a candidate recommendation policy.
   This does not activate that policy.
8. Evaluate on protected validation projects. Each validation project can be
   consumed by only one candidate evaluation; its labels never train either
   model. A failed evaluation needs new training feedback and fresh validation
   projects, not repeated tuning against the same holdout.
9. If the evaluation gate passes, a named person promotes the candidate.
   The next recommendation uses the new policy and still requires approval.

The synthetic seed supplies fictitious approvals and reviews for demonstration.
Training, evaluation, and promotion remain explicit button presses.

## Two distinct forms of learning

### Outcome prediction

Ridge regressions reuse the repository's numerical implementation. Each target
is fitted separately: satisfaction, SOW quality, staffing fit, self-reported
impact (1 to 5), AES, customer ROI, provider profit, and customer net benefit.

Inputs are available before delivery: functionality kind, complexity, package,
interactions, assigned scope, estimated tokens, budgets, observation days, and
the reference benefit. Ratings, comments, actual tokens, and realized financial
outcomes are labels or evidence, never input features.

Models persist their fitted parameters, exact training snapshots, evidence IDs,
and versions. Financial targets require a closed observation window and
reviewed, verified financial evidence. Ratings can train before finances are
available. Such provisional feedback can be revised and re-reviewed; refitting
uses the latest record once, while prior snapshots stay unchanged.

Predictions abstain unless at least three independent training projects match
the functionality kind, complexity, package, and target. They also abstain
outside observed numeric input ranges. Support counts and held-out mean
absolute error (MAE) are shown instead of invented confidence percentages.
These are associational pilot estimates, not calibrated intervals, guaranteed
profit, or causal attribution.

### Reward-trained recommendations

The contextual bandit keeps a learned average reward for each
`functionality kind x complexity x package` cell. Candidate training rebuilds
those sufficient statistics from unique reviewed, reward-complete decisions:

```text
n[action] = n[action] + 1
Q[action] = Q[action] + (observed_reward - Q[action]) / n[action]
```

After promotion, an epsilon-greedy policy uses these values to sample future
packages. Exploration is fixed at 20%; ties share the remaining probability.
Each decision records the complete probability distribution, selected action,
context, active version, frozen project split, and reward specification.

This is a real one-step reinforcement-learning problem, not an LLM
fine-tuning job or a multistep agent controller. No foundation-model weights
change. Customer feedback does not train the token-cost predictor; that
existing predictor still learns separately from actual token measurements.

## AES, financial outcomes, and reward

Work units must stay comparable through the same autonomous-delivery cohort.
Mandatory approval is not rescue; substantive human correction excludes the
affected work from independently completed units.

```text
A = autonomously completed units / assigned units
K = kept autonomous units / autonomously completed units
U = verified-useful kept autonomous units / kept autonomous units
AES = 100 * A * K * U
Value-aligned token yield = useful autonomous units * 1000 / all actual tokens

Customer net benefit = attributable customer benefit - total customer cost
Customer ROI = customer net benefit / total customer cost
Provider profit = provider revenue - total provider cost

reward = 0.4 * AES / 100
       + 0.2 * satisfaction / 5
       + 0.1 * SOW quality / 5
       + 0.1 * staffing fit / 5
       + 0.2 * clip(customer net benefit / frozen reference benefit, -1, 1)
```

All money is USD and uses the declared observation window. Customer cost and
provider cost are different accounting perspectives, not additive costs.
Include tokens, tools, implementation, review, rework, and operations in the
appropriate total; do not count them twice. Time saved needs an agreed
monetization and attribution method. Self-reported impact is not cash profit.

Missing evidence stays unknown. Zero customer cost makes ROI unavailable, not
infinite. Negative ROI and profit are retained. Known zero completion or
retention makes AES zero, while undefined conditional rates stay unavailable.
The actual token denominator includes failed work and retries.

Rewards stay pending without complete ratings, a closed outcome window,
positive token accounting, financial evidence, and delivery evidence. Failed
safety or budget gates block reward learning regardless of satisfaction.
Exceeding a frozen customer or provider budget fails the gate even if the user
checks the budget checkbox. The fixed reward prioritizes customer value;
provider profit and self-reported impact remain separate prediction targets.

Comments are stored verbatim and tagged with transparent keyword themes such
as scope, staffing, quality, cost, and timing. These tags are not sentiment
inference and never manufacture financial or satisfaction labels.

## Evaluation and limitations

Candidate evaluation compares reward against the active policy using logged
action probabilities (inverse propensity scoring). It averages within each
project before calculating the gain and an approximate 95% normal lower bound.
Promotion requires at least 20 reward-bearing validation projects, effective
sample size of at least 10, at least 20 training rewards, and a strictly
positive lower bound. Regression validation also compares MAE to the training
mean baseline.

Evaluation is conditional on human-approved choices. Rejected recommendations
remain in the audit but have no observed counterfactual reward. Human selection,
missing responses, dishonest feedback, and changing business conditions can
bias results. The prototype does not establish causal impact or replace a
prospective controlled pilot. Package-specific constraints beyond the three
predefined safe options must be enforced by the delivery team.

Reviewed reward-bearing and evaluated feedback is frozen to preserve exact
credit assignment. Correcting settled evidence requires a governed model
invalidation workflow, not editing records in this pilot. Each functionality
has one accepted decision per project; new delivered work needs a new cohort.

The SQLite audit is append-only through the application, not tamper-proof
against someone with filesystem access. Approver, reviewer, evidence reference,
and elapsed-window fields are local attestations, not authenticated identities,
automatic runtime telemetry, elapsed-time enforcement, or verified documents.
The loopback server enforces same-origin JSON requests but has no multi-user
authentication. Do not expose it to a network or load sensitive client data
without an approved access, retention, consent, and deletion design.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_delivery_feedback.py tests\test_customer_outcomes.py -q
```

Tests cover the exact AES funnel, pending evidence, negative financial results,
invalid inputs, hard gates, approval/review sequencing, source and project-split
isolation, persisted models, prediction support, idempotent training, holdout
evaluation, changed next-decision probabilities, and server request validation.
Synthetic improvements validate the learning mechanism, not real customer ROI.

The delivered-project Chromium walkthrough covers manual and approved custom
selections, whole-workflow expansion, a single-submit feedback round trip,
unchanged-receipt retries, safe text rendering, reload persistence,
prospective suggestion attribution, source isolation, and a 390-pixel mobile
layout. It runs against a separate scratch database and makes no paid calls.

The Chromium browser walkthrough also exercises seeding, training, evaluation,
explicit promotion, project creation, new recommendation approval, reported
versus verified finances, human review, persistence after reload, and a
390-pixel mobile layout. An illustrative 80 completed / 60 kept / 30 useful
cohort produces AES 30, not a satisfaction-derived score.
