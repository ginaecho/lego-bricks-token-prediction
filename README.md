# 🧱 Token Yield Marketplace Studio

**Predict the cost. Find the value. Learn what is worth building.**

Token Yield turns an AI project brief into reusable, LEGO-like functionality
bricks. It predicts token consumption and cost for the selected composition,
then connects that forecast to staffing, expected outcomes, and explicit
business assumptions.

The goal is **token economics, not token counting**: understand which work
consumes tokens, which work creates useful client impact, and how to invest in
the combinations with the strongest evidence.

<p align="center">
  <a href="docs/media/Token_Yield_Studio_hack_video.mp4">
    <img src="docs/media/marketplace-cover.jpg" width="900"
         alt="Token Yield marketplace for composing measured AI functionality bricks and forecasting cost, staffing, and value">
  </a>
</p>

## Why Token Yield?

AI projects are often scoped before teams can answer three basic questions:

1. **What will the agent do?** Compose a transparent workflow from reusable
   capabilities such as Retrieve, Extract, Classify, Write, and Verify.
2. **What will it consume?** Predict token usage and cost from measured
   executions instead of relying on a generic estimate.
3. **What will it yield?** Track whether the delivered work was retained,
   useful, and connected to a verified client outcome.

Cost is observed through token measurements. Value is learned separately from
delivery evidence and client feedback. Forecast ROI and staffing remain editable
planning assumptions until real outcomes verify them.

## See it in action

### Live Foundry Studio demo

<p align="center">
  <a href="docs/media/Token_Yield_Live_Foundry_Demo.mp4">
    <img src="docs/media/Token_Yield_Live_Foundry_Demo.gif" width="800"
         alt="Animated preview of the live Token Yield Foundry Studio walkthrough">
  </a>
</p>

<p align="center">
  <strong><a href="docs/media/Token_Yield_Live_Foundry_Demo.mp4">Watch the live Studio walkthrough (1:48)</a></strong>
</p>

The demo selects functionality bricks, changes a subtype, and immediately shows
trained composition forecasts, cost, staffing, and assumption-based ROI. It
then opens the completed training stages, holdout metrics, backend events, and
PowerShell audit evidence.

The recording inspects a completed, budget-controlled Foundry run; browsing the
saved evidence makes no new provider calls. The accepted pilot used 88 audited
training rows and eight fresh unseen-combination test rows. Its held-out mean
absolute error was 19.42 input tokens and 141.35 output tokens. This is
small-pilot evidence, not proof of production accuracy or client ROI.

### Story and model explainer

<p align="center">
  <a href="docs/media/Token_Yield_Studio_hack_video.mp4">
    <img src="docs/media/Token_Yield_Studio_hack_video.gif" width="440"
         alt="Animated preview of the Token Yield Marketplace Studio story">
  </a>
  <a href="docs/media/Token_Yield_Explainer.mp4">
    <img src="docs/media/Token_Yield_Explainer_preview.gif" width="440"
         alt="Animated preview of the LEGO-like token prediction model">
  </a>
</p>

<p align="center">
  <strong><a href="docs/media/Token_Yield_Studio_hack_video.mp4">Marketplace story</a></strong>
  &nbsp;|&nbsp;
  <strong><a href="docs/media/Token_Yield_Explainer.mp4">Model explainer</a></strong>
</p>

## How it works

```text
Client brief
   -> compose functionality bricks
   -> approve scope, budget, and assumptions
   -> predict the complete workflow's tokens and cost
   -> deliver and measure actual usage
   -> collect reviewed quality and client-outcome feedback
   -> improve candidate forecasts and recommendations
```

### 1. Train on LEGO-like compositions

Each brick is a repeatable agent capability. The current low-level vocabulary
includes `extract`, `classify`, `score`, `plan`, `retrieve`, `verify`, and
`write`; these combine into client-facing functionality such as research,
document review, support, personalization, and reporting.

Training uses real instrumented executions. Features describe operation counts,
source size, documents, components, handoffs, shared work, and execution order.
The model is selected using grouped validation and tested on held-out
combinations.

**A composition is predicted directly.** Token Yield never estimates A+B by
adding standalone forecasts for A and B: interaction, shared context, order,
and handoffs change the complete workflow's usage. Unsupported or oversized
workflows abstain instead of returning an invented total.

### 2. Turn token cost into token yield

The marketplace connects the measured forecast to:

* **Consumption:** input/output tokens, model rates, and quote-to-actual cost.
* **Delivery:** human setup, review effort, tool fees, and staffing assumptions.
* **Impact:** expected outcomes and financial assumptions, clearly marked as
  forecast rather than verified benefit.
* **Evidence:** actual usage, retained work, quality, and client-confirmed
  usefulness after delivery.

This creates a practical Token Yield question: **which composition produces the
most useful, verified outcome for its total AI and human investment?**

### 3. Learn from client feedback

The [delivered-project feedback prototype](docs/customer-outcomes-prototype.md)
carries the selected functionality into a client feedback form. Reviewed
submissions update candidate outcome predictors and an epsilon-greedy
contextual bandit that can recommend better-fitting bricks.

This is a lightweight reinforcement-learning loop over recommendations, **not
foundation-model fine-tuning**. Human approval and protected evaluation are
required before a learned policy affects future recommendations. Token
observations recalibrate consumption forecasts separately; experience ratings,
customer-reported finances, and verified business value are never treated as
the same evidence.

## Foundry vision

[![Proposed Foundry architecture: marketplace agents govern brick agents, a predictive model supports scoping, and Fabric connects insights and feedback](https://raw.githubusercontent.com/ginaecho/lego-bricks-token-prediction/main/docs/media/token-yield-foundry-marketplace-architecture.png)](https://github.com/ginaecho/lego-bricks-token-prediction/blob/main/docs/media/token-yield-foundry-marketplace-architecture.png)

The proposed architecture has brick agents deliver reusable work while
marketplace agents compose projects, govern execution, and propose improvements
from reviewed evidence. Microsoft Foundry hosts the agent workflow; the Token
Yield predictor supports scoping; Fabric could connect approved evidence to
Power BI, Excel, Word, Teams, and email.

This is a target architecture, not a deployed end-to-end cloud integration.
People approve scope, paid execution, budgets, and policy releases. The code
enforces bounded workflows, evidence separation, acceptance gates, and
abstention when support is insufficient.

## Quick start

The default demo is local and makes no paid model or Azure calls. It requires
Git and Python 3.9 or newer.

```powershell
git clone https://github.com/ginaecho/lego-bricks-token-prediction.git
Set-Location lego-bricks-token-prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m examples.marketplace_demo_server --port 8765
```

On macOS or Linux, activate the environment with
`source .venv/bin/activate` instead.

Open
<http://127.0.0.1:8765/marketplace-sales-demo.html?mode=custom>, select
**Load security project**, then **Start step-by-step + open operations**. The
shipped measured model supports read-only local shopping without retraining or
Azure credentials.

Run the measured predictor and tests:

```powershell
python -m examples.composition_demo
python -m pytest -q
```

Run the feedback-learning prototype in a second terminal:

```powershell
python -m examples.customer_outcomes_server --port 8793
```

Open <http://127.0.0.1:8793> for client feedback or
<http://127.0.0.1:8793/learning> for the internal learning view.

Live Foundry execution is optional and requires explicit configuration,
authorization, and budget approval. Follow the
[real Foundry pilot instructions](docs/marketplace-prototype.md#real-foundry-agent-pilot);
do not enable paid execution from the quick start.

## Evidence and implementation

The core experiments use real instrumented agent runs rather than invented
token labels. Earlier composition experiments reported 2.55% average
cross-validation error across 35 measured runs and 2.2% average error on four
combinations excluded from training. Results belong to their documented
datasets and scopes; they are not universal accuracy claims.

Start here:

* [Marketplace vision and Token Yield learning](docs/token-yield-learning-marketplace.md)
* [Prototype and Foundry pilot guide](docs/marketplace-prototype.md)
* [LEGO-like model training explanation](docs/lego-model-training.md)
* [Experiment and prediction results](docs/composition-findings.md)
* [Current architecture and safeguards](docs/architecture.md)
* [Testing and evaluation workflow](docs/how-it-was-tested.md)
* [Model and quoting roadmap](docs/marketplace-models/plan.md)

`token_yield/` contains the predictor, `openharness` provides measurement, and
Microsoft's [Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit)
supports the governance integration.
