---
title: Token Yield Marketplace Studio
description: Predict the cost and value of AI agent work by composing measured, LEGO-like task bricks.
ms.date: 2026-09-17
ms.topic: overview
---

# Token Yield Marketplace Studio

[![CI](https://github.com/ginaecho/lego-bricks-token-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/ginaecho/lego-bricks-token-prediction/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/1314056228.svg)](https://zenodo.org/badge/latestdoi/1314056228)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <a href="docs/media/Token_Yield_Studio_hack_video.mp4">
    <img src="docs/media/marketplace-cover.jpg" width="900"
         alt="Token Yield functionality marketplace: shop measured AI task bricks, forecast cost, staffing, and ROI, then build the project">
  </a>
</p>

<p align="center">
  <a href="docs/media/Token_Yield_Studio_hack_video.mp4">
    <img src="docs/media/Token_Yield_Studio_hack_video.gif" width="440"
         alt="Animated preview of the Token Yield Marketplace Studio hackathon story">
  </a>
  <a href="docs/media/Token_Yield_Explainer.mp4">
    <img src="docs/media/Token_Yield_Explainer_preview.gif" width="440"
         alt="Animated preview explaining the LEGO-like token prediction model">
  </a>
</p>

<p align="center">
  <strong><a href="docs/media/Token_Yield_Studio_hack_video.mp4">Hackathon demo</a></strong>
  &nbsp;|&nbsp;
  <strong><a href="docs/media/Token_Yield_Explainer.mp4">Model explainer</a></strong>
  <br>
  Select either animation to open its full MP4 with playback controls.
</p>

**Predict the cost. Prove the value. Win client approval faster.**

AI projects are often sold before anyone can clearly explain what they will
cost or what value they will deliver. Token Yield turns a plain-English request
into a transparent, evidence-based proposal. Agents decompose the work into
reusable task bricks, combine the right capabilities, and predict token use,
cost, and risk. People approve the scope and commercial assumptions. Actual
delivery results feed the learning loop for the next project.

## Live Foundry demo

<video controls preload="metadata" width="100%"
       poster="docs/media/marketplace-cover.jpg">
  <source src="docs/media/Token_Yield_Live_Foundry_Demo.mp4"
          type="video/mp4">
</video>

<p align="center">
  <a href="docs/media/Token_Yield_Live_Foundry_Demo.mp4">
    <img src="docs/media/Token_Yield_Live_Foundry_Demo.gif" width="800"
         alt="Animated preview that starts with the Token Yield marketplace UI, then shows live terminal model training and prediction evidence">
  </a>
</p>

<p align="center">
  <strong><a href="docs/media/Token_Yield_Live_Foundry_Demo.mp4">Open the full live demo with playback controls</a></strong>
</p>

The recording starts with the front-end marketplace and project submission,
then moves behind the scenes to operations and the real terminal event stream.
It captures model measurement, training, evaluation, prediction, and cost
reporting. The run used GPT-5.4 and an explicitly approved USD 25 campaign cap;
it is a real agent workflow rather than a front-end animation.

| Live run evidence | Result |
| --- | ---: |
| Provider responses | 107 |
| Measured usage | 56,726 input + 24,280 output = 81,006 tokens |
| Calculated API cost from the recorded rate card | USD 0.492767 |
| Conservative safety-budget settlement | USD 5.99052 |
| Model data split | 64 training + 32 held-out rows |
| Held-out model error | 95.62 input tokens + 53.19 output tokens MAE |
| Final status | Completed; broad project quote withheld for human review |

The retail-rate calculation is not an Azure invoice. The larger safety amount
is the conservative budget guard reserved before calls. The model learned from
the measured bricks, but the system did not publish a complete project quote
because the requested scope exceeded the supplied evidence and agent
disagreement remained. That abstention is part of the governance design.

## Quick start

The default demo runs entirely on your computer. It uses the offline simulation
and makes no Azure or paid model calls.

### 1. Install the prerequisites

You need:

* [Git](https://git-scm.com/downloads)
* [Python 3.9 or newer](https://www.python.org/downloads/)
* A modern web browser

Confirm Python is available:

```powershell
python --version
```

On Windows, use `py` instead of `python` if that is how Python is installed.

### 2. Clone and install

On Windows PowerShell:

```powershell
git clone https://github.com/ginaecho/lego-bricks-token-prediction.git
Set-Location lego-bricks-token-prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS or Linux:

```bash
git clone https://github.com/ginaecho/lego-bricks-token-prediction.git
cd lego-bricks-token-prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 3. Start the marketplace

```powershell
python -m examples.marketplace_demo_server --port 8765
```

Keep that terminal open, then visit:

<http://127.0.0.1:8765/marketplace-sales-demo.html?mode=custom>

In the studio:

1. Select **Load security project**.
2. Select **Start step-by-step + open operations**.
3. Use **Next step** to approve and run each stage.
4. Review the brick composition, token forecast, cost, staffing, and ROI view.

If port `8765` is busy, start the server with `--port 8766` and use the same
port in the browser URL. Press `Ctrl+C` in the terminal to stop the server.
Local run artifacts are written to `.demo-runs/`.

### 4. Run the measured prediction demo

The marketplace demonstrates the product experience. This command runs the
core LEGO-like token predictor against the committed measurement data:

```powershell
python -m examples.composition_demo
```

No credentials or external services are required.

### 5. Run the tests

```powershell
python -m pytest -q
```

## Videos

### Hackathon demo

[Watch Token Yield Marketplace Studio](docs/media/Token_Yield_Studio_hack_video.mp4)
to see the client request, agent workflow, human approval points, marketplace
quote, and business-value story.

### Model explainer

[Watch the Token Yield model explainer](docs/media/Token_Yield_Explainer.mp4)
to see how measured task bricks are combined, trained, and used to predict a
new workload.

## How it works

```text
Client request
    -> agents find and compose task bricks
    -> humans approve scope and assumptions
    -> the model predicts tokens, cost, and risk
    -> agents deliver the work
    -> actual usage, quality, and client feedback improve future predictions
```

The model uses nine enterprise document-work bricks:

* Review
* Extract
* Classify
* Retrieve
* Reconcile
* Draft
* Remediate
* Validate
* Report

Each brick is a measured input feature. The same project object supports the
initial forecast, the itemised quote, and quote-to-actual reconciliation.

## What the experiments show

The core model was trained and evaluated with real, instrumented agent runs
instead of invented token counts.

| Measurement | Result |
| --- | ---: |
| Cross-validation on 35 measured agent runs | 2.55% average total-token error |
| Four brick combinations excluded from training | 2.2% average error |
| Three requests written in plain English | 0% to 3.5% error |
| Marketplace model on 20 calls from two held-back projects | 32.5 input-token and 35.2 output-token average error per call |

Read the [composition findings](docs/composition-findings.md), the
[plain-English training explanation](docs/lego-model-training.md), or the
[HTML results report](docs/customer-model-report.html) for the underlying
measurements and evaluation context.

## Explore the project

* [Marketplace vision and results](docs/token-yield-learning-marketplace.md)
* [Marketplace prototype guide](docs/marketplace-prototype.md)
* [Marketplace model and quoting roadmap](docs/marketplace-models/plan.md)
* [Service selection and quoting guide](docs/marketplace-models/quoting.md)
* [Model training explanation](docs/lego-model-training.md)
* [Experiment and prediction results](docs/composition-findings.md)
* [Current system architecture](docs/architecture.md)
* [Agent and model testing workflow](docs/how-it-was-tested.md)

Live Azure Foundry execution is optional and requires explicit configuration,
authorization, and budget approval. Follow the
[real Foundry agent pilot instructions](docs/marketplace-prototype.md#real-foundry-agent-pilot)
instead of enabling paid execution from the quick start.

Token Yield is released under the [MIT License](LICENSE).
