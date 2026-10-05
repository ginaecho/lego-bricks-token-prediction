# Datapoint Factory

A local demo app showing how **one brick composition becomes one training data
point**, and how all data points train the construction-token model. It uses an
agent-interaction graph:

```text
① Create one data point (isolated agent build)
Solution designer ─▶ Composition planner ─▶ Builder brief ─▶ Builder agent (LLM) ⇄ Sandbox tools
                          │                                        │
                          └──── pre-build features ───▶ Data point ◀── Usage collector ◀── Runtime verifier
② Learn from all data points
Data point ─▶ Feature engineer ─▶ Grouped splitter ─▶ Ridge trainer ─▶ Validation gate ─▶ Token model ─▶ designer
```

The page reads top to bottom, full width:

1. **Basic brick**: seven Studio functionalities in one row, with variant chips.
   Beside it, the **default intent** shows the goal, the exact task line given to the
   builder, its core operations, the "done when" rule, the deliverables and the measured
   standalone tokens. The **full builder prompt** is shown next to it. Switch tabs to
   see the composed prompt including your additional requirements. For example,
   Research means: *Turn questions and source material into decision-ready evidence*,
   through extract → retrieve → verify → write, with at least 6 passing tests.
2. **Additional requirements**: up to three more bricks (in execution order), each
   showing its task, plus an industry context with its entities, constraints, formats
   and fixtures.
3. **Run the builder**:
   - **Live build on Azure Foundry** (default). A real LLM agent on your Foundry
     deployment receives the prompt and works through `write_file`, `read_file`,
     `run_tests`, `run_cli` and `finish`. Each request's API-reported usage streams into
     the graph. The app then independently re-runs the tests and CLI and saves the new
     data point.
   - **Replay a measured corpus build**: replays the recorded gpt-6-astra session of
     that membership, one request at a time.

Below the graph, the **inspector** shows the selected node's output (charts and data)
next to the **code running that stage**. Each function involved is shaded, with its
file and line range: the app's own caller first, then `main`'s functions it calls.
## Nothing is simulated

- Token counts come from each data point's `build_token_usage.events`, i.e. the
  Copilot runtime's `assistant_usage_events` for that builder agent.
- Build specs, builder instructions and features come from `main`'s
  `token_yield.build_waves_v3.build_spec_v3` and `instructions`.
- Training calls `main`'s `examples/train_construction_model.py` functions
  (`feature_names`, `matrix`, `grouped_cv`, `fit_predict`, `interval_bounds`).
  It reproduces the published wave-3 numbers: alpha 100, validation MAE 25,400,
  MAPE 14.2%.
- The test result is read from the stored report and never re-evaluated.
- **Forecast before this point** refits the model without that membership's
  group, giving the quote we would have made before measuring it.
- Compositions with no measured build are marked **not yet measured**. The app
  shows a forecast only and does not invent a build.

## Live builds on Azure Foundry

| Setting | Default |
| --- | --- |
| Project endpoint | `https://foundary-tzuc06.services.ai.azure.com/api/projects/firstProject` |
| API endpoint (OpenAI v1) | `https://foundary-tzuc06.openai.azure.com/openai/v1` |
| Deployments offered | `gpt-5-mini` (default), `gpt-5.4`, `gpt-5.6-sol` |
| Authentication | `AZURE_OPENAI_API_KEY` from the server environment, otherwise `az login` pinned to subscription `ef669702-542a-4abc-95a6-edf9f972cd3c` (tenant `16b3c013-d300-468d-ac64-7eda0820b6d3`, Microsoft Non-Production) |

- Token label = sum of `prompt_tokens` + `completion_tokens` over every chat request.
  Cached prompt tokens are shown separately and are already included in input.
- Safety limits: one live build at a time, at most 30 requests and 900,000 tokens.
- Each run is saved to `demo\datapoint-factory\live_runs\<point id>\`: the four
  deliverables and `data_point.json` (features, usage events, evidence). This folder
  is git-ignored.
- Live points are **not pooled** into the corpus fit: their builder model and agent
  harness differ from the gpt-6-astra Copilot builds. The model view compares the
  forecast with the live label.
- Model-written code runs locally in its own folder with timeouts, an isolated
  interpreter and no credentials in its environment. This is not a security sandbox.
## Token Yield Studio marketplace

The app also starts `main`'s **Token Yield Studio** (`examples.marketplace_demo_server`)
from a clean export of `main`. Open it from the **Token Yield Studio ↗** button in the
header, or at <http://127.0.0.1:8766/marketplace-sales-demo.html>.

- **Live Foundry mode is on** by default: `--enable-foundry` with the pinned `gpt-5.4`
  deployment from `experiments/customer_requests/pilot.json`, a **$10 campaign cap**
  (`--studio-budget-usd`, at most 25) and approval ID
  `ginaecho-datapoint-factory-studio-20261005`.
- Campaign budget state lives in `demo\datapoint-factory\.studio-runs\agent-state`
  (git-ignored). It persists across restarts, so restarting cannot bypass the cap.
  The Studio log is `.studio-runs\studio.log`.
- At startup the app checks that the `az` default account is the pinned subscription
  and tenant, and switches to it if needed. The Studio requests its tokens through that
  default account.
- In Foundry mode, `main`'s Studio does not load the offline published predictor.
  Use `--studio-offline` for offline trained inference with no paid calls.
- Port 8765 is used by the SkillC app, so the Studio uses 8766. The app refuses to
  start the Studio on a port that is already in use.
## Where the training data lives

| Branch | Path | What it is |
| --- | --- | --- |
| `main` | `data/build_simulations/data_points/*.json` | **299 measured builds** (waves 1–3) plus 14 reference-only real cases. These are the construction-token model's training records. |
| `main` | `data/build_simulations/builds/<id>/` | Each build's implementation, tests, example input and manifest |
| `main` | `data/build_simulations/data_points.csv` | Flat overview of the records |
| `main` | `data/build_simulations/models/construction_model_wave3*.json` | Trained artifact and validation/test report |
| `main` | `data/build_simulations/waves/wave{2,3}.json` | Frozen trial manifests and evaluation gates |
| `gc/model-improvement`, `gc/model-imrpovement` | `runs/2026082*_wave{1,2,3}/*.jsonl`, `experiments/**/*.jsonl` | Earlier **workload-execution** records (live Foundry calls of fetch/extract bricks): a different target, never pooled with build tokens |
| `gc/prototype` | `experiments/engagements.jsonl`, `examples/usecases/manifest.jsonl` | Prototype engagement and use-case records |

## Run

From the repository root (Python 3.12+, `numpy`, `scikit-learn`, `openai`, `azure-identity`):

```powershell
python demo\datapoint-factory\app.py
```

The app opens <http://127.0.0.1:8777>. On first start it exports the corpus and
code from `main` with `git archive` into `demo\datapoint-factory\.cache\<commit>`
(ignored by git) and fits the model. This takes about 30–40 seconds.

| Option | Meaning |
| --- | --- |
| `--ref REF` | Git ref holding `data/build_simulations` (default `main`) |
| `--source-root PATH` | Use an existing checkout instead of exporting `--ref` |
| `--foundry-project-endpoint URL` / `--azure-openai-endpoint URL` | Foundry endpoints for live builds |
| `--deployments a,b,c` | Deployments offered for live builds (first is default) |
| `--azure-subscription-id ID` / `--azure-tenant-id ID` | Pinned Azure account (defaults above) |
| `--no-studio` / `--studio-offline` | Do not launch the Studio / launch it without paid calls |
| `--studio-port N` / `--studio-budget-usd X` / `--studio-approval-id ID` | Studio port (8766), campaign cap (10) and approval |
| `--port N` | Port (default 8777) |
| `--no-browser` | Do not open a browser |

The server binds to localhost only. API keys are never sent to or accepted from the browser.

## Files

| File | Role |
| --- | --- |
| `app.py` | HTTP server and CLI |
| `studio.py` | Exports `main` and launches the Token Yield Studio; pins the az account |
| `source_tree.py` | Exports the corpus and Python packages from a git ref |
| `corpus.py` | Brick catalog and measured-build index |
| `foundry.py` | Foundry client; one usage event per chat request |
| `live_builder.py` | Live builder agent loop, tools and job registry |
| `sandbox.py` | Builder workspace and independent verification |
| `intents.py` | Default intent per brick and variant, rendered from main's builder template |
| `training.py` | Feature engineering, grouped CV, validation and forecasts using main's functions |
| `pipeline.py` | Builds the per-stage agent graph for one composition |
| `static/graph.js` | SVG graph layout and request pulses |
| `static/charts.js` | Token timeline, corpus scatter, alpha curve, predicted-vs-actual, feature charts |
| `static/inspector.js` | Node detail panel |
| `static/app.js` | Brick composer and replay orchestration |
