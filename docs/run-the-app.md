# Run the Token Yield app yourself

This guide starts the marketplace demo on your own machine. Everything here runs
offline and makes no paid calls unless you explicitly enable Foundry.

## Prerequisites

* Python 3.13 with the project virtual environment already created.
* Run every command from the repository root.

The examples below use the checked-in interpreter path; adjust it if your
environment lives elsewhere.

```powershell
$py = ".\.venv\Scripts\python.exe"
```

## Offline exploration (no setup, no cost)

Start the server in mock mode. It uses synthetic fixtures, never authenticates,
and never calls a provider.

```powershell
& $py -m examples.marketplace_demo_server --port 8811 --run-dir .demo-runs --mock-agents
```

Then open these pages in a browser:

* Original marketplace (instant cute-brick estimates): <http://127.0.0.1:8811/marketplace-prototype.html?variant=A>
* Sales studio (three tabs, agent pipeline): <http://127.0.0.1:8811/marketplace-sales-demo.html>
* Execution console (terminal view): <http://127.0.0.1:8811/marketplace-console.html>
* Detailed operations: <http://127.0.0.1:8811/marketplace-operations-demo.html>

What you can do offline:

* Build a product from the cute bricks and watch tokens, cost, and ROI update instantly.
* Open **Describe your idea** and submit a request to run the mock agent pipeline
  (orchestrator plus three agents), including the human approval gate for a new brick.
* Watch the console replay the pipeline stage by stage.

Mock mode has an empty measured catalog, so the sales studio **Build with bricks**
tab shows "Awaiting supported forecasts" until a measured model is published. Use
the Original marketplace page for instant offline estimates.

## Live measured run (real GPT-5.4 calls, spends money)

A live run measures real workloads, trains the predictor, and can publish a model.
It requires Azure credentials for the pinned deployment and an explicit budget.
Only submitting a request in **Describe your idea** spends; viewing pages does not.

```powershell
az login
& $py -m examples.marketplace_demo_server `
  --port 8809 `
  --run-dir .demo-runs `
  --enable-foundry `
  --agent-state-dir .feedback-paid-campaign\state `
  --agent-budget-usd 25 `
  --agent-approval-id your-new-approval-id
```

The server refuses to reuse a previous approval identifier, settles each call
against a conservative reservation, and halts on any budget, telemetry, or
contract problem. Read `docs/marketplace-models/measurements.md` before enabling
Foundry.

## Notes

* Predictions are token-cost forecasts, not billed invoices. Staffing and ROI are
  planning assumptions.
* Brick combinations are additive sums of individual forecasts, not jointly
  measured workflows.
* The console distinguishes live runs from saved replays; saved runs are labeled
  as replays.
