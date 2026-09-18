// Explicit local MOCK server only. Verifies the three separated sales tabs.
const assert = require("node:assert/strict");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");

const base = process.env.MARKETPLACE_URL || "http://127.0.0.1:8803";

async function json(path, options = {}) {
  const response = await fetch(base + path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(`${path} ${response.status} ${JSON.stringify(data)}`);
  return data;
}

async function waitForRun(id, predicate, timeout = 45000) {
  const deadline = Date.now() + timeout;
  let run;
  while (Date.now() < deadline) {
    run = await json(`/api/runs/${id}`);
    if (predicate(run)) return run;
    if (["failed", "cancelled"].includes(run.status)) throw new Error(`run ${run.status}: ${run.error}`);
    await new Promise(resolve => setTimeout(resolve, 150));
  }
  throw new Error(`timed out waiting for ${id}: ${JSON.stringify({status: run?.status, next: run?.next_stage})}`);
}

async function seedCatalog() {
  const existing = await json("/api/catalog");
  if ((existing.items || existing.catalog || []).some(item => item.supported === true)) return null;
  const started = await json("/api/runs", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      description: "Review fictional document requirements and identify evidence gaps for a stored catalog seed.",
      new_function: "",
      model_id: "gpt",
      runs_per_month: 1,
      execution_mode: "automatic",
      runtime: "foundry",
    }),
  });
  return waitForRun(started.id, run => run.status === "completed", 60000);
}

async function approveHumanGate(runId, actor) {
  const waiting = await waitForRun(runId, run =>
    run.status === "waiting" && run.next_stage === "human_establishment_approval");
  await json(`/api/runs/${runId}/establishment`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      decision: "approve",
      actor,
      contract_hash: waiting.establishment_request.contract_hash,
    }),
  });
  return waitForRun(runId, run => run.status === "completed", 60000);
}

async function assertMalformedSnapshotsRejectedAtomically(page) {
  const baseline = await page.evaluate(() => ({
    selections: exportData().selections,
    estimate: exportData().estimate,
    inputs: {
      inputRate: document.querySelector("#measured-input-rate").value,
      outputRate: document.querySelector("#measured-output-rate").value,
      reserve: document.querySelector("#reserve").value,
      operating: document.querySelector("#operating").value,
      cases: document.querySelector("#cases").value,
      weeks: document.querySelector("#weeks").value,
      reviewRate: document.querySelector("#reviewRate").value,
    },
    saved: JSON.parse(localStorage.getItem("token-yield-saved-build-v1")),
  }));
  const probes = [
    snapshot => { delete snapshot.estimateInputs; },
    snapshot => { delete snapshot.estimateInputs.fields.dsRate; },
    snapshot => { delete snapshot.estimateInputs.measuredRates["measured-output-rate"]; },
    snapshot => { snapshot.estimateInputs.modelScenarioRates[1] = {...snapshot.estimateInputs.modelScenarioRates[0]}; },
    snapshot => { delete snapshot.estimateInputs.modelScenarioRates[0].output; },
    snapshot => { snapshot.estimateInputs.modelScenarioRates[0].input = null; },
    snapshot => { snapshot.estimateInputs.atomicScenarioTokens.Retrieve = []; },
    snapshot => { delete snapshot.estimateInputs.atomicScenarioTokens.Retrieve; },
    snapshot => { snapshot.estimateInputs.atomicScenarioTokens.Retrieve[1] = null; },
    snapshot => { snapshot.estimateInputs.atomicScenarioTokens.Unknown = [1, 2]; },
    snapshot => { snapshot.estimateInputs.coordinator = "missing-model"; },
    snapshot => { snapshot.estimateInputs.fields.operating = null; },
    snapshot => { snapshot.estimateInputs.measuredRates["measured-input-rate"] = "NaN"; },
    snapshot => { snapshot.estimateInputs.modelScenarioRates.pop(); },
  ];
  for (const [index, mutate] of probes.entries()) {
    await page.evaluate(({saved, index}) => {
      const copy = JSON.parse(JSON.stringify(saved));
      window.__malformedMutators[index](copy);
      localStorage.setItem("token-yield-saved-build-v1", JSON.stringify(copy));
    }, {saved: baseline.saved, index});
    await page.locator("#load-build").click();
    await page.waitForFunction(() => /Saved project .*missing|invalid|duplicate|unknown|malformed|incomplete/.test(document.querySelector("#status")?.textContent || ""));
    await page.locator("#operating").evaluate(element => {
      element.value = "999.25";
      element.dispatchEvent(new Event("input", {bubbles: true}));
    });
    const current = await page.evaluate(() => ({
      selections: exportData().selections,
      estimate: exportData().estimate,
      inputs: {
        inputRate: document.querySelector("#measured-input-rate").value,
        outputRate: document.querySelector("#measured-output-rate").value,
        reserve: document.querySelector("#reserve").value,
        operating: document.querySelector("#operating").value,
        cases: document.querySelector("#cases").value,
        weeks: document.querySelector("#weeks").value,
        reviewRate: document.querySelector("#reviewRate").value,
      },
      reviewDisabled: document.querySelector("#review").disabled,
    }));
    assert.equal(current.selections.length, baseline.selections.length, `selection count changed for probe ${index}`);
    assert.deepEqual(current.selections, baseline.selections, `selections changed for probe ${index}`);
    assert.equal(current.inputs.inputRate, baseline.inputs.inputRate, `input rate changed for probe ${index}`);
    assert.equal(current.inputs.outputRate, baseline.inputs.outputRate, `output rate changed for probe ${index}`);
    assert.equal(current.estimate.monthlyApi, baseline.estimate.monthlyApi, `default-rate quote leak for probe ${index}`);
    assert.equal(current.reviewDisabled, false, `valid active build disabled by rejected probe ${index}`);
  }
  await page.evaluate(saved => localStorage.setItem("token-yield-saved-build-v1", JSON.stringify(saved)), baseline.saved);
}

async function checkViewport(width, height) {
  const browser = await chromium.launch({headless: true, channel: process.env.PLAYWRIGHT_CHANNEL || undefined});
  const context = await browser.newContext({viewport: {width, height}});
  const page = await context.newPage();
  try {
    await page.addInitScript(() => {
      window.__malformedMutators = [
        snapshot => { delete snapshot.estimateInputs; },
        snapshot => { delete snapshot.estimateInputs.fields.dsRate; },
        snapshot => { delete snapshot.estimateInputs.measuredRates["measured-output-rate"]; },
        snapshot => { snapshot.estimateInputs.modelScenarioRates[1] = {...snapshot.estimateInputs.modelScenarioRates[0]}; },
        snapshot => { delete snapshot.estimateInputs.modelScenarioRates[0].output; },
        snapshot => { snapshot.estimateInputs.modelScenarioRates[0].input = null; },
        snapshot => { snapshot.estimateInputs.atomicScenarioTokens.Retrieve = []; },
        snapshot => { delete snapshot.estimateInputs.atomicScenarioTokens.Retrieve; },
        snapshot => { snapshot.estimateInputs.atomicScenarioTokens.Retrieve[1] = null; },
        snapshot => { snapshot.estimateInputs.atomicScenarioTokens.Unknown = [1, 2]; },
        snapshot => { snapshot.estimateInputs.coordinator = "missing-model"; },
        snapshot => { snapshot.estimateInputs.fields.operating = null; },
        snapshot => { snapshot.estimateInputs.measuredRates["measured-input-rate"] = "NaN"; },
        snapshot => { snapshot.estimateInputs.modelScenarioRates.pop(); },
      ];
    });
    await page.goto(`${base}/marketplace-sales-demo.html`);
    await page.waitForFunction(() => document.querySelector("#catalog-status")?.textContent.startsWith("Stored catalog:"));
    assert.equal(await page.locator("#custom-panel").isHidden(), true);
    assert.equal(await page.locator("#saved-panel").isHidden(), true);
    assert.equal(await page.locator("#measured-marketplace").isVisible(), true);
    assert.equal(await page.locator("#manual-marketplace").isHidden(), true);
    assert.match(await page.locator("#cart-items").innerText(), /Start with one capability/);
    assert.equal(await page.locator("#review").isDisabled(), true);

    const runPosts = [];
    page.on("request", request => {
      const url = new URL(request.url());
      if (request.method() === "POST" && url.pathname === "/api/runs") runPosts.push(request.url());
    });
    const first = page.locator("[data-measured]:enabled").first();
    await first.evaluate(element => { element.closest("details[data-measured-feature]").open = true; });
    await first.check();
    await page.waitForFunction(() => document.querySelector("#quote-summary")?.textContent.includes("/ month"));
    await page.locator("[data-measured-volume]").first().fill("321");
    await page.evaluate(() => {
      for (const [id, value] of Object.entries({
        "measured-input-rate": "10",
        "measured-output-rate": "60",
        reserve: "7",
        operating: "321.25",
        cases: "654",
        weeks: "9",
        reviewRate: "77",
      })) {
        const input = document.getElementById(id);
        input.value = value;
        input.dispatchEvent(new Event("input", {bubbles: true}));
      }
    });
    let savedMonthlyApi = null;
    const savedInputs = await page.evaluate(() => ({
      inputRate: document.querySelector("#measured-input-rate").value,
      outputRate: document.querySelector("#measured-output-rate").value,
      reserve: document.querySelector("#reserve").value,
      operating: document.querySelector("#operating").value,
      cases: document.querySelector("#cases").value,
      weeks: document.querySelector("#weeks").value,
      reviewRate: document.querySelector("#reviewRate").value,
      runs: document.querySelector("[data-measured-volume]").value,
    }));
    assert.equal(runPosts.length, 0, "Tab 01 shopping must not start an agent run");
    assert.match(await page.locator("#quote-summary").innerText(), /measured|MOCK|forecast/i);

    await page.locator("#saved-tab").evaluate(element => element.click());
    await page.waitForFunction(() => !document.querySelector("#saved-panel").hidden);
    assert.equal(await page.locator("#saved-panel").isVisible(), true);
    assert.equal(await page.locator("#custom-panel").isHidden(), true);
    assert.equal(await page.locator("#measured-marketplace").isHidden(), true);
    await page.locator("#save-build").click();
    await page.waitForFunction(() => document.querySelector("#saved-state")?.textContent.includes("Saved 1 variation"));
    savedMonthlyApi = await page.evaluate(() =>
      JSON.parse(localStorage.getItem("token-yield-saved-build-v1")).savedEstimate.monthlyApi);
    await page.locator("#clear").click();
    assert.match(await page.locator("#cart-items").innerText(), /Start with one capability/);
    await page.reload();
    await page.waitForFunction(() => document.querySelector("#catalog-status")?.textContent.startsWith("Stored catalog:"));
    await page.locator("#saved-tab").click();
    await page.locator("#load-build").click();
    await page.waitForFunction(() => !document.querySelector("#cart-items")?.textContent.includes("Start with one capability"));
    await page.locator("#reestimate-saved").click();
    await page.waitForFunction(() => document.querySelector("#quote-summary")?.textContent.includes("/ month"));
    const restoredInputs = await page.evaluate(() => ({
      inputRate: document.querySelector("#measured-input-rate").value,
      outputRate: document.querySelector("#measured-output-rate").value,
      reserve: document.querySelector("#reserve").value,
      operating: document.querySelector("#operating").value,
      cases: document.querySelector("#cases").value,
      weeks: document.querySelector("#weeks").value,
      reviewRate: document.querySelector("#reviewRate").value,
      runs: document.querySelector("[data-measured-volume]").value,
    }));
    assert.deepEqual(restoredInputs, savedInputs);
    assert.notEqual(Number(restoredInputs.inputRate), 2.5);
    assert.notEqual(Number(restoredInputs.outputRate), 15);
    assert.equal(await page.evaluate(() => exportData().estimate.monthlyApi), savedMonthlyApi);
    await assertMalformedSnapshotsRejectedAtomically(page);
    await page.locator("#send-saved-custom").click();
    assert.equal(await page.locator("#custom-panel").isVisible(), true);
    assert.equal(await page.locator("#saved-panel").isHidden(), true);
    assert.equal(await page.locator("#cart-panel").isHidden(), true);
    assert.match(await page.locator("#brief").inputValue(), /saved, shopped functionalities/i);

    const created = page.waitForResponse(response =>
      new URL(response.url()).pathname === "/api/runs" && response.request().method() === "POST");
    await page.locator("#pipeline-runtime").selectOption("foundry");
    await page.locator("#execution-mode").selectOption("automatic");
    await page.locator("#new-function").fill(`Three tab governed ledger ${width}`);
    await page.locator("#run-pipeline").click();
    const creation = await (await created).json();
    assert.match(creation.id, /^[0-9a-f]{32}$/);
    const waiting = await waitForRun(creation.id, run =>
      run.status === "waiting" && run.next_stage === "human_establishment_approval");
    const ops = await context.newPage();
    await ops.goto(`${base}/marketplace-operations-demo.html?run=${creation.id}`);
    await ops.waitForSelector("#human-approval:not([hidden])");
    assert.match(await ops.locator("#human-approval").innerText(), /Approve establishment/);
    await ops.fill("#human-actor", `three-tab-${width}`);
    await ops.click("#approve-establishment");
    const completed = await waitForRun(creation.id, run => run.status === "completed", 60000);
    assert.ok(completed.result.requested_custom.startsWith("novel_"));
    await page.goto(`${base}/marketplace-sales-demo.html?run=${creation.id}`);
    await page.waitForFunction(actor =>
      document.querySelector("#capability-status")?.textContent.includes(`Human approved by ${actor}`),
      `three-tab-${width}`);
    assert.doesNotMatch(await page.locator("#capability-status").innerText(), /Awaiting human approval/);
    return {width, run: creation.id, gateHash: waiting.establishment_request.contract_hash,
            custom: completed.result.requested_custom, version: completed.result.training.version};
  } finally {
    await browser.close();
  }
}

async function main() {
  const runtime = await json("/api/runtime");
  assert.equal(runtime.source, "mocked-test-provider", "Never run this browser check on a paid server");
  await seedCatalog();
  const desktop = await checkViewport(1440, 900);
  const mobile = await checkViewport(390, 900);
  console.log(JSON.stringify({status: "PASS", desktop, mobile}, null, 2));
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
