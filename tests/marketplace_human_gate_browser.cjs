// Explicit local MOCK server only. Verifies sales approval state after publication.
const assert = require("node:assert/strict");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");

const base = process.env.MARKETPLACE_URL || "http://127.0.0.1:8799";

async function json(path, options = {}) {
  const response = await fetch(base + path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(`${path} ${response.status} ${JSON.stringify(data)}`);
  return data;
}

async function waitForRun(id, predicate) {
  const deadline = Date.now() + 30000;
  let run;
  while (Date.now() < deadline) {
    run = await json(`/api/runs/${id}`);
    if (predicate(run)) return run;
    if (["failed", "cancelled"].includes(run.status)) throw new Error(`run ${run.status}: ${run.error}`);
    await new Promise(resolve => setTimeout(resolve, 120));
  }
  throw new Error(`timed out waiting for ${id}: ${JSON.stringify({status: run?.status, next: run?.next_stage})}`);
}

async function main() {
  const runtime = await json("/api/runtime");
  assert.equal(runtime.source, "mocked-test-provider", "Never run this browser check on a paid server");
  const name = `Governed sales state ${process.pid} ${Date.now()}`;
  const started = await json("/api/runs", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      description: "Review fictional records and build a governed evidence ledger from supplied documents only.",
      new_function: name,
      model_id: "gpt",
      runs_per_month: 1,
      execution_mode: "automatic",
      runtime: "foundry",
    }),
  });
  const waiting = await waitForRun(started.id, run =>
    run.status === "waiting" && run.next_stage === "human_establishment_approval");
  const browser = await chromium.launch({headless: true, channel: process.env.PLAYWRIGHT_CHANNEL || undefined});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 900}});
    await page.goto(`${base}/marketplace-sales-demo.html?run=${started.id}`);
    await page.waitForFunction(() =>
      document.querySelector("#capability-status")?.textContent.includes("Awaiting human approval"));
    await json(`/api/runs/${started.id}/establishment`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        decision: "approve",
        actor: "browser-state-test",
        contract_hash: waiting.establishment_request.contract_hash,
      }),
    });
    const completed = await waitForRun(started.id, run => run.status === "completed");
    assert.ok(completed.result.requested_custom.startsWith("novel_"));
    await page.goto(`${base}/marketplace-sales-demo.html?run=${started.id}`);
    await page.waitForFunction(() =>
      document.querySelector("#capability-status")?.textContent.includes("Human approved by browser-state-test"));
    const text = await page.locator("#capability-status").innerText();
    assert.doesNotMatch(text, /Awaiting human approval/);
    assert.match(text, /established/);
    console.log(JSON.stringify({
      status: "PASS",
      run: started.id,
      custom: completed.result.requested_custom,
      version: completed.result.training.version,
    }, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
