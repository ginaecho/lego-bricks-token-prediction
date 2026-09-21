const assert = require("node:assert/strict");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const base = process.env.MARKETPLACE_URL || "http://127.0.0.1:8765";

(async () => {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    await page.route("**/api/**", route => {
      assert.equal(route.request().method(), "GET", "Evidence inspection must not submit work");
      return route.continue();
    });
    await page.goto(`${base}/marketplace-sales-demo.html?mode=custom`);
    await page.locator("#open-completed-run").waitFor({ state: "visible" });
    const id = await page.locator("#completed-run").inputValue();
    await page.locator("#open-completed-run").click();
    await page.waitForFunction(() => document.querySelector("#pipeline-status").textContent.includes("completed"));
    await page.waitForFunction(() => {
      const box = document.querySelector("#pipeline-progress").getBoundingClientRect();
      return box.top >= 0 && box.top < innerHeight;
    }, null, { timeout: 1500 });
    assert.equal(new URL(page.url()).searchParams.get("run"), id);
    const opened = browser.contexts()[0].waitForEvent("page");
    await page.locator("#operations-link").click();
    const operations = await opened;
    await operations.waitForURL(`**/marketplace-operations-demo.html?run=${id}`);
    await operations.waitForFunction(() => document.querySelector("#status").textContent.includes("completed"));
    await operations.close();
    console.log("PASS: completed link reveals evidence on screen and opens the same saved run in Operations");

    await page.goto(`${base}/marketplace-sales-demo.html?mode=custom`);
    await page.locator("#open-completed-run").waitFor({ state: "visible" });
    await page.route("**/api/runs/*", route => route.abort("connectionrefused"));
    await page.locator("#open-completed-run").click();
    await page.waitForFunction(() => document.querySelector("#completed-training-status").textContent.includes("Cannot open"));
    assert.ok(await page.locator("#completed-training-status").isVisible());
    console.log("PASS: backend failures are reported beside the link, not hidden elsewhere");
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
