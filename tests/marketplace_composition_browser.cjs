// Read-only end-to-end check against the measured shopping campaign.
const assert = require("node:assert/strict");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const base = process.env.MARKETPLACE_URL || "http://127.0.0.1:8765";

(async () => {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    const errors = [], writes = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", route => {
      const inference = route.request().method() === "POST" && new URL(route.request().url()).pathname === "/api/forecast";
      if (route.request().method() !== "GET" && !inference) {
        writes.push(route.request().url());
        return route.abort();
      }
      return route.continue();
    });
    await page.goto(`${base}/marketplace-sales-demo.html`);
    await page.waitForFunction(() => document.querySelector("#catalog-status").textContent.startsWith("Stored catalog:"));
    assert.equal(await page.evaluate(() => catalogMetadata.composition_model?.accepted), true,
      "Publish an accepted generalizing model before declaring the Sales Studio ready");
    async function edit(feature, variant, volume) {
      await page.locator(`[data-measured-configure="${feature}"]`).click();
      await page.locator(`#pick-${variant}`).check();
      if (volume !== undefined) await page.locator(`[name="runs-${variant}"]`).fill(String(volume));
      await page.getByRole("button", { name: "Save selections", exact: true }).click();
    }
    const variants=await page.evaluate(()=>CATALOG.filter(feature=>!feature.backendPrediction).flatMap(feature=>feature.variants.map(variant=>({feature:feature.id,variant:variant.id}))));
    assert.equal(variants.length,16);
    for(const {feature,variant} of variants){
      await edit(feature,variant);
      const result=await page.evaluate(()=>latestEstimate);
      assert.ok(result&&Number.isFinite(result.monthlyApi)&&Number.isFinite(result.roi),`${feature}:${variant} must predict immediately`);
      await page.locator(`[data-remove="${feature}:${variant}"]`).click();
    }
    assert.equal(await page.locator("#measured-catalog .card").count(),7,"Internal workflows are not extra untrained capability cards");
    assert.doesNotMatch(await page.locator("#measured-catalog").innerText(),/Awaiting supported forecasts/);
    console.log("PASS: all 16 individual subtypes produce finite token costs and ROI through UI clicks");
    await edit("recommend", "behavior");
    const started = Date.now();
    await edit("search", "compare");
    const snapshot = await page.evaluate(() => exportData());
    assert.equal(snapshot.selections.length, 2);
    assert.ok(snapshot.estimate.composition);
    assert.deepEqual(snapshot.estimate.composition.component_contracts, ["behaviorx1", "comparex1"]);
    assert.equal(snapshot.estimate.standaloneForecastsSummed, false);
    const expected = snapshot.estimate.composition;
    assert.ok(Math.abs(snapshot.estimate.monthlyApi - expected.usd_per_run * 1000) < 1e-6);
    assert.ok(Math.abs(snapshot.estimate.totalTokens - expected.total_tokens * 1000) < 1e-6);
    assert.equal(snapshot.estimate.runs, 1000);
    assert.ok(Number.isFinite(snapshot.estimate.roi));
    assert.equal(snapshot.estimate.rows[0].api, null, "Do not invent a component cost allocation");
    assert.doesNotMatch(await page.locator("#quote-summary").innerText(), /pending/i);
    assert.match(await page.locator("#quote-summary").innerText(), /12-month ROI/);
    assert.ok(Date.now() - started < 5000, "Selection should resolve locally, not retrain");
    await edit("recommend", "behavior", 2000);
    assert.deepEqual(await page.evaluate(() => selections.map(item=>item.runs)), [2000,2000],
      "The explicitly shared workflow volume applies to every component");
    assert.ok(Math.abs((await page.evaluate(() => latestEstimate.monthlyApi)) - expected.usd_per_run * 2000) < 1e-6);
    await page.locator('#review').click();
    assert.match(await page.locator("#review-body").innerText(), /Not allocated/);
    assert.match(await page.locator("#review-body").innerText(), /Complete measured workflow/);
    assert.doesNotMatch(await page.locator("#review-body").innerText(), /Integration routing/);
    await page.locator('[data-close="review-dialog"]').click();
    const feedback = await page.evaluate(() => {
      let url;
      const original = window.open;
      window.open = target => { url = target; return {}; };
      try { openDeliveryFeedback(); } finally { window.open = original; }
      return JSON.parse(new URLSearchParams(new URL(url).hash.slice(1)).get("handoff"));
    });
    assert.ok(feedback.functions.every(item => item.estimated_tokens === null));
    assert.ok(feedback.estimated_project_tokens > 0);
    await page.locator('[data-remove="search:compare"]').click();
    assert.equal(await page.evaluate(() => latestEstimate.composition), null);
    await page.locator('[data-remove="recommend:behavior"]').click();
    for (let index = 0; index < variants.length; index++) {
      const pair = [variants[index], variants[(index + 5) % variants.length]];
      for (const item of pair) await edit(item.feature, item.variant);
      const started = Date.now();
      await page.waitForFunction(() => latestEstimate?.composition && Number.isFinite(latestEstimate.roi), null, { timeout: 1500 });
      const estimate = await page.evaluate(() => latestEstimate);
      assert.deepEqual(estimate.composition.component_contracts, pair.map(item => `${item.variant}x1`));
      assert.equal(estimate.standaloneForecastsSummed, false);
      assert.ok(estimate.totalTokens > 0 && estimate.monthlyApi > 0);
      assert.ok(Date.now() - started < 1000, "Local inference must finish within one second after selection");
      assert.doesNotMatch(await page.locator("#quote-summary").innerText(), /pending|not supported|not predicted/i);
      for (const item of pair) await page.locator(`[data-remove="${item.feature}:${item.variant}"]`).click();
    }
    console.log("PASS: unseen compositions spanning all 16 subtypes infer locally within one second");
    const larger = [["recommend","journey"],["search","compare"],["documents","extract"],
      ["support","faq"],["insights","sentiment"]];
    for(const [feature,variant] of larger){
      await edit(feature,variant);
      await page.waitForFunction(()=>latestEstimate&&Number.isFinite(latestEstimate.roi),null,{timeout:1500});
    }
    assert.equal((await page.evaluate(()=>latestEstimate.composition.component_contracts)).length,5);
    assert.ok(await page.evaluate(()=>latestEstimate.totalTokens>0&&latestEstimate.monthlyApi>0));
    console.log("PASS: five-brick build keeps immediate direct-workflow cost and ROI");
    const savedCost=await page.evaluate(()=>{saveCurrentBuild();return latestEstimate.monthlyApi;});
    await page.reload();
    await page.waitForFunction(()=>catalogMetadata.composition_model?.accepted);
    assert.equal(await page.evaluate(()=>loadSavedBuild()),true);
    await page.waitForFunction(()=>latestEstimate?.composition&&latestEstimate.rows.length===5,null,{timeout:1500});
    assert.ok(Math.abs(await page.evaluate(()=>latestEstimate.monthlyApi)-savedCost)<1e-9);
    await page.locator("#refresh-catalog").click();
    await page.waitForFunction(()=>latestEstimate?.composition&&latestEstimate.rows.length===5,null,{timeout:1500});
    assert.ok(Math.abs(await page.evaluate(()=>latestEstimate.monthlyApi)-savedCost)<1e-9);
    console.log("PASS: saved build, page reload and model refresh retain the same direct forecast");
    assert.deepEqual(errors, []);
    assert.deepEqual(writes, []);
    console.log("PASS: two visible bricks, instant direct-workflow forecast, ROI, volume edits, removal, no paid calls");
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
