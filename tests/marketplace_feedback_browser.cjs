// Explicit local MOCK server only. Refuses paid provenance before any POST.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const {checkQuote} = require("./marketplace_provenance_browser.cjs");
const base = process.env.MARKETPLACE_URL || "http://127.0.0.1:8772";
const output = path.resolve(process.env.MARKETPLACE_EVIDENCE || ".feedback-browser-check");
fs.mkdirSync(output, {recursive:true});
async function main() {
  assert.match(base, /^http:\/\/(127\.0\.0\.1|localhost):\d+$/);
  const runtime = await (await fetch(base+"/api/runtime")).json();
  assert.equal(runtime.source, "mocked-test-provider", "Refuse non-mock runtime");
  const browser = await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||undefined});
  const errors=[], requests=[], report={base, source:runtime.source, runs:[], checks:[]};
  const context = await browser.newContext({viewport:{width:1440,height:1000}});
  context.on("page", p=>{
    p.on("pageerror", e=>errors.push(e.message));
    p.on("console", m=>{if(m.type()==="error"&&!m.text().includes("404")&&!m.text().includes("ERR_FAILED"))errors.push(m.text());});
    p.on("request", r=>{if(r.method()==="POST")requests.push(r.url());});
  });
  try {
    const sales=await context.newPage();
    await sales.goto(base+"/marketplace-sales-demo.html");
    await sales.waitForFunction(()=>document.querySelector("#runtime-status").textContent.includes("mocked-test-provider"));
    assert.equal(requests.length,0,"Loading must never start a run");
    assert.match(await sales.locator("#cart-items").innerText(),/Start with one capability/);
    assert.match(await sales.locator("#yield-summary").innerText(),/choose|build|select|—/i);
    await sales.locator("#custom-tab").click();
    assert.equal(await sales.locator("[data-scenario]").count(),3);
    await sales.locator('[data-scenario="0"]').click();
    await sales.locator("#pipeline-runtime").selectOption("foundry");
    await sales.locator("#run-pipeline").click();
    await sales.waitForFunction(()=>document.querySelector("#console-link").href.includes("?run="));
    const stepUrl=await sales.locator("#console-link").getAttribute("href");
    const consolePage=await context.newPage();
    await consolePage.goto(base+stepUrl);
    await consolePage.locator("#next").waitFor();
    await consolePage.waitForFunction(()=>!document.querySelector("#next").disabled);
    await consolePage.locator("#follow").uncheck();
    await consolePage.locator("#next").click();
    await consolePage.waitForFunction(()=>document.querySelector("#run-summary").textContent.includes("Waiting before propose"));
    assert.ok(await consolePage.locator("#terminal details").count()>5);
    assert.match(await consolePage.locator("#terminal").innerText(),/contract|reconcil/i);
    await consolePage.locator("#cancel").click();
    await consolePage.waitForFunction(()=>document.querySelector("#run-summary").textContent.includes("cancelled"));
    await sales.waitForFunction(()=>!document.querySelector("#run-pipeline").disabled);
    report.checks.push("real stage release, progression and cancellation");
    for(let index=0;index<3;index++){
      await sales.locator(`[data-scenario="${index}"]`).click();
      await sales.locator("#execution-mode").selectOption("automatic");
      if(index===2)assert.equal(await sales.locator("#new-function").inputValue(),"");
      await sales.locator("#run-pipeline").click();
      await sales.waitForFunction(()=>document.querySelector("#pipeline-result").textContent.includes("MOCK provider fixtures"),null,{timeout:120000});
      const href=await sales.locator("#console-link").getAttribute("href");
      const id=new URL("http://local"+href).searchParams.get("run");
      const snapshot=await (await fetch(base+"/api/runs/"+id)).json();
      assert.equal(snapshot.status,"completed");
      assert.equal(snapshot.result.source,"mocked-test-provider");
      assert.equal(snapshot.result.training.pilot_published,true);
      assert.equal(snapshot.result.composition.measured_combinations,false);
      assert.equal(snapshot.result.capability_reviews[0].outcome,index===1?"reused":"established");
      await consolePage.goto(base+href);
      await consolePage.waitForFunction(()=>document.querySelector("#run-summary").textContent.includes("completed"));
      assert.ok((await consolePage.locator("#sales").getAttribute("href")).includes(id));
      assert.ok((await sales.locator("#operations-link").getAttribute("href")).includes(id));
      assert.equal(await consolePage.locator("#next").isDisabled(),true);
      assert.equal(await consolePage.locator("#cancel").isDisabled(),true);
      assert.match(await consolePage.locator("#terminal").innerText(),/Publication finished/);
      report.runs.push({id,outcome:snapshot.result.capability_reviews[0].outcome,
                        reused:snapshot.result.training.reused_train_count,version:snapshot.result.training.version});
      await sales.locator("#approve-pipeline").check();
      await sales.locator("#use-pipeline").click();
      await checkQuote(sales,snapshot.result.source);
    }
    const catalogPage=await context.newPage();
    await catalogPage.goto(base+"/marketplace-sales-demo.html");
    await catalogPage.waitForFunction(()=>document.querySelector("#catalog-status").textContent.startsWith("Stored catalog:"));
    const catalogSnapshot=await (await fetch(base+"/api/catalog")).json();
    assert.equal(catalogSnapshot.source,runtime.source);
    assert.ok(catalogSnapshot.items.every(item=>item.source===runtime.source));
    const available=catalogPage.locator("[data-measured]:enabled");
    assert.ok(await available.count()>=2);
    for(let index=0;index<2;index++){
      const box=available.nth(index);
      await box.evaluate(element=>element.closest("details[data-measured-feature]").open=true);
      await box.check();
    }
    await checkQuote(catalogPage,catalogSnapshot.source);
    await catalogPage.close();
    report.checks.push("project approval and multi-brick catalog quote/export preserve mock source and frozen arithmetic");
    for(const width of [1440,390]){
      for(const [name,p] of [["sales",sales],["console",consolePage]]){
        await p.setViewportSize({width,height:width===390?844:1000});
        if(name==="console")await p.locator("#follow").uncheck();
        await p.evaluate(()=>scrollTo(0,0));
        assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${name} overflow at ${width}`);
        await p.screenshot({path:path.join(output,`${name}-${width}.png`)});
        report.checks.push(`${name} no horizontal overflow at ${width}px`);
      }
    }
    await consolePage.route("**/api/runs/*",route=>route.abort());
    await consolePage.locator("#refresh").click();
    await consolePage.waitForFunction(()=>document.querySelector("#connection").textContent.includes("Disconnected"));
    assert.equal(await consolePage.locator("#next").isDisabled(),true);
    await consolePage.unroute("**/api/runs/*");
    await consolePage.locator("#refresh").click();
    await consolePage.waitForFunction(()=>document.querySelector("#connection").textContent.includes("Saved terminal"));
    await consolePage.locator("#run-id").fill("0".repeat(32));
    await consolePage.locator("#open-run button").click();
    await consolePage.waitForFunction(()=>document.querySelector("#connection").textContent.includes("run not found"));
    assert.equal(await consolePage.locator("#terminal details").count(),0,"No stale output under another ID");
    report.checks.push("disconnection, reconnect, absent run and stale-output clearing");
    assert.deepEqual(errors,[]);
    report.checks.push("zero page errors or unexpected console errors; no paid auto-start");
    fs.writeFileSync(path.join(output,"report.json"),JSON.stringify(report,null,2));
    console.log(JSON.stringify(report,null,2));
  } finally {
    await browser.close();
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
