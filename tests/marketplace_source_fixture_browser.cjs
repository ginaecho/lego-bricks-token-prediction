// This test refuses non-mock runtimes before opening a page or creating a run.
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path");
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||"playwright");
const base=process.env.MARKETPLACE_URL||"http://127.0.0.1:8784";
const output=path.resolve(process.env.MARKETPLACE_EVIDENCE||".feedback-source-browser");
async function main(){
  const runtime=await (await fetch(base+"/api/runtime")).json();
  assert.equal(runtime.source,"mocked-test-provider","Never run this test on a paid server");
  assert.equal(runtime.source_fixture?.id,"archive-exceptions-v2");
  const scenarios=await (await fetch(base+"/api/scenarios")).json();
  assert.equal(scenarios.scenarios.length,1);
  const scenario=scenarios.scenarios[0],errors=[],escaped=[],posts=[];
  fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||undefined});
  try{
    const context=await browser.newContext({viewport:{width:1440,height:1000}});
    await context.route("**/*",route=>{
      const request=route.request();
      if(new URL(request.url()).origin!==new URL(base).origin){
        escaped.push(request.url());return route.abort();
      }
      if(request.method()==="POST")posts.push({url:request.url(),body:request.postDataJSON()});
      return route.continue();
    });
    const page=await context.newPage();
    page.on("pageerror",error=>errors.push(error.message));
    await page.goto(base+"/marketplace-sales-demo.html");
    await page.getByRole("button",{name:"02 Describe your idea",exact:true}).click();
    await page.locator('[data-scenario="0"]').click();
    assert.match(await page.locator("#scenario-heading").innerText(),/Versioned fictional policy/);
    assert.match(await page.locator("#scenario-context").innerText(),/original three reviewed cases are unchanged/);
    assert.equal(await page.locator("#brief").inputValue(),scenario.description);
    assert.equal(posts.length,0,"Loading a scenario must not execute it");
    await page.waitForFunction(()=>!document.querySelector('#pipeline-runtime option[value="foundry"]').disabled);
    await page.locator("#pipeline-runtime").selectOption("foundry");
    await page.locator("#execution-mode").selectOption("automatic");
    assert.match(await page.locator("#run-pipeline").innerText(),/MOCK agents automatic/);
    const created=page.waitForResponse(r=>new URL(r.url()).pathname==="/api/runs"&&r.request().method()==="POST");
    await page.locator("#run-pipeline").click();
    const response=await created;
    assert.ok(response.ok());
    const creation=await response.json(),id=creation.id;
    assert.match(id,/^[0-9a-f]{32}$/);
    const terminal=await context.newPage();
    terminal.on("pageerror",error=>errors.push(error.message));
    await terminal.goto(`${base}/marketplace-console.html?run=${id}`);
    if(runtime.measurement_policy){
      await terminal.waitForFunction(()=>document.querySelector("#connection").textContent.includes("live backend"));
      const firstCount=await terminal.locator("#terminal details").count();
      for(const width of [1440,390]){
        await terminal.setViewportSize({width,height:width===390?844:1000});
        assert.equal(await terminal.locator("#activity").isVisible(),true);
        await terminal.screenshot({path:path.join(output,`policy-live-${width}.png`)});
      }
      await terminal.waitForFunction(count=>document.querySelectorAll("#terminal details").length>count,firstCount);
    }
    let saved;
    for(let attempt=0;attempt<240;attempt++){
      saved=await (await fetch(`${base}/api/runs/${id}`)).json();
      if(["completed","failed","cancelled"].includes(saved.status))break;
      await new Promise(resolve=>setTimeout(resolve,500));
    }
    assert.equal(saved.status,"completed",saved.error);
    assert.equal(saved.result.scenario.id,scenario.id);
    assert.equal(saved.result.request.description,scenario.description);
    assert.equal(saved.result.source,"mocked-test-provider");
    assert.equal(saved.result.composition.measured_combinations,false);
    assert.ok(saved.result.documents.every(d=>d.id.startsWith("archive-exceptions-v2-")));
    if(runtime.measurement_policy){
      assert.equal(saved.result.measurement_policy.rounds,4);
      assert.equal(saved.result.measurement_policy.policy.cost_basis,"simulated-rate-card");
      await terminal.waitForFunction(()=>document.querySelector("#terminal").textContent.includes("Calibration reward recorded."));
      assert.match(await terminal.locator("#terminal").innerText(),/propensity=/);
      assert.match(await terminal.locator("#terminal").innerText(),/reward=/);
    }
    await terminal.waitForFunction(()=>document.querySelector("#connection").textContent.includes("Saved terminal"));
    for(const width of [1440,390]){
      await page.setViewportSize({width,height:width===390?844:1000});
      await page.locator("#scenario-heading").scrollIntoViewIfNeeded();
      await page.screenshot({path:path.join(output,`source-scenario-${width}.png`)});
      await terminal.setViewportSize({width,height:width===390?844:1000});
      await terminal.locator("#follow").check();
      await terminal.screenshot({path:path.join(output,`source-replay-${width}.png`)});
      for(const current of [page,terminal])
        assert.equal(await current.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
      assert.equal(await terminal.locator("#activity").isVisible(),false);
    }
    assert.equal(posts.length,1);assert.deepEqual(errors,[]);assert.deepEqual(escaped,[]);
    const report={runId:id,source:saved.result.source,scenario:scenario.id,status:saved.status,
      measurementPolicy:saved.result.measurement_policy||null,
      training:saved.result.training,checks:["exact v2 input plumbing","no execution on load",
        "one explicit mock run","fictional labels","desktop/mobile replay","no page errors or overflow",
        "no external browser requests","not measured combinations"]};
    fs.writeFileSync(path.join(output,"source-browser-report.json"),JSON.stringify(report,null,2));
    console.log(JSON.stringify({runId:id,scenario:scenario.id,status:saved.status,checks:report.checks},null,2));
  }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
