// Every request is intercepted. Fixtures exercise rendering, never a paid server.
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path");
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||"playwright");
const html=fs.readFileSync(path.join(__dirname,"..","marketplace-console.html"),"utf8");
const output=path.resolve(process.env.MARKETPLACE_EVIDENCE||".feedback-terminal-browser");
const first="a".repeat(32),second="b".repeat(32),base="http://terminal.test";
const event=(seq,stage,message,data={})=>({seq,time:"2026-09-17T08:00:01.123456+02:00",stage,message,data});
const waiting=()=>({id:first,status:"waiting",next_stage:"novelty",request:{runtime:"foundry"},
  events:[event(1,"novelty","Operation awaiting execution gate.",{source:"mocked-test-provider"})],result:null,error:null});
async function main(){
  fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||undefined});
  const reports=[];
  try{
    for(const width of [1440,390]){
      const context=await browser.newContext({viewport:{width,height:width===390?844:1000}});
      let snapshot=waiting(),unavailable=false,missing=false,malformed=false,invalidStatus=false,rejectNext=false,delayNext=false,held=null,reads=0;
      const writes=[],errors=[],unexpected=[];
      await context.route("**/*",async route=>{
        const req=route.request(),url=new URL(req.url());
        if(url.origin!==base){unexpected.push(req.url());return route.abort();}
        if(url.pathname==="/marketplace-console.html")return route.fulfill({contentType:"text/html",body:html});
        if(url.pathname==="/favicon.ico")return route.fulfill({status:204,body:""});
        if(req.method()==="POST"){
          writes.push({path:url.pathname,body:req.postDataJSON()});
          if(url.pathname===`/api/runs/${first}/next`){
            assert.deepEqual(req.postDataJSON(),{stage:"novelty"});
            if(rejectNext)return route.fulfill({status:409,json:{error:"Execution gate changed"}});
            snapshot={...snapshot,status:"running",next_stage:null};
          }else if(url.pathname===`/api/runs/${first}/cancel`)snapshot={...snapshot,status:"cancelled",next_stage:null};
          else{unexpected.push(req.url());return route.fulfill({status:400,json:{error:"Unexpected write"}});}
          return route.fulfill({json:{id:first}});
        }
        if(url.pathname.startsWith("/api/runs/")){
          reads++;
          if(delayNext){delayNext=false;held=route;return;}
          if(unavailable)return route.abort();
          if(missing)return route.fulfill({status:404,json:{error:"run not found"}});
          if(invalidStatus)return route.fulfill({json:{...snapshot,status:"unknown-status"}});
          return route.fulfill({json:malformed?{id:snapshot.id,status:"running",events:[null]}:snapshot});
        }
        unexpected.push(req.url());return route.fulfill({status:404,body:"Unexpected fixture route"});
      });
      const page=await context.newPage();
      page.on("pageerror",e=>errors.push(e.message));
      await page.goto(`${base}/marketplace-console.html?run=${first}`);
      await page.waitForFunction(()=>document.querySelector("#run-summary").textContent.includes("Waiting before novelty"));
      assert.equal(writes.length,0);
      assert.equal(await page.locator("#activity").isVisible(),true);
      assert.match(await page.locator("#activity-text").innerText(),/waiting before novelty/);
      assert.match(await page.locator("#title").innerText(),/^waiting/);
      assert.equal(await page.locator("#next").isEnabled(),true);
      assert.ok((await page.locator("#operations").getAttribute("href")).includes(first));
      assert.equal(await page.locator(".brick,.badge,h2").count(),0,"No explanatory card layout");
      assert.equal(await page.locator("#viewport").evaluate(el=>el.clientHeight>innerHeight*.55),true,"Terminal is primary surface");
      assert.equal(await page.locator("body").evaluate(el=>getComputedStyle(el).backgroundColor),"rgba(0, 0, 0, 0)");
      assert.equal(await page.locator("html").evaluate(el=>getComputedStyle(el).backgroundColor),"rgb(17, 24, 22)");
      await page.locator("#next").click();
      await page.waitForFunction(()=>document.querySelector("#title").textContent.startsWith("mock running"));
      assert.equal(writes.length,1);
      assert.equal(await page.locator("#next").isDisabled(),true);
      await page.evaluate(()=>{globalThis.firstEvent=document.querySelector("#terminal details");});
      snapshot.events.push(event(2,"novelty","Agent public message.",{role:"architect",source:"mocked-test-provider",
        public_output:{decision:"establish",new_name:"Fixture ledger",atoms:{extract:2,verify:1},agreed:true,dissent:[]}}));
      await page.waitForFunction(()=>document.querySelectorAll("#terminal details").length===2);
      assert.equal(await page.evaluate(()=>globalThis.firstEvent===document.querySelector("#terminal details")),true);
      assert.match(await page.locator("#terminal").innerText(),/08:00:01\+02:00.*log novelty \/ architect/s);
      assert.match(await page.locator("#terminal").innerText(),/decision="establish"/);
      assert.doesNotMatch(await page.locator("#terminal").innerText(),/^\s*\$/m,"Internal events never get shell prompts");
      assert.equal(await page.locator("#terminal").evaluate(el=>el.textContent.includes("input_mae")),false,"No invented metrics");
      await page.locator("#refresh").click();
      await page.waitForTimeout(150);
      assert.equal(await page.locator("#terminal details").count(),2,"Refresh does not replay duplicate lines");

      for(let i=3;i<=26;i++)snapshot.events.push(event(i,"measure","Validated measurement row persisted.",
        {brick_id:"fixture-ledger",split:i<25?"train":"holdout",input_tokens:123,output_tokens:45,features:{extract:2,verify:1},source:"mocked-test-provider"}));
      await page.waitForFunction(()=>document.querySelectorAll("#terminal details").length===26);
      assert.equal(await page.locator("#viewport").evaluate(el=>el.scrollHeight-el.scrollTop-el.clientHeight<32),true,"New events follow the bottom");
      await page.locator("#follow").uncheck();
      await page.locator("#viewport").evaluate(el=>{el.scrollTop=0;});
      snapshot.events.push(event(27,"measure","Frozen source-group split before measurements and fitting.",
        {new_workload_calls:6,reused_train_rows:4,decision_id:"fixture-policy:1",action:"ordered_fixture",
          propensity:.9,reward:-.25,before_mae:4,after_mae:5,cost_usd:.001,
          cost_basis:"simulated-rate-card",updates:2,
          probabilities:{ordered_fixture:.9,brick:.1},scores:{ordered_fixture:{n:2,q:-.25}},
          baseline:{policy:"uniform-random"}}));
      await page.waitForFunction(()=>document.querySelectorAll("#terminal details").length===27);
      assert.match(await page.locator("#terminal").innerText(),/propensity=0.9/);
      assert.match(await page.locator("#terminal").innerText(),/reward=-0.25/);
      assert.match(await page.locator("#terminal").innerText(),/simulated-rate-card/);
      assert.match(await page.locator("#terminal").innerText(),/uniform-random/);
      assert.equal(await page.locator("#viewport").evaluate(el=>el.scrollTop),0,"Paused user position is preserved");
      assert.match(await page.locator("#scroll-state").innerText(),/paused/);
      await page.locator("#follow").check();
      assert.equal(await page.locator("#viewport").evaluate(el=>el.scrollHeight-el.scrollTop-el.clientHeight<32),true);
      await page.locator("#viewport").evaluate(el=>{el.scrollTop=0;});
      await page.waitForFunction(()=>!document.querySelector("#follow").checked);
      snapshot.events.push(event(28,"features","Actual numeric predictors computed.",
        {builder:{builder:"atoms_context_v1"},features:{extract:2,verify:1,context_bytes:217}}));
      snapshot.events.push(event(29,"train","Fit real local input/output ridge models; candidate is not production-promoted.",
        {train_count:8,alpha:1,version:"fixture-v1"}));
      snapshot.events.push(event(30,"evaluate","Fresh held-out outcomes evaluated after model/configuration freeze.",
        {input_mae:12.5,output_mae:4.25,test_count:2,review:{accepted:false,limitations:["MOCK insufficient evidence"]}}));
      const dissent="Unsupported source claim. ".repeat(20)+"Do not suppress this final objection.";
      snapshot.events.push(event(31,"discuss","Agent public message.",{role:"skeptical_reviewer",
        public_output:{agreed:true,dissent:[dissent],summary:"<img src=x onerror=alert('unsafe')>"}}));
      snapshot.events.push(event(32,"novelty","Capability reused; no duplicate contract created.",
        {outcome:"reused",brick_id:"fixture-ledger"}));
      await page.waitForFunction(()=>document.querySelectorAll("#terminal details").length===32);
      const text=await page.locator("#terminal").innerText();
      for(const snippet of ['features {"extract":2','new_workload_calls=6','reused_train_rows=4','train_count=8',
        'test_count=2','input_mae=12.5','output_mae=4.25','metric review: rejected','outcome="reused"',dissent]){
        assert.ok(text.includes(snippet),snippet);
      }
      assert.equal(await page.locator("#terminal img").count(),0);
      assert.match(await page.locator(".dissent").last().innerText(),/Do not suppress this final objection/);
      await page.locator("#follow").check();
      await page.screenshot({path:path.join(output,`terminal-live-${width}.png`)});
      assert.equal(await page.locator(".result").first().evaluate(el=>getComputedStyle(el).color),"rgb(240, 170, 106)");

      snapshot={...snapshot,status:"failed",error:"Provider telemetry incomplete; reservation retained.",next_stage:null};
      await page.waitForFunction(()=>document.querySelector("#run-summary").textContent.includes("failed"));
      assert.equal(await page.locator("#activity").isVisible(),false);
      assert.match(await page.locator("#connection").innerText(),/Saved terminal.*replay/);
      assert.match(await page.locator("#run-error").innerText(),/reservation retained/);
      assert.equal(await page.locator("#cancel").isDisabled(),true);
      const stoppedReads=reads;await page.waitForTimeout(900);assert.equal(reads,stoppedReads,"Saved failure does not poll or animate");
      await page.screenshot({path:path.join(output,`terminal-error-${width}.png`)});

      unavailable=true;await page.locator("#refresh").click();
      await page.waitForFunction(()=>document.querySelector("#connection").textContent.includes("Disconnected"));
      assert.equal(await page.locator("#terminal details").count(),32);
      assert.equal(await page.locator("#activity").isVisible(),false);
      assert.match(await page.locator("#run-summary").innerText(),/stale snapshot/);
      unavailable=false;snapshot={...snapshot,replay:true,status:"running",error:null};
      await page.locator("#refresh").click();
      await page.waitForFunction(()=>document.querySelector("#connection").textContent.includes("Persisted replay"));
      assert.equal(await page.locator("#activity").isVisible(),false,"Persisted active-status snapshot is not live");
      const replayReads=reads;await page.waitForTimeout(900);assert.equal(reads,replayReads);

      snapshot={...snapshot,replay:false,status:"completed",result:{source:"mocked-test-provider",
        training:{pilot_published:true,version:"fixture-v1"}}};
      snapshot.events.push(event(33,"complete","Publication finished.",{published:true,version:"fixture-v1",production_promoted:false}));
      await page.locator("#refresh").click();
      await page.waitForFunction(()=>document.querySelectorAll("#terminal details").length===33);
      assert.match(await page.locator("#terminal").innerText(),/published=true/);
      assert.equal(await page.locator("#activity").isVisible(),false);
      assert.match(await page.locator("#run-summary").innerText(),/Published: true/);
      // An older overlapping response must never replace a newer snapshot.
      delayNext=true;await page.locator("#refresh").click();
      await page.waitForTimeout(100);assert.ok(held);
      await page.locator("#refresh").click();
      await page.waitForTimeout(100);
      await held.fulfill({json:waiting()});held=null;
      await page.waitForTimeout(100);
      assert.equal(await page.locator("#terminal details").count(),33);
      assert.match(await page.locator("#run-summary").innerText(),/completed/);

      snapshot={...waiting(),id:second,status:"cancelled",next_stage:null,
        events:[event(1,"novelty","Cancelled fixture run.")],result:null};
      await page.locator("#run-id").fill(second);
      await page.locator("#open-run button").click();
      await page.waitForFunction(id=>document.querySelector("#run-summary").textContent.includes(id),second);
      assert.equal(await page.locator("#terminal details").count(),1,"Run switch clears previous evidence");
      assert.match(await page.locator("#title").innerText(),/^replay/);
      assert.equal(await page.locator("#activity").isVisible(),false);
      malformed=true;await page.locator("#refresh").click();
      await page.waitForFunction(()=>document.querySelector("#connection").textContent.includes("Invalid run snapshot"));
      assert.equal(await page.locator("#terminal details").count(),1);
      malformed=false;invalidStatus=true;await page.locator("#refresh").click();
      await page.waitForTimeout(150);
      assert.match(await page.locator("#connection").innerText(),/Invalid run snapshot/);
      assert.equal(await page.locator("#activity").isVisible(),false);
      invalidStatus=false;missing=true;await page.locator("#refresh").click();
      await page.waitForFunction(()=>document.querySelector("#connection").textContent.includes("run not found"));
      assert.equal(await page.locator("#next").isDisabled(),true);
      missing=false;snapshot=waiting();
      await page.goto(`${base}/marketplace-console.html?run=${first}`);
      await page.waitForFunction(()=>!document.querySelector("#next").disabled);
      rejectNext=true;await page.locator("#next").click();
      await page.waitForFunction(()=>document.querySelector("#connection").textContent.includes("Control rejected"));
      assert.match(await page.locator("#run-summary").innerText(),/stale snapshot/);
      assert.equal(await page.locator("#activity").isVisible(),false);
      assert.equal(await page.locator("#next").isDisabled(),true);
      rejectNext=false;await page.locator("#refresh").click();
      await page.waitForFunction(()=>!document.querySelector("#cancel").disabled);
      await page.locator("#cancel").click();
      await page.waitForFunction(()=>document.querySelector("#run-summary").textContent.includes("cancelled"));
      assert.equal(await page.locator("#activity").isVisible(),false);
      assert.deepEqual(writes.map(w=>w.path),[`/api/runs/${first}/next`,`/api/runs/${first}/next`,`/api/runs/${first}/cancel`]);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
      assert.deepEqual(errors,[]);assert.deepEqual(unexpected,[]);
      reports.push({width,checks:["incremental append","stable DOM","follow and user pause","roles and timestamps",
        "actual fixture numeric fields","complete dissent","escaped output","live waiting cursor","failure",
        "disconnection","replay never polls","publication","stale response guard","run switch","malformed/missing",
        "invalid status rejected","rejected control marks snapshot stale","explicit controls only","no horizontal overflow"],fixturePosts:writes.length,networkEscapes:0});
      await context.close();
    }
    fs.writeFileSync(path.join(output,"terminal-report.json"),JSON.stringify(reports,null,2));
    console.log(JSON.stringify(reports,null,2));
  }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
