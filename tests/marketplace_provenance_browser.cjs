// All HTTP responses are local fixtures, including the measured-provenance case.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

async function checkQuote(page, source, {reference=true, connected=false}={}) {
  await page.locator("#review").click();
  const exported=JSON.parse(await page.locator("#export-json").inputValue());
  const label=source==="measured-foundry"?"measured-data":source==="mocked-test-provider"?"mock":source==="synthetic"?"synthetic":"unknown";
  assert.equal(exported.trainingDataSource,source);
  assert.equal(exported.liveLLMConnected,connected);
  assert.equal(exported.basis,source==="measured-foundry"?"measured-data-pilot-forecast":source==="mocked-test-provider"?"mocked-provider-pilot-forecast":source==="synthetic"?"synthetic-data-forecast":"unknown-provenance-forecast");
  assert.ok(exported.selections.every(item=>item.predictionSource===source));
  assert.ok(exported.estimate.rows.every(item=>item.featureVector.source===source));
  for(const selector of ["#cart-items","#quote-summary","#review-body"]){
    const text=await page.locator(selector).innerText();
    assert.match(text,new RegExp(label,"i"),selector);
    if(source!=="measured-foundry")assert.doesNotMatch(text,/measured-data|Foundry proposed/i,selector);
  }
  if(reference){
    assert.equal(exported.estimate.routingRuns,0);
    assert.deepEqual(exported.estimate.routing,{input:0,output:0});
    assert.deepEqual(exported.estimate.setupTokens,{input:0,output:0});
    const expectedTokens=exported.selections.reduce((sum,item)=>sum+(item.backendPrediction.input_tokens+item.backendPrediction.output_tokens)*item.runs,0);
    assert.ok(Math.abs(exported.estimate.totalTokens-expectedTokens)<=Math.max(1,expectedTokens)*1e-12,"Only floating-point summation rounding may differ from frozen tokens");
    exported.estimate.rows.forEach(item=>assert.deepEqual(item.perRun,{input:item.backendPrediction.input_tokens,output:item.backendPrediction.output_tokens}));
    const rates=exported.commercialMeasuredRates;
    assert.equal(exported.estimate.monthlyApi,exported.estimate.rows.reduce((sum,item)=>sum+(item.tokens.input*rates.input+item.tokens.output*rates.output)/1e6,0));
  }
  await page.locator("#review-dialog").evaluate(element=>element.close());
  return exported;
}

async function main(){
  const {chromium}=require(process.env.PLAYWRIGHT_MODULE||"playwright");
  const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||undefined});
  const html=fs.readFileSync(path.join(__dirname,"..","marketplace-sales-demo.html"),"utf8");
  const reports=[];
  try{
    const cases=[
      ...["mocked-test-provider","measured-foundry","synthetic",null].map(source=>({source,runtimeSource:source,parentSource:source})),
      {source:"measured-foundry",runtimeSource:"mocked-test-provider",parentSource:"measured-foundry"},
      {source:"measured-foundry",runtimeSource:"measured-foundry",parentSource:"measured-foundry",enabled:false},
      {source:null,runtimeSource:"measured-foundry",parentSource:null},
      {source:"mocked-test-provider",runtimeSource:"mocked-test-provider",parentSource:"measured-foundry",expected:"unknown"}
    ];
    for(const {source,runtimeSource,parentSource,expected=source||"unknown",enabled=source!=="synthetic"} of cases){
      for(const flow of ["catalog","project"]){
        const offline=source==="synthetic";
        const connected=enabled&&runtimeSource==="measured-foundry";
        const forecast={supported:true,input_tokens:123,output_tokens:45,model_id:offline?"gpt":"gpt-5.4",version:"fixture-version",source};
        const items=["research:normal","support:faq"].map(id=>({...forecast,model_id:"gpt-5.4",id,name:id,scope:"Fixture reference context"}));
        const run={id:"fixture-run",status:"completed",events:[],request:{runtime:offline?"offline":"foundry",model_id:"gpt",runs_per_month:10,description:"Invented original project fixture"},result:{
          source,model_id:forecast.model_id,bricks:[{id:"extract",name:"Fixture source extraction",atoms:["extract"]}],after:forecast,
          training:{source:parentSource,train_count:4,test_count:2,version:"fixture-version"},features:{context:1}
        }};
        const context=await browser.newContext({viewport:{width:390,height:844}});
        const errors=[],writes=[];
        await context.route("**/*",async route=>{
          const request=route.request(),url=new URL(request.url());
          if(request.method()!=="GET"){writes.push(url.pathname);return route.abort();}
          if(url.pathname==="/marketplace-sales-demo.html")return route.fulfill({contentType:"text/html",body:html});
          const data={"/api/health":{status:"ok"},"/api/runtime":{enabled,source:runtimeSource,model_id:"gpt-5.4"},
            "/api/catalog":{source:parentSource,version:"fixture-version",items},"/api/scenarios":{scenarios:[]},"/api/runs/fixture-run":run}[url.pathname];
          if(data)return route.fulfill({json:data});
          return route.fulfill({status:404,body:"Unexpected fixture route"});
        });
        const page=await context.newPage();
        page.on("pageerror",error=>errors.push(error.message));
        await page.goto("http://127.0.0.1:8772/marketplace-sales-demo.html"+(flow==="project"?"?run=fixture-run":""));
        await page.waitForFunction(()=>document.querySelector("#catalog-status").textContent.startsWith("Stored catalog:"));
        if(offline&&flow==="catalog"){
          await page.locator("#starter").click();
          await page.locator("#review").click();
          const manual=JSON.parse(await page.locator("#export-json").inputValue());
          assert.equal(manual.basis,"illustrative-simulation");
          assert.equal(manual.trainingDataSource,null);
          assert.equal(manual.trainedPredictorConnected,false);
          assert.equal(manual.liveLLMConnected,false);
          assert.ok(manual.estimate.routingRuns>0,"Manual estimates retain illustrative routing");
          await page.locator("#review-dialog").evaluate(element=>element.close());
        }
        if(flow==="project"){
          await page.locator("#approve-pipeline").check();
          await page.locator("#use-pipeline").click();
        }else{
          for(const index of [0,1]){
            await page.locator(`details[data-measured-feature="${index?"support":"research"}"] > summary`).click();
            await page.locator(`[data-measured="${index}"]`).check();
          }
        }
        const reference=flow==="catalog"||!offline;
        const actual=await checkQuote(page,expected,{reference,connected});
        if(reference){
          await page.locator("#context").fill("8000");
          await page.locator(".catalog-settings > summary").click();
          await page.locator("#measured-input-rate").fill("5");
          const changed=await checkQuote(page,expected,{reference,connected});
          assert.equal(changed.estimate.totalTokens,actual.estimate.totalTokens,"Reference token snapshots must ignore context and rate changes");
          assert.ok(changed.estimate.monthlyApi>actual.estimate.monthlyApi,"Commercial rates change planning cost only");
        }
        if(flow==="project"){
          await page.locator('[data-configure="project-pipeline"]').first().click();
          await page.locator('#configure button[type="submit"]').click();
          await checkQuote(page,expected,{reference:!offline,connected});
        }
        assert.deepEqual(errors,[]);
        assert.deepEqual(writes,[],"Fixture test must never start model work");
        assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
        reports.push({source:source||"unknown",runtimeSource,parentSource,enabled,flow,basis:actual.basis,tokens:actual.estimate.totalTokens});
        await context.close();
      }
    }
    const report=JSON.stringify({checks:reports,noNetworkProviderCalls:true},null,2);
    fs.mkdirSync(".feedback-provenance-check",{recursive:true});
    fs.writeFileSync(path.join(".feedback-provenance-check","report.json"),report);
    console.log(report);
  }finally{await browser.close();}
}
module.exports={checkQuote};
if(require.main===module)main().catch(error=>{console.error(error);process.exitCode=1;});
