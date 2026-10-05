// Brick composer, replay / live-build orchestration and graph animation.

import { createGraph } from "./graph.js";
import { renderInspector } from "./inspector.js";

const $ = (id) => document.getElementById(id);
const fmt = (n) => Math.round(n).toLocaleString();
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const post = (url, body) => fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
  .then(async (r) => { const data = await r.json(); if (!r.ok) throw new Error(data.error); return data; });

const ctx = { catalog: null, corpus: [], training: null, provenance: "", result: null };
const state = { base: null, extras: [], industry: "", promptTab: "brick" };
let running = false;
const graph = createGraph($("graph"), (id) => showStage(id));

/* ---------- composition model ---------- */
const bricksById = () => Object.fromEntries(ctx.catalog.bricks.map((b) => [b.id, b]));
const parts = () => [state.base, ...state.extras].filter(Boolean).map((i) => i.part);
const usedFamilies = () => [state.base, ...state.extras].filter(Boolean).map((i) => i.family);
const level = () => {
  const generic = parts().every((p) => p.startsWith("basic:"));
  return state.industry ? (generic ? "BI" : "BTI") : (generic ? "B" : "BT");
};
const variantOptions = (family) => [{ id: `basic:${family}`, name: "Generic (no variant)" }, ...bricksById()[family].variants];

/* ---------- step 1: bricks, intent and prompt ---------- */
function renderBricks() {
  $("bricks").innerHTML = "";
  ctx.catalog.bricks.forEach((b) => {
    const button = document.createElement("button");
    const used = usedFamilies().includes(b.id) && state.base?.family !== b.id;
    button.className = `brick${state.base?.family === b.id ? " on" : ""}${used ? " used" : ""}`;
    button.innerHTML = `<b>${b.name}</b><small>${b.category}${used ? " · in requirements" : ""}</small>`;
    button.onclick = () => {
      state.extras = state.extras.filter((e) => e.family !== b.id);
      state.base = { family: b.id, part: `basic:${b.id}` };
      changed();
    };
    $("bricks").appendChild(button);
  });
  $("variant-row").innerHTML = `<span class="label">VARIANT</span>`;
  variantOptions(state.base.family).forEach((v) => {
    const chip = document.createElement("button");
    chip.className = `chip${state.base.part === v.id ? " on" : ""}`;
    chip.textContent = v.name;
    chip.onclick = () => { state.base.part = v.id; changed(); };
    $("variant-row").appendChild(chip);
  });
}

function renderIntent() {
  const brick = bricksById()[state.base.family], intent = ctx.catalog.intents[state.base.part];
  const ops = intent.operations.map((o) => `<li><b>${o.operation}${o.count > 1 ? ` ×${o.count}` : ""}</b> — ${o.meaning}</li>`).join("");
  const measured = intent.measured_builds
    ? `${intent.measured_builds} measured standalone build(s) · mean ${fmt(intent.mean_tokens)} tokens` : "Not yet measured as a standalone build";
  $("intent-card").innerHTML = `<div class="intent">
    <h4>Default intent · ${esc(brick.name)} ${state.base.part.startsWith("basic:") ? "(generic)" : `· ${esc(state.base.part)}`}</h4>
    <p class="goal">${esc(brick.description)}</p>
    <h5>Task given to the builder</h5><p class="task">${esc(intent.task)}</p>
    <h5>Core operations</h5><ul>${ops}</ul>
    <h5>Done when</h5><p>${esc(intent.acceptance)}</p>
    <h5>Deliverables</h5><p>${esc(intent.deliverables)}</p>
    <p class="measured">${measured}</p></div>`;
}

async function renderPrompt() {
  $("tab-brick").classList.toggle("on", state.promptTab === "brick");
  $("tab-composed").classList.toggle("on", state.promptTab === "composed");
  if (state.promptTab === "brick") {
    $("prompt").textContent = ctx.catalog.intents[state.base.part].prompt;
    return;
  }
  const snapshot = JSON.stringify([parts(), state.industry]);
  const { prompt } = await post("/api/brief", { parts: parts(), industry: state.industry || null });
  if (snapshot === JSON.stringify([parts(), state.industry])) $("prompt").textContent = prompt;
}

/* ---------- step 2: additional requirements ---------- */
function renderExtras() {
  const list = $("extras");
  list.innerHTML = "";
  state.extras.forEach((extra, index) => {
    const card = document.createElement("div");
    card.className = "extra";
    const family = document.createElement("select");
    ctx.catalog.bricks.forEach((b) => {
      if (!usedFamilies().includes(b.id) || b.id === extra.family) family.add(new Option(b.name, b.id, false, b.id === extra.family));
    });
    family.onchange = () => { state.extras[index] = { family: family.value, part: `basic:${family.value}` }; changed(); };
    const variant = document.createElement("select");
    variantOptions(extra.family).forEach((v) => variant.add(new Option(v.name, v.id, false, v.id === extra.part)));
    variant.onchange = () => { extra.part = variant.value; changed(); };
    const remove = document.createElement("button");
    remove.className = "x";
    remove.textContent = "✕";
    remove.onclick = () => { state.extras.splice(index, 1); changed(); };
    const number = document.createElement("span");
    number.className = "n";
    number.textContent = `#${index + 2}`;
    const task = document.createElement("div");
    task.className = "task";
    task.textContent = ctx.catalog.intents[extra.part].task;
    card.append(number, family, variant, remove, task);
    list.appendChild(card);
  });
  $("add-extra").disabled = state.extras.length >= 3;
}

function renderIndustry() {
  const profile = ctx.catalog.industries.find((i) => i.id === state.industry);
  $("industry-detail").innerHTML = profile
    ? `<b>Entities</b>${esc(profile.entities.join("; "))}<b>Constraints enforced</b>${esc(profile.constraints.join("; "))}
       <b>Data formats</b>${esc(profile.formats.join("; "))}<b>Synthetic fixtures</b>${esc(profile.guidance)}`
    : "The builder gets no industry block; the build is level B or BT.";
}

function renderComposition() {
  const items = parts().map((p) => `<span class="part">${p.replace("basic:", "◇ ")}</span>`).join(" → ");
  const industry = state.industry ? `<span class="ind">+ ${state.industry}</span>` : "";
  $("composition").innerHTML = `${items} ${industry}<span class="lvl">level ${level()}</span>`;
  $("run").disabled = running;
  $("deployment-field").style.display = $("mode").value === "live" ? "" : "none";
  $("run").textContent = $("mode").value === "live" ? `Build live on ${$("deployment").value} & create data point ▸` : "Replay measured build & create data point ▸";
}

function changed() {
  renderBricks(); renderIntent(); renderExtras(); renderIndustry(); renderComposition(); renderPrompt();
  $("runs").innerHTML = "";
}

/* ---------- graph animation ---------- */
const stage = (id) => ctx.result.stages.find((s) => s.id === id);

function showStage(id) {
  if (!ctx.result) return;
  graph.select(id);
  renderInspector($("inspector"), stage(id), ctx);
}

function finish(id) {
  const s = stage(id);
  graph.setState(id, s.status, s.summary);
}

async function visit(from, to, step, kind) {
  if (from) await graph.pulse(from, to, step, kind);
  graph.setState(to, "active", "working…");
  showStage(to);
  await sleep(step / 2);
}

function meters(calls, tokens, forecast) {
  $("meter-calls").textContent = calls;
  $("meter-tokens").textContent = fmt(tokens);
  $("meter-forecast").textContent = forecast ? fmt(forecast) : "–";
}

async function intro(step) {
  meters(0, 0);
  await visit(null, "designer", step); finish("designer");
  await visit("designer", "planner", step); finish("planner");
  await visit("planner", "briefing", step); finish("briefing");
}

async function outro(step) {
  for (const [from, to] of [["builder", "verifier"], ["verifier", "telemetry"]]) { await visit(from, to, step); finish(to); }
  await Promise.all([graph.pulse("telemetry", "datapoint", step), graph.pulse("planner", "datapoint", step, "data")]);
  await learn(step);
}

async function learn(step) {
  graph.setState("datapoint", "active"); showStage("datapoint"); await sleep(step * 1.5); finish("datapoint");
  const chain = [["datapoint", "features"], ["features", "splitter"], ["splitter", "trainer"], ["trainer", "gate"], ["gate", "model"]];
  for (const [from, to] of chain) { await visit(from, to, step, "data"); await sleep(step); finish(to); }
  $("meter-forecast").textContent = fmt(stage("model").output.forecast.total_tokens);
  await graph.pulse("model", "designer", step * 1.5, "data");
  showStage("model");
}

async function replay(step) {
  const builder = stage("builder");
  if (builder.status !== "done") {
    ["builder", "workspace", "verifier", "telemetry"].forEach(finish);
    await graph.pulse("planner", "datapoint", step, "data");
    return learn(step);
  }
  await visit("briefing", "builder", step);
  const events = builder.events, per = Math.max(50, Math.min(320, 7000 / events.length)) / Number($("speed").value);
  let tokens = 0;
  for (let i = 0; i < events.length; i += 1) {
    await graph.pulse("builder", "workspace", per);
    graph.setState("workspace", "active", `request ${i + 1}/${events.length}`);
    await graph.pulse("workspace", "builder", per);
    tokens += events[i].input_tokens + events[i].output_tokens;
    meters(i + 1, tokens);
    graph.setState("builder", "active", `${fmt(tokens)} tokens`);
    renderInspector($("inspector"), { ...builder, events: events.slice(0, i + 1) }, ctx);
  }
  finish("builder"); finish("workspace");
  await outro(step);
}

/* ---------- live build on Azure Foundry ---------- */
async function live(step, deployment) {
  const job = await post("/api/live", { parts: parts(), industry: state.industry || null, deployment });
  await visit("briefing", "builder", step);
  const base = stage("builder"), log = [], usage = [];
  const liveStage = () => ({ ...base, status: "active", summary: `${deployment} · ${usage.length} requests`,
    rule: `Live: ${ctx.catalog.foundry.openai_endpoint} · deployment ${deployment}. Each request's usage is reported by the API.`,
    events: usage, tool_log: log, input: { deployment, endpoint: ctx.catalog.foundry.openai_endpoint, point_id: job.point_id } });
  let since = 0;
  for (;;) {
    const poll = await fetch(`/api/live/${job.job_id}?since=${since}`).then((r) => r.json());
    since = poll.next;
    for (const event of poll.events) {
      log.push(event);
      if (event.type === "request") {
        usage.push(event.usage);
        const tokens = usage.reduce((a, u) => a + u.input_tokens + u.output_tokens, 0);
        meters(usage.length, tokens);
        graph.setState("builder", "active", `${fmt(tokens)} tokens`);
        await graph.pulse("builder", "workspace", 260);
      } else if (event.type === "tool") {
        graph.setState("workspace", "active", `${event.name}`);
        await graph.pulse("workspace", "builder", 260);
      }
      renderInspector($("inspector"), liveStage(), ctx);
    }
    if (poll.status === "failed") {
      graph.setState("builder", "failed", "error");
      throw new Error(poll.error);
    }
    if (poll.status === "done") {
      ctx.result = poll.result;
      stage("builder").tool_log = log;
      ["designer", "planner", "briefing", "builder", "workspace"].forEach(finish);
      return outro(step);
    }
    await sleep(900);
  }
}

function renderRuns(result) {
  const box = $("runs");
  if (!result.measured_runs.length) {
    box.innerHTML = "<b>No measured corpus build</b> of this membership yet.";
    return;
  }
  box.innerHTML = `<b>${result.measured_runs.length} measured corpus build(s)</b> of this membership — replay one:`;
  result.measured_runs.forEach((run) => {
    const button = document.createElement("button");
    button.className = !result.live && run.id === result.replayed ? "on" : "";
    button.textContent = `${run.id.slice(0, 40)} · ${fmt(run.total_tokens)}${run.exact_order ? "" : " · other order"}`;
    button.onclick = () => { $("mode").value = "replay"; renderComposition(); execute(run.id); };
    box.appendChild(button);
  });
}

async function execute(buildId = null) {
  if (running) return;
  running = true;
  $("error").textContent = "";
  renderComposition();
  const step = 650 / Number($("speed").value);
  try {
    const result = await post("/api/run", { parts: parts(), industry: state.industry || null, build_id: buildId });
    ctx.result = result;
    graph.render(result.stages, result.edges);
    renderRuns(result);
    document.querySelector(".stage").scrollIntoView({ behavior: "smooth" });
    await intro(step);
    if ($("mode").value === "live") await live(step, $("deployment").value);
    else await replay(step);
  } catch (error) {
    $("error").textContent = error.message;
  } finally {
    running = false;
    renderComposition();
  }
}

async function init() {
  const [catalog, corpus, training] = await Promise.all(["/api/catalog", "/api/corpus", "/api/training"]
    .map((url) => fetch(url).then((r) => r.json())));
  Object.assign(ctx, { catalog, corpus: corpus.builds, training, provenance: catalog.provenance });
  $("provenance").textContent = catalog.provenance;
  $("foundry").textContent = `live LLM: ${catalog.foundry.openai_endpoint} · ${catalog.foundry.auth}`;
  $("foundry").title = catalog.foundry.project_endpoint;
  $("measured-count").textContent = catalog.measured_builds;
  if (catalog.studio) {
    $("studio-link").hidden = false;
    $("studio-link").href = catalog.studio.url;
    $("studio-link").innerHTML = `Token Yield Studio ↗<small>${catalog.studio.foundry ? `live Foundry · cap $${catalog.studio.budget_usd}` : "offline"}</small>`;
  }
  catalog.industries.forEach((i) => $("industry").add(new Option(i.name, i.id)));
  catalog.foundry.deployments.forEach((d) => $("deployment").add(new Option(d, d)));
  $("industry").onchange = (e) => { state.industry = e.target.value; changed(); };
  $("mode").onchange = renderComposition;
  $("deployment").onchange = renderComposition;
  $("tab-brick").onclick = () => { state.promptTab = "brick"; renderPrompt(); };
  $("tab-composed").onclick = () => { state.promptTab = "composed"; renderPrompt(); };
  $("add-extra").onclick = () => {
    const free = catalog.bricks.find((b) => !usedFamilies().includes(b.id));
    if (free) state.extras.push({ family: free.id, part: `basic:${free.id}` });
    changed();
  };
  $("run").onclick = () => execute();
  state.base = { family: catalog.bricks.find((b) => b.id === "research").id, part: "basic:research" };
  changed();
}

init();
