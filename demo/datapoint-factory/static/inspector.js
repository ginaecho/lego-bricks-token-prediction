// Node inspector: output (charts + data) beside the shaded source code that runs the stage.

import * as charts from "./charts.js";

const ACTOR = { human: ["human", "#e58b2a"], llm: ["LLM agent", "#6b5bd2"], tool: ["tools", "#b0782d"],
  code: ["deterministic code", "#0b7e89"], artifact: ["artifact", "#4ea66c"] };
const fmt = (n) => Math.round(n).toLocaleString();
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const json = (v) => `<pre>${esc(JSON.stringify(v, null, 2))}</pre>`;

function kpis(items) {
  return `<div class="kpis">${items.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("")}</div>`;
}

function toolLog(log) {
  if (!log?.length) return "";
  return `<ul class="toollog">${log.map((e) => e.type === "request"
    ? `<li class="req">LLM request ${e.index}: ${fmt(e.usage.input_tokens)} in (${fmt(e.usage.cache_read_tokens)} cached) · ${fmt(e.usage.output_tokens)} out → ${esc(e.tools.join(", ") || "no tool call")}</li>`
    : e.type === "tool" ? `<li class="tool">↳ ${esc(e.name)}: ${esc(e.summary)}</li>`
      : `<li>${esc(e.text || e.reason || "")}</li>`).join("")}</ul>`;
}

const VISUALS = {
  builder(stage) {
    if (!stage.events.length) return toolLog(stage.tool_log);
    const t = stage.events.reduce((a, e) => ({ in: a.in + e.input_tokens, out: a.out + e.output_tokens,
      cr: a.cr + e.cache_read_tokens, ms: a.ms + e.duration_ms }), { in: 0, out: 0, cr: 0, ms: 0 });
    return kpis([["requests", stage.events.length], ["input", fmt(t.in)], ["output", fmt(t.out)],
      ["cache-read share", `${((t.cr / Math.max(t.in, 1)) * 100).toFixed(0)}%`], ["model time", `${(t.ms / 1000).toFixed(0)} s`]]) +
      charts.tokenTimeline(stage.events) +
      `<p class="chart-note">Each bar is one LLM request (write → run tests → repair). Every request re-sends the growing conversation.</p>` +
      toolLog(stage.tool_log);
  },
  datapoint(stage, ctx) {
    const live = ctx.result.live && stage.output?.label_total_tokens
      ? [{ id: stage.output.id, count: ctx.result.selection.parts.length, total_tokens: stage.output.label_total_tokens }] : [];
    return charts.corpusScatter(ctx.corpus, ctx.result.replayed, live) +
      `<p class="chart-note">All ${ctx.corpus.length} measured corpus builds${live.length ? "; the live point is shown in violet" : ""}.
       More bricks usually cost more, but integrated builds are not the sum of their parts.</p>`;
  },
  features(stage) {
    return charts.featureRow(stage.output.vector) +
      `<p class="chart-note">The exact numeric row the trainer receives. Staffing, months and anything observed after the build are excluded.</p>`;
  },
  trainer(stage) {
    return charts.alphaCurve(stage.output.alpha_search, stage.output.selected_alpha) +
      `<p class="chart-note">Grouped 5-fold CV over ${stage.input.development_records} train+validation records; membership groups never cross folds.</p>`;
  },
  gate(stage, ctx) {
    const v = ctx.training.validation, test = ctx.training.stored_test;
    return kpis([["validation MAE", fmt(v.candidate.mae)], ["MAPE", `${(v.candidate.mape * 100).toFixed(1)}%`],
      ["mean-baseline MAE", fmt(v.training_mean_baseline.mae)], ["test MAPE (once)", test ? `${(test.mape * 100).toFixed(1)}%` : "–"]]) +
      charts.predictedVsActual(v.points, ctx.result.replayed) +
      `<p class="chart-note">Validation records (${v.points.length}). The test result is read from the stored report and never re-run.</p>`;
  },
  model(stage) {
    const o = stage.output, f = o.forecast;
    const rows = [{ label: "forecast (current model)", value: f.total_tokens, color: "#0b7e89" }];
    if (o.forecast_before_this_datapoint) rows.unshift({ label: "forecast before this point", value: o.forecast_before_this_datapoint.total_tokens, color: "#9aa3a6" });
    if (o.actual_total_tokens) rows.push({ label: o.live_builder ? `live: ${o.live_builder}` : "measured label", value: o.actual_total_tokens, color: "#6b5bd2" });
    const note = o.live_builder
      ? `Live build by ${o.live_builder}. The model was trained on gpt-6-astra Copilot builds, so this is a cross-builder comparison, not a pooled label.`
      : o.actual_total_tokens
        ? (o.forecast_before_this_datapoint ? "“Before” refits the model without this membership: the quote before measuring it."
          : "This membership is in the held-out test split; the model never trained on it.")
        : "Not yet measured: forecast only.";
    return charts.forecastBars(rows, f.interval) + `<p class="chart-note">${note}</p>` +
      `<div class="col-title">Strongest feature effects (input tokens)</div>${charts.effects(o.top_effects_input_tokens)}`;
  },
};

function codeBlocks(refs) {
  if (!refs.length) return `<div class="no-code">No repository code runs here: this node is a human choice or the recorded LLM session itself.</div>`;
  return refs.map((ref) => {
    const lines = ref.source.replace(/\n$/, "").split("\n").map((text, i) => {
      const def = /^\s*(async\s+)?(def|class)\s/.test(text) && i < 3;
      return `<span class="ln${def ? " def" : ""}"><span class="no">${ref.line + i}</span>${esc(text)}</span>`;
    }).join("");
    return `<div class="code-block"><div class="code-head"><span class="run">RUNNING</span><b>${esc(ref.function)}</b>
      <span class="loc">${esc(ref.file)}:${ref.line}–${ref.end_line}</span></div><pre class="src">${lines}</pre></div>`;
  }).join("");
}

export function renderInspector(container, stage, ctx) {
  const [actor, color] = ACTOR[stage.actor];
  const visual = VISUALS[stage.id] ? VISUALS[stage.id](stage, ctx) : "";
  const { events, ...output } = stage.output || {};
  const outputView = stage.output?.instructions ? `<pre>${esc(stage.output.instructions)}</pre>` : json(output);
  container.innerHTML = `
    <div class="insp-head"><h3>${esc(stage.label)}</h3>
      <span class="badge" style="background:${color}">${actor}</span>
      <span class="badge" style="background:${stage.status === "done" ? "#172126" : stage.status === "failed" ? "#c4553f" : "#9aa3a6"}">${stage.status.replace("_", " ")}</span>
      <span>${esc(stage.summary)}</span></div>
    <div class="rule">${esc(stage.rule)}</div>
    <div class="insp-grid">
      <div><div class="col-title">Output</div>${visual}${outputView}
        <details><summary>Input</summary>${json(stage.input)}</details></div>
      <div><div class="col-title">Code running this stage · token_yield/ and examples/ from ${esc(ctx.provenance)}; demo/ is this app</div>${codeBlocks(stage.code)}</div>
    </div>`;
}
