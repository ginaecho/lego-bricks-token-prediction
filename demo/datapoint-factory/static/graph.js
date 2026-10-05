// SVG agent-interaction graph: layout, edge routing and travelling pulses.

const NS = "http://www.w3.org/2000/svg";
const W = 158, H = 58;
const ICON = { human: "👤", llm: "✦", tool: "🛠", code: "⚙", artifact: "▣" };
const COLOR = { human: "#e58b2a", llm: "#6b5bd2", tool: "#b0782d", code: "#0b7e89", artifact: "#4ea66c" };

const POS = {
  designer: [100, 140], planner: [310, 140], briefing: [520, 140], builder: [740, 140],
  workspace: [740, 238], verifier: [960, 140], telemetry: [1170, 140], datapoint: [1385, 140],
  features: [1385, 415], splitter: [1160, 415], trainer: [935, 415], gate: [710, 415], model: [485, 415],
};

function el(name, attrs = {}, parent) {
  const node = document.createElementNS(NS, name);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
  if (parent) parent.appendChild(node);
  return node;
}

function route(from, to) {
  const [ax, ay] = POS[from], [bx, by] = POS[to];
  if (from === "planner" && to === "datapoint") {
    return { d: `M${ax},${ay - H / 2} C${ax + 80},${ay - 95} ${bx - 80},${by - 95} ${bx},${by - H / 2}`, lx: (ax + bx) / 2, ly: ay - 76 };
  }
  if (from === "model" && to === "designer") {
    return { d: `M${ax - W / 2},${ay} C${ax - 300},${ay} ${bx},${by + 170} ${bx},${by + H / 2}`, lx: 250, ly: 430 };
  }
  if (ax === bx) {
    const dx = from === "builder" ? -20 : from === "workspace" ? 20 : 0;
    const [y1, y2] = ay < by ? [ay + H / 2, by - H / 2] : [ay - H / 2, by + H / 2];
    return { d: `M${ax + dx},${y1} L${bx + dx},${y2}`, lx: ax + dx + (dx < 0 ? -6 : 6), ly: (y1 + y2) / 2 + 4, anchor: dx < 0 ? "end" : "start" };
  }
  const sx = ax < bx ? ax + W / 2 : ax - W / 2, ex = ax < bx ? bx - W / 2 : bx + W / 2;
  return { d: `M${sx},${ay} L${ex},${by}`, lx: (sx + ex) / 2, ly: ay - H / 2 - 8 };
}

export function createGraph(svg, onSelect) {
  const nodes = {}, edges = {};

  function render(stages, edgeList) {
    svg.innerHTML = "";
    const defs = el("defs", {}, svg);
    const marker = el("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", fill: "#7d878a" }, marker);
    el("rect", { class: "lane", x: 8, y: 14, width: 1484, height: 268, rx: 10 }, svg);
    el("text", { class: "lane-title", x: 22, y: 34 }, svg).textContent = "① CREATE ONE DATA POINT — ISOLATED AGENT BUILD";
    el("rect", { class: "lane", x: 8, y: 300, width: 1484, height: 192, rx: 10 }, svg);
    el("text", { class: "lane-title", x: 22, y: 320 }, svg).textContent = "② LEARN FROM ALL DATA POINTS — FEATURES → TRAINING";
    const edgeLayer = el("g", {}, svg), nodeLayer = el("g", {}, svg);
    edgeList.forEach(({ from, to, label }) => {
      const r = route(from, to);
      const path = el("path", { class: "edge", d: r.d, "marker-end": "url(#arrow)" }, edgeLayer);
      const text = el("text", { class: "edge-label", x: r.lx, y: r.ly, "text-anchor": r.anchor || "middle" }, edgeLayer);
      text.textContent = label;
      edges[`${from}>${to}`] = path;
    });
    stages.forEach((stage) => {
      const [x, y] = POS[stage.id];
      const g = el("g", { class: "node pending", transform: `translate(${x - W / 2},${y - H / 2})` }, nodeLayer);
      el("rect", { width: W, height: H, rx: 8 }, g);
      el("rect", { class: "bar", width: 5, height: H, rx: 2, fill: COLOR[stage.actor] }, g);
      el("text", { class: "icon", x: 14, y: 24 }, g).textContent = ICON[stage.actor];
      el("text", { class: "title", x: 36, y: 24 }, g).textContent = stage.label;
      const sub = el("text", { class: "sub", x: 14, y: 45 }, g);
      sub.textContent = "waiting";
      g.addEventListener("click", () => onSelect(stage.id));
      nodes[stage.id] = { g, sub, stage };
    });
  }

  function setState(id, state, subtitle) {
    const n = nodes[id];
    if (!n) return;
    n.g.classList.remove("pending", "active", "done", "not_measured", "failed");
    n.g.classList.add(state);
    if (subtitle !== undefined) n.sub.textContent = subtitle.length > 24 ? `${subtitle.slice(0, 23)}…` : subtitle;
  }

  function select(id) {
    Object.values(nodes).forEach((n) => n.g.classList.toggle("selected", n.stage.id === id));
  }

  function pulse(from, to, ms, kind = "") {
    const path = edges[`${from}>${to}`];
    if (!path) return Promise.resolve();
    path.classList.add("lit");
    const dot = el("circle", { r: 6, class: `pulse ${kind}` }, svg);
    const length = path.getTotalLength();
    return new Promise((resolve) => {
      const start = performance.now();
      const step = (now) => {
        const t = Math.min((now - start) / ms, 1);
        const p = path.getPointAtLength(t * length);
        dot.setAttribute("cx", p.x);
        dot.setAttribute("cy", p.y);
        if (t < 1) requestAnimationFrame(step);
        else { dot.remove(); resolve(); }
      };
      requestAnimationFrame(step);
    });
  }

  return { render, setState, select, pulse };
}
