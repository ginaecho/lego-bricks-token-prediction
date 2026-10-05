// Small dependency-free SVG charts. Each function returns an SVG string.

const fmt = (n) => Math.round(n).toLocaleString();
const kfmt = (n) => (n >= 1000 ? `${Math.round(n / 1000)}k` : `${Math.round(n)}`);

function frame(width, height, body) {
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" style="max-width:${Math.round(width * 1.25)}px" xmlns="http://www.w3.org/2000/svg">${body}</svg>`;
}

function ticks(max, count = 4) {
  const step = Math.pow(10, Math.floor(Math.log10(max / count)));
  const nice = [1, 2, 2.5, 5, 10].map((m) => m * step).find((s) => max / s <= count) || step * 10;
  return Array.from({ length: Math.floor(max / nice) + 1 }, (_, i) => i * nice);
}

const SEGMENTS = [
  ["cache_read_tokens", "#c9c2f2", "cache read"],
  ["cache_write_tokens", "#e5b06d", "cache write"],
  ["fresh_input", "#6b5bd2", "fresh input"],
  ["output_tokens", "#4ea66c", "output"],
];

/** Stacked tokens per LLM request plus the cumulative total (right axis). */
export function tokenTimeline(events, shown = events.length) {
  const W = 760, H = 250, L = 52, R = 58, T = 16, B = 34;
  const totals = events.map((e) => e.input_tokens + e.output_tokens);
  const cumulative = totals.reduce((acc, t) => [...acc, (acc.at(-1) || 0) + t], []);
  const maxBar = Math.max(...totals, 1), maxCum = Math.max(cumulative.at(-1) || 1, 1);
  const bw = (W - L - R) / Math.max(events.length, 1);
  const y = (v) => H - B - (v / maxBar) * (H - T - B);
  const yc = (v) => H - B - (v / maxCum) * (H - T - B);
  let body = ticks(maxBar).map((t) => `<line class="axis" x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}"/>
    <text x="${L - 6}" y="${y(t) + 3}" text-anchor="end">${kfmt(t)}</text>`).join("");
  body += ticks(maxCum).map((t) => `<text x="${W - R + 6}" y="${yc(t) + 3}" fill="#6b5bd2">${kfmt(t)}</text>`).join("");
  events.slice(0, shown).forEach((e, i) => {
    let base = 0;
    const x = L + i * bw + bw * 0.15;
    SEGMENTS.forEach(([key, color]) => {
      const v = Math.max(e[key], 0);
      body += `<rect x="${x}" y="${y(base + v)}" width="${bw * 0.7}" height="${y(base) - y(base + v)}" fill="${color}">
        <title>request ${i + 1} · ${key}: ${fmt(v)}</title></rect>`;
      base += v;
    });
    if (events.length <= 40 || i % Math.ceil(events.length / 20) === 0) {
      body += `<text x="${x + bw * 0.35}" y="${H - B + 14}" text-anchor="middle">${i + 1}</text>`;
    }
  });
  const pts = cumulative.slice(0, shown).map((c, i) => `${L + i * bw + bw / 2},${yc(c)}`).join(" ");
  body += `<polyline points="${pts}" fill="none" stroke="#6b5bd2" stroke-width="2"/>`;
  body += `<text x="${(W - R + L) / 2}" y="${H - 4}" text-anchor="middle">LLM request #</text>`;
  body += SEGMENTS.map(([, color, name], i) => `<rect x="${L + i * 95}" y="2" width="10" height="10" fill="${color}"/>
    <text x="${L + 14 + i * 95}" y="11">${name}</text>`).join("") +
    `<text x="${W - R - 4}" y="11" text-anchor="end" fill="#6b5bd2">— cumulative</text>`;
  return frame(W, H, body);
}

/** All measured builds: tokens vs number of bricks, coloured by split. */
export function corpusScatter(builds, highlight, extra = []) {
  const W = 760, H = 270, L = 56, R = 16, T = 14, B = 34;
  const colors = { train: "#0b7e89", validation: "#e58b2a", test: "#c4553f" };
  builds = [...builds, ...extra.map((e) => ({ ...e, split: "live" }))];
  highlight = extra.length ? extra[0].id : highlight;
  const maxY = Math.max(...builds.map((b) => b.total_tokens));
  const x = (c, j) => L + ((c - 0.5) / 4) * (W - L - R) + j;
  const y = (v) => H - B - (v / maxY) * (H - T - B);
  let body = ticks(maxY).map((t) => `<line class="axis" x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}"/>
    <text x="${L - 6}" y="${y(t) + 3}" text-anchor="end">${kfmt(t)}</text>`).join("");
  body += [1, 2, 3, 4].map((c) => `<text x="${x(c, 0)}" y="${H - B + 16}" text-anchor="middle">${c} brick${c > 1 ? "s" : ""}</text>`).join("");
  builds.forEach((b) => {
    const hash = [...b.id].reduce((h, ch) => (h * 31 + ch.charCodeAt(0)) % 9973, 7);
    const jitter = (hash / 9973 - 0.5) * 120;
    const on = b.id === highlight;
    body += `<circle cx="${x(b.count, jitter)}" cy="${y(b.total_tokens)}" r="${on ? 8 : 3.2}"
      fill="${on ? "#6b5bd2" : colors[b.split]}" fill-opacity="${on ? 1 : 0.55}" stroke="${on ? "#172126" : "none"}" stroke-width="2">
      <title>${b.id} · ${fmt(b.total_tokens)} tokens · ${b.split}</title></circle>`;
  });
  body += Object.entries(colors).map(([k, c], i) => `<circle cx="${L + 8 + i * 90}" cy="8" r="4" fill="${c}"/>
    <text x="${L + 16 + i * 90}" y="11">${k}</text>`).join("") +
    `<circle cx="${L + 290}" cy="8" r="5" fill="#6b5bd2" stroke="#172126"/><text x="${L + 300}" y="11">this data point</text>`;
  return frame(W, H, body);
}

/** Grouped-CV mean absolute error for each ridge penalty. */
export function alphaCurve(search, selected) {
  const W = 520, H = 220, L = 60, R = 16, T = 14, B = 34;
  const xs = search.map((s) => Math.log10(s.alpha));
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...search.map((s) => s.mae)) * 0.995, maxY = Math.max(...search.map((s) => s.mae)) * 1.005;
  const x = (v) => L + ((v - minX) / (maxX - minX)) * (W - L - R);
  const y = (v) => H - B - ((v - minY) / (maxY - minY)) * (H - T - B);
  let body = `<polyline fill="none" stroke="#0b7e89" stroke-width="2" points="${search.map((s) => `${x(Math.log10(s.alpha))},${y(s.mae)}`).join(" ")}"/>`;
  search.forEach((s) => {
    const on = s.alpha === selected;
    body += `<circle cx="${x(Math.log10(s.alpha))}" cy="${y(s.mae)}" r="${on ? 7 : 4}" fill="${on ? "#6b5bd2" : "#0b7e89"}">
      <title>alpha ${s.alpha} · MAE ${fmt(s.mae)} · MAPE ${(s.mape * 100).toFixed(1)}%</title></circle>
      <text x="${x(Math.log10(s.alpha))}" y="${H - B + 16}" text-anchor="middle">${s.alpha}</text>`;
  });
  body += `<text x="${L - 6}" y="${y(maxY) + 4}" text-anchor="end">${kfmt(maxY)}</text><text x="${L - 6}" y="${y(minY)}" text-anchor="end">${kfmt(minY)}</text>`;
  body += `<text x="${(W + L) / 2}" y="${H - 2}" text-anchor="middle">ridge alpha (grouped 5-fold CV MAE, tokens)</text>`;
  return frame(W, H, body);
}

/** Predicted vs actual with the y = x reference line. */
export function predictedVsActual(points, highlight) {
  const W = 520, H = 260, L = 56, R = 16, T = 14, B = 34;
  const all = points.flatMap((p) => [p.actual, p.predicted]);
  const lo = Math.min(...all) * 0.9, hi = Math.max(...all) * 1.05;
  const s = (v, a, b) => a + ((v - lo) / (hi - lo)) * (b - a);
  const x = (v) => s(v, L, W - R), y = (v) => s(v, H - B, T);
  let body = `<line x1="${x(lo)}" y1="${y(lo)}" x2="${x(hi)}" y2="${y(hi)}" stroke="#c8c6bd" stroke-dasharray="4 4"/>`;
  points.forEach((p) => {
    const on = p.id === highlight;
    body += `<circle cx="${x(p.actual)}" cy="${y(p.predicted)}" r="${on ? 8 : 4}" fill="${on ? "#6b5bd2" : "#e58b2a"}" fill-opacity=".75">
      <title>${p.id}\nactual ${fmt(p.actual)} · predicted ${fmt(p.predicted)}</title></circle>`;
  });
  body += `<text x="${(W + L) / 2}" y="${H - 4}" text-anchor="middle">actual tokens →</text>
    <text x="12" y="${(H - B) / 2}" transform="rotate(-90 12 ${(H - B) / 2})" text-anchor="middle">predicted →</text>
    <text x="${x(lo)}" y="${H - B + 14}">${kfmt(lo)}</text><text x="${x(hi)}" y="${H - B + 14}" text-anchor="end">${kfmt(hi)}</text>`;
  return frame(W, H, body);
}

/** Diverging bars of each feature's log-space effect on the forecast. */
export function effects(items) {
  const W = 520, row = 22, L = 230, H = items.length * row + 26;
  const max = Math.max(...items.map((i) => Math.abs(i.log_effect)), 1e-6);
  const mid = L + (W - L - 10) / 2, half = (W - L - 10) / 2;
  let body = `<line x1="${mid}" x2="${mid}" y1="0" y2="${H - 18}" class="axis"/>`;
  items.forEach((it, i) => {
    const w = (Math.abs(it.log_effect) / max) * half;
    const up = it.log_effect > 0;
    body += `<text x="${L - 8}" y="${i * row + 15}" text-anchor="end">${it.name} = ${it.value}</text>
      <rect x="${up ? mid : mid - w}" y="${i * row + 5}" width="${w}" height="13" fill="${up ? "#c4553f" : "#0b7e89"}">
      <title>×${it.multiplier.toFixed(3)} on input tokens</title></rect>`;
  });
  body += `<text x="${mid - 6}" y="${H - 4}" text-anchor="end">fewer tokens</text><text x="${mid + 6}" y="${H - 4}">more tokens</text>`;
  return frame(W, H, body);
}

/** Engineered feature row: every one of the 46 inputs, non-zero highlighted. */
export function featureRow(vector) {
  const cols = 2, row = 17, W = 760, H = Math.ceil(vector.length / cols) * row + 8;
  const max = Math.max(...vector.map((v) => v.value), 1);
  let body = "";
  vector.forEach((v, i) => {
    const c = Math.floor(i / Math.ceil(vector.length / cols)), r = i % Math.ceil(vector.length / cols);
    const x0 = c * (W / cols), yy = r * row + 4, bw = 120 * (v.value / max);
    const on = v.value !== 0;
    body += `<text x="${x0 + 222}" y="${yy + 11}" text-anchor="end" style="fill:${on ? "#172126" : "#b7bcbd"}">${v.name}</text>
      <rect x="${x0 + 228}" y="${yy + 2}" width="${Math.max(bw, on ? 2 : 0)}" height="11" fill="#0b7e89"/>
      <text x="${x0 + 232 + bw}" y="${yy + 11}" style="fill:${on ? "#172126" : "#b7bcbd"}">${v.value}</text>`;
  });
  return frame(W, H, body);
}

/** Forecast before vs after this data point, against the measured label. */
export function forecastBars(rows, interval) {
  const W = 520, row = 34, L = 190, H = rows.length * row + 30;
  const max = Math.max(...rows.map((r) => r.value), interval ? interval[1] : 0) * 1.08;
  const x = (v) => L + (v / max) * (W - L - 70);
  let body = "";
  if (interval) {
    body += `<rect x="${x(interval[0])}" y="2" width="${x(interval[1]) - x(interval[0])}" height="${rows.length * row}" fill="#25c5cf" fill-opacity=".12"/>
      <text x="${x(interval[1])}" y="${rows.length * row + 14}" text-anchor="end">80% interval</text>`;
  }
  rows.forEach((r, i) => {
    body += `<text x="${L - 8}" y="${i * row + 21}" text-anchor="end" style="fill:#172126">${r.label}</text>
      <rect x="${L}" y="${i * row + 8}" width="${x(r.value) - L}" height="18" fill="${r.color}"/>
      <text x="${x(r.value) + 6}" y="${i * row + 21}" style="fill:#172126">${fmt(r.value)}</text>`;
  });
  return frame(W, H, body);
}
