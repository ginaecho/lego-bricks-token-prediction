"""Render harness cards to a self-contained, theme-aware HTML dashboard.

The dashboard *is* the transparency the project is after: every module's
competence, conformance, cost, momentum, and upstream news on one page, in
numbers instead of anecdotes. No external assets — one file you can open or
publish anywhere.
"""

from __future__ import annotations

import html
from typing import Iterable

from .card import HarnessCard


def _bar(score: int | None) -> str:
    if score is None:
        return '<span class="na">no data</span>'
    hue = round(score * 1.2)  # 0=red → 120=green
    return (f'<span class="scorebar"><span class="fill" '
            f'style="width:{score}%;background:hsl({hue} 70% 45%)"></span>'
            f'<b>{score}</b></span>')


def _card_html(c: HarnessCard) -> str:
    conf = c.conformance
    pr = conf.pass_rate
    pr_txt = f"{pr*100:.0f}%" if pr is not None else "—"

    comp_rows = "".join(
        f'<tr><td>{html.escape(t)}</td><td>{_bar(comp.score)}</td>'
        f'<td class="dim">{comp.passes}/{comp.judged}</td></tr>'
        for t, comp in sorted(c.competence.items(),
                              key=lambda kv: (kv[1].score is None, -(kv[1].score or 0)))
    ) or '<tr><td colspan="3" class="dim">not yet exercised</td></tr>'

    tier_class = {"deterministic": "free", "static": "cheap", "llm_judge": "priced"}.get(
        c.cost.tier, "cheap")
    tier_label = {"deterministic": "deterministic · free",
                  "static": "static rule · cheap",
                  "llm_judge": "LLM judge · priced"}.get(c.cost.tier, c.cost.tier)

    up = c.upstream
    up_bits = [f'<code>{html.escape(up.repo)}</code> @ v{html.escape(up.version)}']
    if up.update_available:
        up_bits.append(f'<span class="badge new">v{html.escape(up.latest_version)} available</span>')
    if up.impact:
        up_bits.append(f'<div class="impact">⚠ {html.escape(up.impact)}</div>')
    if up.conflicts:
        up_bits.append('<div class="impact">conflicts: '
                       + ", ".join(html.escape(x) for x in up.conflicts) + '</div>')

    crit = (f'<span class="badge crit">{conf.critical_fails} critical</span>'
            if conf.critical_fails else "")
    tags = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in c.tags)

    return f"""
    <article class="card">
      <header>
        <h2>{html.escape(c.name)} <span class="mid">{html.escape(c.module_id)}</span></h2>
        <p class="summary">{html.escape(c.summary)}</p>
        <div class="tags">{tags}</div>
      </header>

      <section>
        <h3>Competence <span class="hint">conformance rate by task type</span></h3>
        <table>{comp_rows}</table>
      </section>

      <div class="grid">
        <section>
          <h3>Conformance</h3>
          <div class="big {'ok' if (pr or 0) >= .8 else 'warn'}">{pr_txt}</div>
          <div class="dim">{conf.passes} pass · {conf.fails} fail · {conf.errors} err {crit}</div>
        </section>
        <section>
          <h3>Cost to enforce</h3>
          <div class="pill {tier_class}">{tier_label}</div>
          <div class="dim">acc {c.cost.accuracy:.0%} · {c.cost.tokens_per_check} tok/check
            · {c.cost.total_tokens} total</div>
        </section>
        <section>
          <h3>Momentum <span class="hint">{c.momentum.trend}</span></h3>
          <div class="spark">{c.momentum.spark() or '—'}</div>
          <div class="dim">{sum(c.momentum.per_session.values())} bindings
            across {len(c.momentum.per_session)} session(s)</div>
        </section>
      </div>

      <section class="upstream">
        <h3>Upstream</h3>
        {''.join(up_bits)}
      </section>
    </article>"""


def render_dashboard(cards: Iterable[HarnessCard], title: str = "OpenHarness — Harness Cards") -> str:
    cards = list(cards)
    body = "\n".join(_card_html(c) for c in cards)
    n = len(cards)
    total_tokens = sum(c.cost.total_tokens for c in cards)
    total_checks = sum(c.cost.checks for c in cards)
    total_fail = sum(c.conformance.fails for c in cards)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg:#f6f7f9; --fg:#14171c; --muted:#6b7280; --card:#fff; --line:#e5e7eb;
    --accent:#3b5bdb; --ok:#2b8a3e; --warn:#e8590c; --crit:#c92a2a;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0e1116; --fg:#e6e8eb; --muted:#9aa4b2; --card:#161b22;
             --line:#232a34; --accent:#748ffc; }}
  }}
  :root[data-theme="dark"] {{ --bg:#0e1116; --fg:#e6e8eb; --muted:#9aa4b2;
      --card:#161b22; --line:#232a34; --accent:#748ffc; }}
  :root[data-theme="light"] {{ --bg:#f6f7f9; --fg:#14171c; --muted:#6b7280;
      --card:#fff; --line:#e5e7eb; --accent:#3b5bdb; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  header.top {{ padding:32px 24px 8px; max-width:1200px; margin:0 auto; }}
  header.top h1 {{ margin:0 0 4px; font-size:26px; letter-spacing:-.02em; }}
  header.top p {{ margin:0; color:var(--muted); }}
  .stats {{ display:flex; gap:24px; flex-wrap:wrap; max-width:1200px;
    margin:16px auto 0; padding:0 24px; }}
  .stat b {{ font-size:22px; }} .stat span {{ color:var(--muted); font-size:13px; }}
  main {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
    gap:16px; max-width:1200px; margin:24px auto; padding:0 24px 48px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
    padding:18px 18px 8px; }}
  .card h2 {{ font-size:17px; margin:0 0 2px; }}
  .card h2 .mid {{ font:12px monospace; color:var(--muted); font-weight:400; }}
  .summary {{ margin:.2em 0 .6em; color:var(--muted); font-size:13px; }}
  h3 {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em;
    color:var(--muted); margin:14px 0 6px; }}
  h3 .hint {{ text-transform:none; letter-spacing:0; color:var(--muted);
    font-weight:400; }}
  table {{ width:100%; border-collapse:collapse; }}
  td {{ padding:3px 0; font-size:13px; }}
  td:first-child {{ width:34%; }}
  .scorebar {{ position:relative; display:inline-block; width:100%; height:16px;
    background:var(--line); border-radius:4px; }}
  .scorebar .fill {{ position:absolute; left:0; top:0; bottom:0; border-radius:4px; }}
  .scorebar b {{ position:absolute; right:6px; top:-1px; font-size:11px;
    color:var(--fg); mix-blend-mode:difference; }}
  .na {{ color:var(--muted); font-size:12px; }}
  .dim {{ color:var(--muted); font-size:12px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }}
  .grid section {{ min-width:0; }}
  .big {{ font-size:22px; font-weight:700; }}
  .big.ok {{ color:var(--ok); }} .big.warn {{ color:var(--warn); }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:20px; font-size:11px;
    border:1px solid var(--line); }}
  .pill.free {{ color:var(--ok); border-color:var(--ok); }}
  .pill.cheap {{ color:var(--accent); border-color:var(--accent); }}
  .pill.priced {{ color:var(--warn); border-color:var(--warn); }}
  .spark {{ font-size:20px; letter-spacing:1px; }}
  .tag {{ display:inline-block; font-size:11px; background:var(--line);
    color:var(--muted); padding:1px 7px; border-radius:20px; margin:0 4px 4px 0; }}
  .badge {{ font-size:11px; padding:1px 6px; border-radius:5px; }}
  .badge.new {{ background:var(--accent); color:#fff; }}
  .badge.crit {{ background:var(--crit); color:#fff; }}
  .upstream {{ border-top:1px solid var(--line); margin-top:12px; padding-top:2px; }}
  .upstream code {{ font-size:12px; }}
  .impact {{ color:var(--warn); font-size:12px; margin-top:4px; }}
  footer {{ text-align:center; color:var(--muted); font-size:12px; padding:0 0 40px; }}
  footer a {{ color:var(--accent); }}
</style>
</head>
<body>
  <header class="top">
    <h1>OpenHarness — Harness Cards</h1>
    <p>The harness is the object of study, not the agent. Each card characterizes one
       module across real sessions.</p>
    <div class="stats">
      <div class="stat"><b>{n}</b> <span>modules</span></div>
      <div class="stat"><b>{total_checks}</b> <span>checks run</span></div>
      <div class="stat"><b>{total_fail}</b> <span>violations caught</span></div>
      <div class="stat"><b>{total_tokens}</b> <span>tokens spent</span></div>
    </div>
  </header>
  <main>
    {body}
  </main>
  <footer>
    Generated by <a href="https://github.com/ginaecho/lego-bricks-token-prediction">OpenHarness</a>
    — see the harness, share the harness, prove the harness works.
  </footer>
</body>
</html>"""
