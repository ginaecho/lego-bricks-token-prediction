"""Draw the compositional story: base tasks -> combinations -> pricing the unseen.

The previous figure showed that cost could be *measured*. This one shows what
the measurements are for: a vocabulary of named business tasks, what each one
costs, what happens when they are combined, and how a request nobody has run
before gets priced by writing it in that vocabulary.

Every number is read from the committed campaign (``experiments/train_runs.jsonl``)
and the model fitted from it. Nothing is typed in by hand, so the picture cannot
outlive the finding it depicts.

Run with:  python docs/media/draw_composition.py
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from draw_token_yield import (  # noqa: E402
    ALERT, BUDGET, COMPOSED, EDGE, FAINT, INFERRED, INK, MEASURED_C, Pen,
    _glyphs, fmt,
)
from token_yield.compose import (  # noqa: E402
    batching_saving, default_runs_path, load_runs, noise_floor, select_model,
)
from token_yield.decompose import Decomposition, price  # noqa: E402
from token_yield.tasks import ORDER, PRIMITIVES  # noqa: E402

W, H = 1480, 1180
PAD = 34


# ── data ─────────────────────────────────────────────────────────────────

def gather() -> dict:
    runs = load_runs(default_runs_path())
    model = select_model(runs)
    by = {r.label: r for r in runs}

    ladder = [(by[k].context_bytes, by[k].tokens) for k in
              ("null", "review_tiny", "review_small", "review_medium",
               "review_large", "review_xlarge") if k in by]
    ladder.sort()

    cases_path = os.path.join(_ROOT, "experiments", "decompose_cases.jsonl")
    cases = []
    if os.path.exists(cases_path):
        for line in open(cases_path, encoding="utf-8"):
            if line.strip():
                cases.append(json.loads(line))

    held = [r for r in runs if r.held_out]
    held_err = (sum(abs(model.predict_run(r) - r.tokens) / r.tokens
                    for r in held) / len(held)) if held else 0.0
    return {
        "runs": runs, "model": model, "by": by, "ladder": ladder,
        "cases": cases, "held": held, "held_err": held_err,
        "boot": min(r.tokens for r in runs if r.total_units == 0),
        "floor": noise_floor(runs),
    }


# ── panel frame ──────────────────────────────────────────────────────────

def panel(p: Pen, x, y, w, h, n: str, title: str, sub: str) -> None:
    p.solid_rect(x, y, w, h, "#ffffff", opacity=0.55, rx=6)
    p.rect(x, y, w, h, EDGE, sw=1.8, rough=0.8, r=6)
    p.circle(x + 26, y + 26, 13, INK)
    p.text(x + 26, y + 31, n, size=15, color="#ffffff", anchor="middle")
    p.text(x + 48, y + 31, title, size=21, color=INK)
    p.text(x + 48, y + 52, sub, size=13.5, color=FAINT)


# ── panel 1: the vocabulary ──────────────────────────────────────────────

def p1(p: Pen, d: dict, x, y, w, h) -> None:
    m = d["model"]
    panel(p, x, y, w, h, "1", "The base tasks",
          "nine things an agent does to a business document, and what each unit adds")
    marg = m.marginals()
    top = max(marg.values()) or 1.0
    # Anything smaller than the run-to-run noise is not a cost we can claim to
    # have measured. Saying "+0" would overstate it; saying "-137" would be
    # worse. Both get reported as indistinguishable from zero.
    noise = d["floor"] * d["boot"]
    bx, by = x + 158, y + 82
    bw = w - 210
    row = (h - 108) / len(ORDER)
    for i, slug in enumerate(ORDER):
        pr = PRIMITIVES[slug]
        cy = by + i * row
        p.text(x + 148, cy + 13, pr.name, size=15, color=INK, anchor="end")
        raw = marg.get(slug, 0.0)
        val = max(raw, 0.0)
        tiny = raw < noise
        bar_w = max(2.0, bw * 0.52 * (val / top))
        p.bar(bx, cy, bar_w, 15, MEASURED_C, hatch=tiny)
        p.text(bx + bar_w + 9, cy + 13,
               "~0" if tiny else f"+{fmt(val)}", size=12.5,
               color=FAINT if tiny else MEASURED_C)
        p.text(bx + bw * 0.52 + 96, cy + 13, pr.industry, size=11.5, color=FAINT)
    p.text(x + 158, y + h - 14,
           f"measured marginal tokens per extra unit  ·  ~0 = below the "
           f"{d['floor']:.1%} repeat-run noise floor", size=11.5, color=FAINT)


# ── panel 2: what context costs ──────────────────────────────────────────

def p2(p: Pen, d: dict, x, y, w, h) -> None:
    m, lad = d["model"], d["ladder"]
    panel(p, x, y, w, h, "2", "What adding context costs",
          "the same instruction, pointed at more and more real filings")
    x0, y0 = x + 84, y + h - 74
    pw, ph = w - 140, h - 168
    maxb = max(b for b, _ in lad) or 1
    maxt = max(t for _, t in lad)
    mint = d["boot"] * 0.965

    def px(b): return x0 + pw * (b / maxb)
    def py(t): return y0 - ph * ((t - mint) / (maxt - mint))

    p.line(x0, y0, x0 + pw, y0, EDGE, w=1.6)
    p.line(x0, y0, x0, y0 - ph, EDGE, w=1.6)
    # the fitted line
    p.line(px(0), py(m.coef[0]), px(maxb), py(m.coef[0] + m.byte_slope() * maxb),
           INFERRED, w=2.2)
    # the floor: what an agent costs before any work
    p.line(x0, py(d["boot"]), x0 + pw, py(d["boot"]), FAINT, w=1.4, single=True)
    p.text(x0 + 8, py(d["boot"]) - 9, f"agent start-up {fmt(d['boot'])} — paid "
           f"before any work", size=12, color=FAINT)
    for b, t in lad:
        p.circle(px(b), py(t), 5.5, MEASURED_C)
    p.text(x0 - 10, py(maxt) + 5, fmt(maxt), size=11.5, color=FAINT, anchor="end")
    p.text(x0 - 10, y0 + 5, fmt(int(mint)), size=11.5, color=FAINT, anchor="end")
    p.text(x0, y0 + 22, "0", size=11.5, color=FAINT, anchor="middle")
    p.text(x0 + pw, y0 + 22, f"{maxb // 1024} KB read", size=11.5, color=FAINT,
           anchor="middle")
    p.text(x + 84, y + h - 34,
           f"{m.byte_slope():.3f} tokens per byte — flat, not exponential; "
           f"{(maxt / d['boot'] - 1):.0%} more cost for {maxb // 1024} KB more work",
           size=12.5, color=INK)


# ── panel 3: combining them ──────────────────────────────────────────────

def p3(p: Pen, d: dict, x, y, w, h) -> None:
    m = d["model"]
    panel(p, x, y, w, h, "3", "Combining base tasks",
          "one agent doing several jobs vs one agent per job")
    combos = [
        ({"review": 1, "extract": 3, "validate": 2}, 1235, "Review + 3xExtract + 2xValidate"),
        ({"review": 1, "remediate": 2, "validate": 2}, 1235, "Review + 2xRemediate + 2xValidate"),
        ({"retrieve": 1, "review": 1, "remediate": 1, "validate": 1}, 1235,
         "Retrieve + Review + Remediate + Validate"),
    ]
    widest = max(batching_saving(m, c, b)[1] for c, b, _ in combos)
    bx = x + 40
    bw = w - 96
    for i, (counts, cb, name) in enumerate(combos):
        batched, separate, saving = batching_saving(m, counts, cb)
        ty = y + 96 + i * ((h - 150) / len(combos))
        p.text(bx, ty, name, size=13.5, color=INK)
        sw_ = bw * (separate / widest)
        bw_ = bw * (batched / widest)
        p.bar(bx, ty + 12, sw_, 15, ALERT, hatch=True)
        p.text(bx + sw_ + 8, ty + 25, f"{fmt(separate)} apart", size=11.5, color=ALERT)
        p.bar(bx, ty + 33, bw_, 15, BUDGET)
        p.text(bx + bw_ + 8, ty + 46, f"{fmt(batched)} together", size=11.5,
               color=BUDGET)
        p.text(bx + bw - 4, ty, f"saves {saving:.0%}", size=14, color=BUDGET,
               anchor="end")
    p.text(bx, y + h - 16,
           "the gap is the start-up cost, paid once instead of once per job",
           size=12, color=FAINT)


# ── panel 4: pricing a task nobody has run ───────────────────────────────

def p4(p: Pen, d: dict, x, y, w, h) -> None:
    m, cases = d["model"], d["cases"]
    panel(p, x, y, w, h, "4", "Pricing a request nobody has run",
          "plain English in, base tasks out, tokens predicted — then checked")
    cx = x + 40
    top = y + 86
    errs = []
    for i, c in enumerate(cases[:3]):
        counts = {s: 0 for s in ORDER}
        counts.update(c["encoded"])
        dec = Decomposition(counts, "", c["context_bytes"])
        pred = price(dec, m)
        act = c["actual_tokens"]
        err = abs(pred - act) / act
        errs.append(err)
        ry = top + i * ((h - 168) / 3)
        req = c["request"]
        req = (req[:78] + "...") if len(req) > 78 else req
        p.text(cx, ry, f'"{req}"', size=12, color=FAINT)
        p.text(cx, ry + 22, dec.notation(), size=14.5, color=COMPOSED)
        p.arrow(cx + w - 340, ry + 17, cx + w - 300, ry + 17, INK, w=1.8, head=8)
        p.text(cx + w - 292, ry + 8, f"predicted {fmt(pred)}", size=13,
               color=INFERRED)
        p.text(cx + w - 292, ry + 26, f"actual     {fmt(act)}", size=13,
               color=MEASURED_C)
        p.text(cx + w - 96, ry + 17, f"{err:.1%} off", size=14,
               color=BUDGET if err < 0.05 else ALERT)
    mean_err = sum(errs) / len(errs) if errs else 0.0
    p.line(cx, y + h - 62, x + w - 40, y + h - 62, EDGE, w=1.4, single=True)
    p.text(cx, y + h - 38,
           f"held-out compositions: {d['held_err']:.1%} mean error over "
           f"{len(d['held'])} tasks never fitted", size=13, color=INK)
    p.text(cx, y + h - 18,
           f"plain-English requests: {mean_err:.1%}   ·   "
           f"model {m.loo_mape:.1%} cross-validated   ·   "
           f"repeat-run noise floor {d['floor']:.1%}", size=12.5, color=FAINT)


# ── assemble ─────────────────────────────────────────────────────────────

def draw(d: dict) -> str:
    p = Pen(W, H, _glyphs(), seed=20260824)
    m = d["model"]
    p.text(PAD + 6, 46, "Token Yield: pricing agent work by decomposing it",
           size=30, color=INK)
    p.text(PAD + 6, 72,
           "Measure a small vocabulary of business tasks once. Then price any "
           "combination of them — including ones never run.",
           size=14.5, color=FAINT)

    gw = (W - PAD * 2 - 26) / 2
    gh = (H - 96 - 26 - PAD) / 2
    top = 96
    p1(p, d, PAD, top, gw, gh)
    p2(p, d, PAD + gw + 26, top, gw, gh)
    p3(p, d, PAD, top + gh + 26, gw, gh)
    p4(p, d, PAD + gw + 26, top + gh + 26, gw, gh)

    p.text(PAD + 6, H - 12,
           f"{m.equation()}   ·   fitted on {m.n} measured agent runs over real "
           f"SEC filings   ·   every number here is measured, not asserted",
           size=12, color=FAINT)
    return p.svg("Token Yield: base tasks, composition, and pricing unseen work")


def main() -> None:
    d = gather()
    svg = draw(d)
    out = os.path.join(_HERE, "token-yield-composition.svg")
    with open(out, "w") as f:
        f.write(svg)
    print(f"✓ {out}  ({len(svg) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
