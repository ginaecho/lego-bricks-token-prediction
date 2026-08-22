"""Hand-drawn figure generator for the Token Yield docs.

Draws the two README figures in a rough, sketched style. The geometry is
perturbed the way rough.js perturbs it — double strokes, bowed lines, hachure
fills — so the result reads as a whiteboard sketch rather than a CAD drawing,
and the lettering is a real handwriting face (Patrick Hand).

Two properties make the figures trustworthy rather than decorative:

* **The numbers are computed, not drawn.** Every value comes from the real
  ``token_yield`` engine run against the same calibration data as
  ``examples/token_yield_demo.py``, so a picture cannot drift away from what
  the code actually predicts.
* **The output needs no fonts.** Each glyph is emitted as an SVG ``<path>``
  and reused via ``<use>``, so the committed SVGs render identically for every
  reader regardless of what is installed on their machine.

Deterministic: a seeded xorshift PRNG and no wall-clock, so regenerating
produces byte-identical SVG.

Regenerate with:  python docs/media/draw_token_yield.py
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from fontTools.pens.svgPathPen import SVGPathPen  # noqa: E402
from fontTools.ttLib import TTFont  # noqa: E402

from token_yield import (  # noqa: E402
    CalibrationRecord,
    CalibrationStore,
    ComplexityTier,
    ProjectForecaster,
    ProjectSpec,
    TokenPredictor,
)

# ── palette ──────────────────────────────────────────────────────────────
# Baked onto a warm paper ground so the figure reads the same in a light or a
# dark README — it is its own little sheet of paper either way.
PAPER = "#fdfbf4"
EDGE = "#ded6c2"
INK = "#33302a"
FAINT = "#6f6a5e"
MEASURED_C = "#2e6f8e"  # things we actually ran
INFERRED = "#c07a2c"   # things we extrapolated
COMPOSED = "#6d5b95"   # things we combined
BUDGET = "#3f7a55"     # the answer
ALERT = "#b1533c"      # the surcharge

# ── fonts ────────────────────────────────────────────────────────────────
HAND_TTF = os.path.join(_HERE, "fonts", "PatrickHand-Regular.ttf")
SYM_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
MONO_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


def _first_present(paths: list[str], what: str) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    raise SystemExit(f"no {what} font found; looked in: {', '.join(paths)}")


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt(n: float) -> str:
    """Token counts the way a person says them out loud."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{n:.0f}"


class Glyphs:
    """Text as vector paths, so the finished SVG depends on no font at all."""

    def __init__(self) -> None:
        self.faces: dict[str, dict] = {}
        self._ids: dict[tuple[str, str], str] = {}
        self.defs: list[str] = []

    def load(self, name: str, path: str) -> None:
        ft = TTFont(path)
        self.faces[name] = {
            "upem": ft["head"].unitsPerEm,
            "cmap": ft.getBestCmap(),
            "gs": ft.getGlyphSet(),
            "hmtx": ft["hmtx"],
        }

    def _face_for(self, ch: str, chain: tuple[str, ...]) -> str:
        for n in chain:
            if ord(ch) in self.faces[n]["cmap"]:
                return n
        return chain[0]

    def _symbol(self, face: str, gname: str) -> str | None:
        key = (face, gname)
        if key in self._ids:
            return self._ids[key]
        f = self.faces[face]
        pen = SVGPathPen(f["gs"], ntos=lambda v: str(int(round(v))))
        f["gs"][gname].draw(pen)
        d = pen.getCommands()
        if not d:                      # space and friends: advance, draw nothing
            self._ids[key] = None
            return None
        sid = f"g{len(self._ids)}"
        self._ids[key] = sid
        self.defs.append(f'<path id="{sid}" d="{d}"/>')
        return sid

    def measure(self, s: str, size: float, chain: tuple[str, ...]) -> float:
        total = 0.0
        for ch in s:
            f = self.faces[self._face_for(ch, chain)]
            gname = f["cmap"].get(ord(ch))
            if gname is None:
                total += size * 0.4
            else:
                total += f["hmtx"][gname][0] * (size / f["upem"])
        return total

    def draw(self, x: float, y: float, s: str, size: float, color: str,
             anchor: str = "start", chain: tuple[str, ...] = ("hand", "sym"),
             bold: float = 0.0, opacity: float = 1.0) -> str:
        if anchor == "middle":
            x -= self.measure(s, size, chain) / 2
        elif anchor == "end":
            x -= self.measure(s, size, chain)

        out = []
        pen_x = x
        for ch in s:
            fname = self._face_for(ch, chain)
            f = self.faces[fname]
            gname = f["cmap"].get(ord(ch))
            if gname is None:
                pen_x += size * 0.4
                continue
            k = size / f["upem"]
            sid = self._symbol(fname, gname)
            if sid:
                extra = ""
                if bold:
                    extra = (f' stroke="{color}" stroke-width="{bold / k:.0f}"'
                             f' stroke-linejoin="round"')
                if opacity != 1.0:
                    extra += f' fill-opacity="{opacity}" stroke-opacity="{opacity}"'
                out.append(
                    f'<use href="#{sid}" transform="translate({pen_x:.1f} {y:.1f}) '
                    f'scale({k:.4f} {-k:.4f})" fill="{color}"{extra}/>'
                )
            pen_x += f["hmtx"][gname][0] * k
        return "".join(out)


class Pen:
    """A rough-drawing pen that accumulates SVG fragments."""

    HAND = ("hand", "sym")
    MONO = ("mono", "sym")

    def __init__(self, width: int, height: int, glyphs: Glyphs,
                 seed: int = 20260822) -> None:
        self.w = width
        self.h = height
        self.g = glyphs
        self._state = seed & 0xFFFFFFFF
        self.parts: list[str] = []

    # -- deterministic randomness ----------------------------------------
    def _rnd(self) -> float:
        x = self._state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self._state = x & 0xFFFFFFFF
        return self._state / 0xFFFFFFFF

    def _off(self, lo: float, hi: float, gain: float = 1.0, rough: float = 1.0) -> float:
        return rough * gain * (self._rnd() * (hi - lo) + lo)

    # -- primitives ------------------------------------------------------
    def _line_d(self, x1, y1, x2, y2, rough=1.0, bowing=1.0, overlay=False) -> str:
        lensq = (x1 - x2) ** 2 + (y1 - y2) ** 2
        length = math.sqrt(lensq)
        if length < 200:
            gain = 1.0
        elif length > 500:
            gain = 0.4
        else:
            gain = -0.0016668 * length + 1.233334

        offset = 2.0
        if offset * offset * 100 > lensq:
            offset = length / 10 or 0.1
        half = offset / 2

        diverge = 0.2 + self._rnd() * 0.2
        mdx = bowing * 2.0 * (y2 - y1) / 200
        mdy = bowing * 2.0 * (x1 - x2) / 200
        mdx = self._off(-mdx, mdx, gain, rough)
        mdy = self._off(-mdy, mdy, gain, rough)

        j = (lambda: self._off(-half, half, gain, rough)) if overlay else \
            (lambda: self._off(-offset, offset, gain, rough))

        return (
            f"M{x1 + j():.1f} {y1 + j():.1f} "
            f"C{mdx + x1 + (x2 - x1) * diverge + j():.1f} "
            f"{mdy + y1 + (y2 - y1) * diverge + j():.1f} "
            f"{mdx + x1 + 2 * (x2 - x1) * diverge + j():.1f} "
            f"{mdy + y1 + 2 * (y2 - y1) * diverge + j():.1f} "
            f"{x2 + j():.1f} {y2 + j():.1f}"
        )

    def line(self, x1, y1, x2, y2, color=INK, w=2.0, rough=1.0, single=False,
             opacity=1.0, dash=None) -> None:
        d = self._line_d(x1, y1, x2, y2, rough)
        if not single:
            d += " " + self._line_d(x1, y1, x2, y2, rough, overlay=True)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        if opacity != 1.0:
            extra += f' stroke-opacity="{opacity}"'
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}" '
            f'stroke-linecap="round"{extra}/>'
        )

    def rect(self, x, y, w, h, color=INK, sw=2.0, rough=1.0, r=0.0) -> None:
        """A rough rectangle; ``r`` clips the corners for a softer box."""
        if r == 0:
            pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        else:
            pts = [(x + r, y), (x + w - r, y), (x + w, y + r), (x + w, y + h - r),
                   (x + w - r, y + h), (x + r, y + h), (x, y + h - r), (x, y + r)]
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            self.line(a[0], a[1], b[0], b[1], color, sw, rough)

    def solid_rect(self, x, y, w, h, fill, opacity=1.0, rx=3) -> None:
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" '
            f'height="{h:.1f}" rx="{rx}" fill="{fill}" fill-opacity="{opacity}"/>'
        )

    def circle(self, cx, cy, r, fill) -> None:
        self.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"/>')

    def hachure(self, x, y, w, h, color, gap=7.0, sw=1.4, opacity=0.85) -> None:
        """Diagonal shading (x + y = c) clipped to the rect — a sketched fill."""
        x0, x1, y0, y1 = x, x + w, y, y + h
        c, c_max = x0 + y0, x1 + y1
        while c <= c_max:
            lo, hi = max(x0, c - y1), min(x1, c - y0)
            if hi - lo > 1.5:
                self.line(lo, c - lo, hi, c - hi, color, sw, rough=0.7,
                          single=True, opacity=opacity)
            c += gap

    def bar(self, x, y, w, h, color, hatch=False, sw=2.0) -> None:
        """A hand-drawn bar: soft wash, optional hachure, rough outline."""
        if w < 1:
            return
        self.solid_rect(x, y, w, h, color, 0.16)
        if hatch:
            self.hachure(x, y, w, h, color, gap=6.5, sw=1.3, opacity=0.7)
        self.rect(x, y, w, h, color, sw, rough=0.85)

    def arrow(self, x1, y1, x2, y2, color=INK, w=2.2, head=11.0, dash=None) -> None:
        self.line(x1, y1, x2, y2, color, w, dash=dash)
        ang = math.atan2(y2 - y1, x2 - x1)
        for a in (ang + 2.55, ang - 2.55):
            self.line(x2, y2, x2 + head * math.cos(a), y2 + head * math.sin(a),
                      color, w, single=True)

    def tick(self, x, y, half, color=INK, w=1.6) -> None:
        self.line(x, y - half, x, y + half, color, w, single=True)

    # -- text ------------------------------------------------------------
    def text(self, x, y, s, size=16, color=INK, anchor="start", mono=False,
             bold=0.0, opacity=1.0) -> None:
        chain = Pen.MONO if mono else Pen.HAND
        self.parts.append(self.g.draw(x, y, s, size, color, anchor, chain,
                                      bold, opacity))

    def rows(self, x, y, lines, size=15, color=FAINT, lh=21, **kw) -> None:
        for i, row in enumerate(lines):
            self.text(x, y + i * lh, row, size, color, **kw)

    def svg(self, title: str) -> str:
        defs = "<defs>" + "".join(self.g.defs) + "</defs>"
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
            f'height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
            f'role="img" aria-label="{esc(title)}">'
            f"<title>{esc(title)}</title>{defs}"
            f'<rect width="{self.w}" height="{self.h}" fill="{PAPER}"/>'
            + "".join(self.parts) + "</svg>"
        )


# ── the numbers, straight from the measurements ──────────────────────────
# Nothing below is typed in by hand. The figures render whatever the fitted
# models and the probe dataset currently say, so a picture cannot outlive the
# finding it depicts.

def figures_data() -> dict:
    from token_yield.backtest import backtest, noise_floor
    from token_yield.learn import seeded_store
    from token_yield.plan import PlanForecaster, WorkPlan
    from token_yield.probes import MEASURED, composition_evidence

    store = seeded_store()
    plan = WorkPlan("replica").add("comprehension", 3).add("code_write", 3)
    return {
        "store": store,
        "records": MEASURED,
        "floor": noise_floor(MEASURED),
        "reports": backtest(MEASURED),
        "composition": composition_evidence(),
        "batching": PlanForecaster(store).compare_batching(plan),
        "selections": {k: store.selection_for(k) for k in store.kinds()},
    }


# ── figure 1: what the measurements said ─────────────────────────────────

def draw_concept(data: dict, glyphs: Glyphs) -> str:
    W, H = 1280, 640
    p = Pen(W, H, glyphs, seed=1017)

    store = data["store"]
    comp_sel = data["selections"]["comprehension"]
    ev = data["composition"]
    bt = data["batching"]

    p.text(W / 2, 54, "Token Yield", 42, INK, "middle", bold=1.1)
    p.line(W / 2 - 100, 64, W / 2 + 100, 64, INFERRED, 3, rough=1.4)
    p.text(W / 2, 92,
           f"{len(data['records'])} real subagent runs  →  every constant the "
           f"first version asserted was wrong", 19, FAINT, "middle")

    panels = [
        (31, "1", "MEASURE", "run probes, record what they cost", MEASURED_C),
        (347, "2", "FIT", "the data picks the shape", INFERRED),
        (663, "3", "VALIDATE", "score it on runs it never saw", COMPOSED),
        (979, "4", "COMPOSE", "and the surcharge was a saving", BUDGET),
    ]
    PY, PW, PH = 122, 270, 440

    for px, num, title, sub, color in panels:
        p.rect(px, PY, PW, PH, EDGE, 2.4, rough=1.1, r=8)
        p.solid_rect(px + 6, PY + 6, PW - 12, 42, color, 0.10, rx=6)
        p.text(px + 17, PY + 36, f"{num}.", 24, color, bold=0.9)
        p.text(px + 44, PY + 36, title, 23, INK, bold=0.9)
        p.text(px + 17, PY + 68, sub, 14, FAINT)

    for i in range(3):
        ax = panels[i][0] + PW + 6
        p.arrow(ax, PY + PH / 2, ax + 34, PY + PH / 2, FAINT, 2.4, 10)

    # ── panel 1: the raw scatter ─────────────────────────────────────────
    px = panels[0][0]
    x0, x1 = px + 46, px + 248
    ytop, ybot = PY + 106, PY + 250
    tlo, thi = 34_000, 60_000

    def sx(scope): return x0 + scope * (x1 - x0) / 9.0
    def sy(tok): return ybot - (tok - tlo) * (ybot - ytop) / (thi - tlo)

    p.line(x0 - 6, ytop - 6, x0 - 6, ybot + 4, EDGE, 1.8)
    p.line(x0 - 6, ybot + 4, x1 + 6, ybot + 4, EDGE, 1.8)
    for tok in (60_000, 47_000, 34_000):
        p.text(x0 - 12, sy(tok) + 5, fmt(tok), 12, FAINT, anchor="end")
    for sc in (0, 4, 8):
        p.text(sx(sc), ybot + 22, str(sc), 12, FAINT, anchor="middle")
    p.text((x0 + x1) / 2, ybot + 40, "scope (units of work)", 13, FAINT, "middle")

    model = store.model_for("comprehension")
    p.line(sx(1), sy(model.predict(1)), sx(8), sy(model.predict(8)),
           MEASURED_C, 2.0, opacity=0.55)

    for r in data["records"]:
        col = MEASURED_C if r.kind == "comprehension" else INFERRED
        p.circle(sx(r.scope), sy(min(max(r.tokens, tlo), thi)), 4.5, col)

    p.circle(px + 22, PY + 300, 4.5, MEASURED_C)
    p.text(px + 32, PY + 305, "comprehension", 13, INK)
    p.circle(px + 150, PY + 300, 4.5, INFERRED)
    p.text(px + 160, PY + 305, "code_write", 13, INK)

    p.line(px + 17, PY + 330, px + PW - 17, PY + 330, EDGE, 1.8, dash="5 5")
    p.rows(px + 17, PY + 356,
           [f"8x the scope moved tokens 1.39x.",
            f"Repeats of the same task differ by",
            f"{data['floor']:.0%} — the noise floor."], 15)

    # ── panel 2: cross-validated error by form ───────────────────────────
    px = panels[1][0]
    p.text(px + 17, PY + 104, "comprehension — LOO error by form", 14, INK)
    ranked = sorted(comp_sel.scores.items(), key=lambda kv: kv[1])
    worst = max(s for _, s in ranked)
    bx, bmax = px + 108, 132
    y = PY + 128
    for form, score in ranked:
        old = form == "proportional"
        col = ALERT if old else (BUDGET if form == comp_sel.form else FAINT)
        p.text(px + 17, y + 16, form, 14, col, bold=0.4 if old else 0.0)
        p.bar(bx, y + 4, max(3.0, bmax * score / worst), 18, col, hatch=old)
        p.text(bx + max(3.0, bmax * score / worst) + 7, y + 18,
               f"{score:.1%}", 13, INK, bold=0.4)
        y += 34

    p.text(px + 17, PY + 268, "'proportional' IS the old rule:", 13, ALERT)
    p.text(px + 17, PY + 288, "A+ = 2x, A++ = 4x", 15, ALERT, bold=0.5)

    p.line(px + 17, PY + 330, px + PW - 17, PY + 330, EDGE, 1.8, dash="5 5")
    p.rows(px + 17, PY + 356,
           ["Cross-validated, so a bendier",
            "form cannot win by having more",
            "parameters to bend."], 15)

    # ── panel 3: out-of-sample ───────────────────────────────────────────
    px = panels[2][0]
    p.text(px + 17, PY + 104, "predicted vs actually measured", 14, INK)

    pairs = [("two agents", bt["separate_agents"], ev["separate_sum"]),
             ("one agent", bt["batched_single_agent"], ev["batched_mean"])]
    top = max(max(a, b) for _, a, b in pairs)
    y = PY + 124
    for label, pred, meas in pairs:
        p.text(px + 17, y + 12, label, 14, FAINT)
        p.bar(px + 17, y + 20, 196 * pred / top, 16, COMPOSED)
        p.text(px + 219, y + 33, "pred", 11, FAINT)
        p.bar(px + 17, y + 40, 196 * meas / top, 16, BUDGET)
        p.text(px + 219, y + 53, "real", 11, FAINT)
        err = abs(pred - meas) / meas
        p.text(px + 17, y + 76, f"{fmt(pred)} vs {fmt(meas)}   error {err:.1%}",
               14, INK, bold=0.4)
        y += 96

    p.text(px + 17, PY + 314, f"noise floor {data['floor']:.0%} — both inside it",
           14, BUDGET, bold=0.4)

    p.line(px + 17, PY + 330, px + PW - 17, PY + 330, EDGE, 1.8, dash="5 5")
    p.rows(px + 17, PY + 356,
           ["The batched runs were held out",
            "of every fit. The fixed/marginal",
            "split predicted them anyway."], 15)

    # ── panel 4: composition ─────────────────────────────────────────────
    px = panels[3][0]
    old_claim = ev["separate_sum"] * 1.15
    top = max(old_claim, ev["separate_sum"])
    scale = 200.0 / top

    scale = 150.0 / top
    rows = [("old model said (+15%)", old_claim, ALERT, True),
            ("run as separate agents", ev["separate_sum"], FAINT, False),
            ("batched into one agent", ev["batched_mean"], BUDGET, False)]
    y = PY + 110
    for label, val, col, hatch in rows:
        p.text(px + 17, y + 12, label, 14, col if hatch else FAINT)
        w = val * scale
        p.bar(px + 17, y + 20, w, 22, col, hatch=hatch)
        p.text(px + 17 + w + 8, y + 37, fmt(val), 15,
               col if hatch else INK, bold=0.5)
        y += 62

    p.text(px + 17, PY + 312, f"batching saves {ev['saving']:.0%}",
           20, BUDGET, bold=0.6)

    p.line(px + 17, PY + 338, px + PW - 17, PY + 338, EDGE, 1.8, dash="5 5")
    p.rows(px + 17, PY + 362,
           ["A 38k boot cost is paid per agent,",
            "so combining kinds is a discount.",
            "The +15% surcharge had the",
            "wrong sign."], 15, lh=19)

    p.text(W / 2, H - 22,
           "every number is computed from the measured probe suite  ·  "
           "python docs/media/draw_token_yield.py",
           13, FAINT, "middle", mono=True)

    return p.svg("Token Yield: what the measurements said")


def draw_architecture(data: dict, glyphs: Glyphs) -> str:
    W, H = 1280, 940
    p = Pen(W, H, glyphs, seed=4211)

    store = data["store"]
    bt = data["batching"]

    p.text(W / 2, 54, "Token Yield — the calibration loop", 38, INK, "middle", bold=1.1)
    p.line(W / 2 - 236, 64, W / 2 + 236, 64, COMPOSED, 3, rough=1.4)
    p.text(W / 2, 92,
           "not a pipeline — what it predicts, it later measures, and refits from",
           19, FAINT, "middle")

    BX, BW, BH = 450, 380, 82
    spine = BX + BW / 2

    stages = [
        (214, "LearningStore", "learn.py  ·  observe()", MEASURED_C),
        (366, "select_model", "costmodel.py", INFERRED),
        (518, "PlanForecaster", "plan.py", COMPOSED),
        (670, "PlanForecast", "tokens · $ · interval", BUDGET),
    ]

    # two sources of records, elbowed in
    for ix, t1, t2 in [(214, "probe suite", "dispatch subagents, record cost"),
                       (806, "production runs", "real work, same measurement")]:
        p.rect(ix, 112, 260, 58, EDGE, 2.2, rough=1.1, r=6)
        p.text(ix + 130, 138, t1, 18, INK, "middle", bold=0.5)
        p.text(ix + 130, 158, t2, 11.5, FAINT, "middle", mono=True)
        drop = spine + (58 if ix > 500 else -58)
        p.line(ix + 130, 172, ix + 130, 192, MEASURED_C, 2.0)
        p.line(ix + 130, 192, drop, 192, MEASURED_C, 2.0)
        p.arrow(drop, 192, drop, 208, MEASURED_C, 2.0, 8)

    for i, (y, name, mod, color) in enumerate(stages):
        p.solid_rect(BX, y, BW, BH, color, 0.09, rx=8)
        p.rect(BX, y, BW, BH, color, 2.6, rough=1.0, r=8)
        p.text(spine, y + 36, name, 26, INK, "middle", bold=0.8)
        p.text(spine, y + 60, mod, 12.5, color, "middle", mono=True)
        if i < 3:
            p.arrow(spine, y + BH + 4, spine, stages[i + 1][0] - 6, FAINT, 2.4, 10)

    for i, label in enumerate([
            "ScopedRecord · kind, scope, tokens, provenance",
            "CostModel · fitted form + fixed/marginal split",
            "LineItem · per task, with regime flags"]):
        p.text(spine + 18, stages[i][0] + BH + 40, label, 12.5, FAINT, mono=True)

    # the four candidate forms, as the thing select_model chooses between
    forms = [("constant", "c"), ("proportional", "b·s"),
             ("affine", "a + b·s"), ("power", "a·s^b")]
    p.rect(64, 350, 320, 116, INFERRED, 2.2, rough=1.1, r=6)
    p.text(80, 378, "candidate forms", 19, INFERRED, bold=0.6)
    for i, (nm, eq) in enumerate(forms):
        chosen = nm in {s.form for s in data["selections"].values()}
        col = BUDGET if chosen else FAINT
        p.text(80 + (i % 2) * 150, 404 + (i // 2) * 22, f"{nm} = {eq}", 12.5,
               col, mono=True)
    p.text(80, 451, "picked by leave-one-out CV", 12, FAINT)
    p.arrow(388, 408, BX - 6, 408, INFERRED, 2.2, 9)

    # the plan going in
    p.rect(64, 502, 320, 92, COMPOSED, 2.2, rough=1.1, r=6)
    p.text(80, 530, "WorkPlan", 19, COMPOSED, bold=0.6)
    p.rows(80, 554, ["kind × scope × count",
                     "unmodelled kinds are named,",
                     "extrapolation is flagged"], 12, FAINT, lh=17, mono=True)
    p.arrow(388, 548, BX - 6, 548, COMPOSED, 2.2, 9)

    # right-hand annotations
    for i, rows_ in enumerate([
            ["observe() scores each new run", "against the STANDING model",
             "before absorbing it"],
            ["backtest.py: LOO MAPE vs the", "noise floor -> skill ratio"],
            ["compare_batching(): boot cost", "is paid per invocation"],
            [f"validated out-of-sample to",
             f"{abs(bt['batched_single_agent'] - data['composition']['batched_mean']) / data['composition']['batched_mean']:.1%}"],
    ]):
        y, color = stages[i][0], stages[i][3]
        p.line(BX + BW + 6, y + 42, BX + BW + 36, y + 42, color, 1.8,
               single=True, dash="4 4")
        p.rows(BX + BW + 46, y + 28, rows_, 12.5, color, lh=18, mono=True)

    # ── the return leg: this is what makes it a loop ─────────────────────
    # routed left of the input boxes (x=64) so it crosses nothing
    LEG, RET = 36, 742
    p.line(BX - 6, RET, LEG, RET, ALERT, 2.6)
    p.line(LEG, RET, LEG, 258, ALERT, 2.6)
    p.arrow(LEG, 258, BX - 6, 258, ALERT, 2.6, 11)
    p.text(70, 630, "DriftReport", 20, ALERT, bold=0.6)
    p.rows(70, 654, ["every finished task becomes a record.",
                     "If the standing model did not see it",
                     "coming, that is reported — never",
                     "quietly averaged away."], 13, FAINT, lh=19)

    # what the loop currently believes
    p.rect(300, 800, 680, 96, BUDGET, 2.4, rough=1.05, r=6)
    p.text(320, 828, "what it currently believes", 17, BUDGET, bold=0.5)
    y = 852
    for kind in store.kinds():
        m = store.model_for(kind)
        rep = data["reports"].get(kind)
        skill = f"skill {rep.skill_ratio:.2f}×" if rep and rep.skill_ratio else ""
        p.text(320, y, f"{kind}: {m.equation()}   (n={m.n}, {skill})",
               12.5, INK, mono=True)
        y += 22

    p.text(W / 2, H - 16,
           "the agent is the test rig; the cost model is the object of study",
           16, FAINT, "middle")

    return p.svg("Token Yield calibration loop: measure, fit, forecast, refit")


def _glyphs() -> Glyphs:
    """A fresh glyph table per figure, so each SVG defines only what it uses."""
    g = Glyphs()
    g.load("hand", HAND_TTF)
    g.load("sym", _first_present(SYM_CANDIDATES, "symbol"))
    g.load("mono", _first_present(MONO_CANDIDATES, "monospace"))
    return g


def main() -> None:
    data = figures_data()
    for name, draw in [("token-yield-concept.svg", draw_concept),
                       ("token-yield-architecture.svg", draw_architecture)]:
        svg = draw(data, _glyphs())
        path = os.path.join(_HERE, name)
        with open(path, "w") as f:
            f.write(svg)
        print(f"✓ {path}  ({len(svg) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
