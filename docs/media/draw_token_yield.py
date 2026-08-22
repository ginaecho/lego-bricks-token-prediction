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
MEASURED = "#2e6f8e"   # things we actually ran
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


# ── the numbers, straight from the engine ────────────────────────────────

def build_store() -> CalibrationStore:
    """The same calibration data as examples/token_yield_demo.py."""
    s = CalibrationStore()
    for t, d in [(12_400, 180), (14_200, 210), (11_800, 165), (13_600, 195), (15_000, 220)]:
        s.add(CalibrationRecord("bug_fix", t, duration_seconds=d, harness_tokens=1200))
    for t, d in [(28_000, 420), (32_000, 480), (25_500, 390), (30_000, 450)]:
        s.add(CalibrationRecord("feature", t, duration_seconds=d, harness_tokens=2400))
    for t, d in [(18_000, 300), (20_000, 330), (19_000, 315)]:
        s.add(CalibrationRecord("refactor", t, duration_seconds=d, harness_tokens=1800))
    for t, d in [(4_500, 60), (5_000, 75), (4_200, 55), (4_800, 65), (5_200, 80), (4_600, 62)]:
        s.add(CalibrationRecord("docs", t, duration_seconds=d, harness_tokens=600))
    for t, d in [(22_000, 360), (25_000, 400), (20_000, 340)]:
        s.add(CalibrationRecord("data_analysis", t, duration_seconds=d, harness_tokens=2000))
    return s


PROJECT = [
    ("bug_fix", ComplexityTier.PLUS, 8),
    ("feature", ComplexityTier.PLUS, 5),
    ("refactor", ComplexityTier.BASE, 3),
    ("data_analysis", ComplexityTier.BASE, 2),
    ("docs", ComplexityTier.PLUS, 5),
]


def figures_data() -> dict:
    store = build_store()
    pred = TokenPredictor(store)

    spec = ProjectSpec("Q3 Platform Upgrade", interaction_overhead=0.15)
    for tt, tier, n in PROJECT:
        spec.add(tt, tier, count=n)
    fc = ProjectForecaster(store).forecast(spec)

    # the naive sum: the same tasks, priced as if each ran in isolation
    naive = sum((pred.predict_single(tt, tier).predicted_tokens
                 + pred.predict_single(tt, tier).harness_overhead) * n
                for tt, tier, n in PROJECT)

    return {
        "stats": {tt: store.stats(tt) for tt in store.task_types},
        "ladder": pred.predict_scaled("bug_fix"),
        "forecast": fc,
        "naive": naive,
        "distinct": len({tt for tt, _, _ in PROJECT}),
    }


# ── figure 1: the concept ────────────────────────────────────────────────

def draw_concept(data: dict, glyphs: Glyphs) -> str:
    W, H = 1280, 620
    p = Pen(W, H, glyphs, seed=1017)

    p.text(W / 2, 54, "Token Yield", 42, INK, "middle", bold=1.1)
    p.line(W / 2 - 100, 64, W / 2 + 100, 64, INFERRED, 3, rough=1.4)
    p.text(W / 2, 92, "measure three doses  →  prescribe any project", 20, FAINT, "middle")

    panels = [
        (31, "1", "MEASURE", "run each type for real", MEASURED),
        (347, "2", "SCALE", "infer A+ and A++ from A", INFERRED),
        (663, "3", "COMPOSE", "mixing is not a sum", COMPOSED),
        (979, "4", "BUDGET", "the number for your sponsor", BUDGET),
    ]
    PY, PW, PH = 122, 270, 424

    for px, num, title, sub, color in panels:
        p.rect(px, PY, PW, PH, EDGE, 2.4, rough=1.1, r=8)
        p.solid_rect(px + 6, PY + 6, PW - 12, 42, color, 0.10, rx=6)
        p.text(px + 17, PY + 36, f"{num}.", 24, color, bold=0.9)
        p.text(px + 44, PY + 36, title, 23, INK, bold=0.9)
        p.text(px + 17, PY + 68, sub, 15, FAINT)

    for i in range(3):
        ax = panels[i][0] + PW + 6
        p.arrow(ax, PY + PH / 2, ax + 34, PY + PH / 2, FAINT, 2.4, 10)

    # ── panel 1: measured baselines ──────────────────────────────────────
    px = panels[0][0]
    st = data["stats"]
    rows = [("A", "bug_fix", st["bug_fix"]),
            ("B", "feature", st["feature"]),
            ("C", "docs", st["docs"])]
    scale = 168 / max(r[2].mean_tokens for r in rows)
    y = PY + 112
    for letter, name, s in rows:
        p.text(px + 17, y, letter, 23, MEASURED, bold=0.9)
        p.text(px + 40, y, name, 13, INK, mono=True)
        p.text(px + PW - 17, y, f"n={s.sample_count}", 14, FAINT, anchor="end")
        bw = s.mean_tokens * scale
        p.bar(px + 17, y + 11, bw, 23, MEASURED)
        sd = s.stddev_tokens * scale
        p.line(px + 17 + bw - sd, y + 22, px + 17 + bw + sd, y + 22,
               INK, 1.6, single=True, opacity=0.6)
        p.tick(px + 17 + bw + sd, y + 22, 6, INK, 1.5)
        p.tick(px + 17 + bw - sd, y + 22, 6, INK, 1.5)
        p.text(px + 17 + bw + sd + 9, y + 29, fmt(s.mean_tokens), 17, INK, bold=0.5)
        y += 76

    p.line(px + 17, PY + 338, px + PW - 17, PY + 338, EDGE, 1.8, dash="5 5")
    p.rows(px + 17, PY + 364,
           ["tokens and hours, recorded", "per run. whiskers are ±σ", "over n real runs."], 15)

    # ── panel 2: the complexity ladder ───────────────────────────────────
    px = panels[1][0]
    ladder = data["ladder"]
    tiers = [(ComplexityTier.BASE, "A", "1×"),
             (ComplexityTier.PLUS, "A+", "2×"),
             (ComplexityTier.PLUS_PLUS, "A++", "4×")]
    scale = 150 / ladder[ComplexityTier.PLUS_PLUS].predicted_tokens
    y = PY + 112
    for i, (tier, label, mult) in enumerate(tiers):
        pr = ladder[tier]
        measured = tier is ComplexityTier.BASE
        color = MEASURED if measured else INFERRED
        p.text(px + 17, y, label, 23, color, bold=0.9)
        p.text(px + 66, y, mult, 17, FAINT)
        p.text(px + PW - 17, y, "measured" if measured else "inferred",
               13, color, anchor="end")
        bw = pr.predicted_tokens * scale
        p.bar(px + 17, y + 11, bw, 23, color, hatch=not measured)
        p.text(px + 17 + bw + 9, y + 29, fmt(pr.predicted_tokens), 17, INK, bold=0.5)
        if i < 2:                     # the ×2 chain, in the empty band between rows
            p.arrow(px + 150, y + 42, px + 150, y + 64, INFERRED, 1.8, 7)
            p.text(px + 159, y + 60, "× 2", 15, INFERRED)
        y += 76

    p.line(px + 17, PY + 338, px + PW - 17, PY + 338, EDGE, 1.8, dash="5 5")
    p.rows(px + 17, PY + 364,
           ["multipliers are per-type and", "calibrated — not one global", "guess for everything."], 15)

    # ── panel 3: the composition surcharge ───────────────────────────────
    px = panels[2][0]
    fc = data["forecast"]
    naive, total = data["naive"], fc.total_with_overhead
    scale = 196 / total
    nb = naive * scale

    p.text(px + 17, PY + 116, "naive Σ of the parts", 16, FAINT)
    p.bar(px + 17, PY + 126, nb, 26, FAINT)
    p.text(px + 17, PY + 176, fmt(naive), 20, FAINT, bold=0.5)

    p.text(px + 17, PY + 224, "what it actually costs", 16, INK)
    p.bar(px + 17, PY + 234, nb, 26, COMPOSED)
    p.bar(px + 17 + nb, PY + 234, (total - naive) * scale, 26, ALERT, hatch=True)
    p.text(px + 17, PY + 284, fmt(total), 22, COMPOSED, bold=0.6)

    sx, ex = px + 17 + nb, px + 17 + total * scale
    p.line(sx, PY + 264, sx, PY + 276, ALERT, 1.6, single=True)
    p.line(ex, PY + 264, ex, PY + 276, ALERT, 1.6, single=True)
    p.line(sx, PY + 276, ex, PY + 276, ALERT, 1.8, single=True)
    pct = (total - naive) / naive * 100
    p.text((sx + ex) / 2, PY + 300, f"+{pct:.0f}%", 19, ALERT, "middle", bold=0.6)

    p.line(px + 17, PY + 338, px + PW - 17, PY + 338, EDGE, 1.8, dash="5 5")
    p.rows(px + 17, PY + 364,
           [f"{data['distinct']} task types in one project:",
            "context switching, shared setup,", "dependency chains. Priced."], 15)

    # ── panel 4: the budget ──────────────────────────────────────────────
    px = panels[3][0]
    cx = px + PW / 2

    p.text(cx, PY + 152, fmt(total), 58, BUDGET, "middle", bold=1.2)
    p.text(cx, PY + 180, "tokens", 17, FAINT, "middle")
    p.line(px + 24, PY + 202, px + PW - 24, PY + 202, EDGE, 1.8, dash="5 5")

    p.text(px + 28, PY + 240, f"${fc.cost_at_rate(3.0):.2f}", 32, INK, bold=0.8)
    p.text(px + 28, PY + 260, "at $3 / M tokens", 14, FAINT)
    p.text(px + PW - 28, PY + 240, f"{fc.estimated_hours:.1f} h", 32, INK,
           anchor="end", bold=0.8)
    p.text(px + PW - 28, PY + 260, "agent time", 14, FAINT, anchor="end")

    bx, bw2, by = px + 30, PW - 60, PY + 302
    lo, hi = fc.total_tokens_low, fc.total_tokens_high
    p.line(bx, by, bx + bw2, by, BUDGET, 2.2)
    p.tick(bx, by, 8, BUDGET, 2.0)
    p.tick(bx + bw2, by, 8, BUDGET, 2.0)
    p.circle(bx + bw2 * (total - lo) / (hi - lo), by, 6, BUDGET)
    p.text(bx, by + 24, fmt(lo), 14, FAINT)
    p.text(bx + bw2, by + 24, fmt(hi), 14, FAINT, anchor="end")

    p.line(px + 17, PY + 338, px + PW - 17, PY + 338, EDGE, 1.8, dash="5 5")
    p.rows(px + 17, PY + 364,
           ["95% CI, propagated from the", "measured σ. Re-forecast as real",
            "runs land — the band tightens."], 15)

    p.text(W / 2, H - 24,
           "every number above is computed by token_yield, not drawn by hand"
           "   ·   python docs/media/draw_token_yield.py",
           13, FAINT, "middle", mono=True)

    return p.svg("Token Yield concept: measure, scale, compose, budget")


# ── figure 2: the architecture ───────────────────────────────────────────

def draw_architecture(data: dict, glyphs: Glyphs) -> str:
    W, H = 1280, 916
    p = Pen(W, H, glyphs, seed=4211)

    p.text(W / 2, 54, "Token Yield — architecture", 38, INK, "middle", bold=1.1)
    p.line(W / 2 - 186, 64, W / 2 + 186, 64, COMPOSED, 3, rough=1.4)
    p.text(W / 2, 92, "four stages, each a plain dataclass boundary you can test",
           19, FAINT, "middle")

    BX, BW, BH = 440, 400, 80
    spine = BX + BW / 2

    stages = [
        (230, "CalibrationStore", "calibrate.py", MEASURED),
        (378, "TokenPredictor", "predict.py", INFERRED),
        (526, "ProjectForecaster", "forecast.py", COMPOSED),
        (674, "ProjectForecast", "models.py · report.py", BUDGET),
    ]

    # two independent sources, elbowed into stage 1
    for ix, t1, t2 in [(224, "real task runs", "tokens · duration · harness"),
                       (806, "OpenHarness trace", "Observation stream")]:
        p.rect(ix, 108, 250, 58, EDGE, 2.2, rough=1.1, r=6)
        p.text(ix + 125, 134, t1, 18, INK, "middle", bold=0.5)
        p.text(ix + 125, 154, t2, 12, FAINT, "middle", mono=True)
        drop = spine + (56 if ix > 500 else -56)
        p.line(ix + 125, 168, ix + 125, 192, MEASURED, 2.0)
        p.line(ix + 125, 192, drop, 192, MEASURED, 2.0)
        p.arrow(drop, 192, drop, 224, MEASURED, 2.0, 9)

    for i, (y, name, mod, color) in enumerate(stages):
        p.solid_rect(BX, y, BW, BH, color, 0.09, rx=8)
        p.rect(BX, y, BW, BH, color, 2.6, rough=1.0, r=8)
        p.text(spine, y + 36, name, 27, INK, "middle", bold=0.8)
        p.text(spine, y + 60, mod, 13, color, "middle", mono=True)
        if i < 3:
            p.arrow(spine, y + BH + 4, spine, stages[i + 1][0] - 6, FAINT, 2.4, 10)

    # what flows down the spine
    for i, label in enumerate(["TaskTypeStats · n, mean, σ, min, max",
                               "TaskPrediction · tokens, CI, duration",
                               "totals + interaction overhead"]):
        p.text(spine + 18, stages[i][0] + BH + 38, label, 13, FAINT, mono=True)

    # left-hand inputs into the middle stages
    for y, rows_, color in [
        (374, ["ComplexityTier", "base 1× · plus 2× · plus_plus 4×",
               "or your own multiplier"], INFERRED),
        (522, ["ProjectSpec", "task type × tier × count",
               "interaction_overhead = 0.15"], COMPOSED),
    ]:
        p.rect(64, y, 306, 92, color, 2.2, rough=1.1, r=6)
        p.text(80, y + 30, rows_[0], 20, color, bold=0.6)
        p.rows(80, y + 54, rows_[1:], 12.5, FAINT, lh=19, mono=True)
        p.arrow(374, y + 46, BX - 6, y + 46, color, 2.2, 9)

    # right-hand method lists
    for i, rows_ in enumerate([
        ["add() · add_many()", "from_observations()", "stats() · all_stats()"],
        ["predict_single()", "predict_scaled()", "predict_combined()",
         "compare_scenarios()"],
        ["forecast()", "forecast_with_cost()"],
        ["text_report()", "markdown_report()", "cost_at_rate()"],
    ]):
        y, color = stages[i][0], stages[i][3]
        p.line(BX + BW + 6, y + 40, BX + BW + 40, y + 40, color, 1.8,
               single=True, dash="4 4")
        p.rows(BX + BW + 50, y + 24, rows_, 13.5, color, lh=19, mono=True)

    # outputs
    fc = data["forecast"]
    last = stages[3][0] + BH
    outs = [(300, fmt(fc.total_with_overhead) + " tokens",
             f"{fmt(fc.total_tokens_low)} – {fmt(fc.total_tokens_high)}"),
            (535, f"${fc.cost_at_rate(3.0):.2f}", "at $3 / M tokens"),
            (770, f"{fc.estimated_hours:.1f} hours", "agent time")]
    for ox, big, small in outs:
        p.arrow(ox + 105, last + 4, ox + 105, last + 31, BUDGET, 2.2, 9)
        p.rect(ox, last + 35, 210, 68, BUDGET, 2.4, rough=1.05, r=6)
        p.text(ox + 105, last + 65, big, 24, BUDGET, "middle", bold=0.7)
        p.text(ox + 105, last + 87, small, 12, FAINT, "middle", mono=True)

    p.text(W / 2, H - 26,
           "the agent is the test rig; the project budget is the object of study",
           16, FAINT, "middle")

    return p.svg("Token Yield architecture: calibrate, predict, forecast, report")


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
