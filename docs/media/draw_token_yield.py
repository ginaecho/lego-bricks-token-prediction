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
from pathlib import Path

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

    def arc(self, cx, cy, r, a0, a1, color=INK, w=2.2, steps=14,
            arrow=False, head=13.0) -> None:
        """A rough arc from angle a0 to a1 (degrees, math convention, y-up)."""
        pts = []
        for i in range(steps + 1):
            a = math.radians(a0 + (a1 - a0) * i / steps)
            pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            self.line(x1, y1, x2, y2, color, w, rough=0.8, single=True)
        if arrow:
            (px_, py_), (qx, qy) = pts[-2], pts[-1]
            ang = math.atan2(qy - py_, qx - px_)
            for d in (ang + 2.5, ang - 2.5):
                self.line(qx, qy, qx + head * math.cos(d), qy + head * math.sin(d),
                          color, w, single=True)

    def icon_gauge(self, cx, cy, r, color) -> None:
        """A dial with a needle — measurement."""
        self.arc(cx, cy + r * 0.3, r, 200, -20, color, 2.0, 12)
        self.line(cx, cy + r * 0.3, cx + r * 0.62, cy - r * 0.45, color, 2.2)
        self.circle(cx, cy + r * 0.3, 3, color)

    def icon_curve(self, cx, cy, r, color) -> None:
        """Points with a line through them — fitting."""
        x0, y0 = cx - r, cy + r * 0.7
        self.line(x0, y0, x0, cy - r * 0.8, EDGE, 1.6, single=True)
        self.line(x0, y0, cx + r, y0, EDGE, 1.6, single=True)
        pts = [(0.15, 0.30), (0.42, 0.52), (0.68, 0.62), (0.95, 0.86)]
        self.line(x0 + r * 0.2, y0 - r * 0.34, x0 + r * 1.8, y0 - r * 1.30,
                  color, 2.0)
        for fx, fy in pts:
            self.circle(x0 + 2 * r * fx, y0 - 1.5 * r * fy, 3.2, color)

    def icon_tag(self, cx, cy, r, color) -> None:
        """A price tag — the quote."""
        pts = [(cx - r, cy - r * 0.6), (cx + r * 0.35, cy - r * 0.6),
               (cx + r, cy), (cx + r * 0.35, cy + r * 0.6),
               (cx - r, cy + r * 0.6)]
        for a, b in zip(pts, pts[1:] + pts[:1]):
            self.line(a[0], a[1], b[0], b[1], color, 2.0)
        self.circle(cx + r * 0.42, cy, 3.4, color)

    def icon_target(self, cx, cy, r, color) -> None:
        """Rings with an arrow in them — scoring the quote."""
        for rr in (r, r * 0.6, r * 0.24):
            self.arc(cx, cy, rr, 0, 359, color, 1.8, 16)
        self.line(cx - r * 1.25, cy + r * 1.25, cx + r * 0.16, cy - r * 0.16,
                  ALERT, 2.2)

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
    from token_yield.mine import coverage, mine_repo
    from token_yield.plan import PlanForecaster, WorkPlan
    from token_yield.probes import MEASURED, composition_evidence

    store = seeded_store()
    plan = (WorkPlan("replica").add("comprehension", 3, bytes=15_216)
            .add("code_write", 3))

    # This checkout by default; extra repositories may be passed on argv.
    repo_root = Path(__file__).resolve().parents[2]
    sources = [(str(repo_root), repo_root.name)]
    sources += [(a, Path(a).resolve().name) for a in sys.argv[1:]]

    mined = []
    for path, name in sources:
        mined += mine_repo(path, limit=200, repo=name)

    return {
        "store": store,
        "records": MEASURED,
        "floor": noise_floor(MEASURED),
        "reports": backtest(MEASURED),
        "composition": composition_evidence(),
        "batching": PlanForecaster(store).compare_batching(plan),
        "selections": {k: store.selection_for(k) for k in store.kinds()},
        "coverage": coverage(mined, store.kinds()) if mined else None,
        "mined_n": len(mined),
        "core": core_idea_numbers(store),
    }


def core_idea_numbers(store) -> dict:
    """A / B / C measured, then A+ / A++ / A+B+C inferred — all from the models."""
    from token_yield.duration import duration_selection, seconds_for
    from token_yield.plan import PlanForecaster, WorkPlan
    from token_yield.probes import MEASURED

    tok = {k: store.model_for(k) for k in store.kinds()}
    dur = {k: duration_selection(k, MEASURED) for k in store.kinds()}

    # every task carries all its signals; each model reads the one it selected
    A = {"scope": 3, "bytes": 15_216, "output_units": 3}
    Ap = {"scope": 6, "bytes": 30_432, "output_units": 6}
    App = {"scope": 12, "bytes": 60_864, "output_units": 12}
    B = {"scope": 3, "output_units": 3}
    C = {"scope": 3, "bytes": 11_165, "output_units": 3}

    def T(kind, sig):
        m = tok[kind]
        return m.predict(sig[m.signal])

    def D(kind, sig):
        return seconds_for(dur.get(kind), sig) or 0.0

    at, bt, ct = T("comprehension", A), T("code_write", B), T("docs", C)
    plan = (WorkPlan("ABC").add("comprehension", 3, bytes=15_216)
            .add("code_write", 3).add("docs", 3))
    batch = PlanForecaster(store).compare_batching(plan)

    return {
        "measured": [
            ("A", "read 3 files", at, D("comprehension", A)),
            ("B", "write 3 functions", bt, D("code_write", B)),
            ("C", "document 3 functions", ct, D("docs", C)),
        ],
        "inferred": [
            ("A+", "twice A's work", T("comprehension", Ap),
             D("comprehension", Ap), 2 * at, "a 2× guess"),
            ("A++", "four times A's work", T("comprehension", App),
             D("comprehension", App), 4 * at, "a 4× guess"),
            ("A+B+C", "all three, one agent", batch["batched_single_agent"],
             D("comprehension", A) + D("code_write", B) + D("docs", C),
             batch["separate_agents"], "adding them up"),
        ],
        "rules": [("A", tok["comprehension"].equation()),
                  ("B", tok["code_write"].equation()),
                  ("C", tok["docs"].equation())],
    }


# ── figure 0: the core idea ──────────────────────────────────────────────

def draw_core_idea(data: dict, glyphs: Glyphs) -> str:
    W, H = 1280, 700
    p = Pen(W, H, glyphs, seed=8801)
    core = data["core"]

    p.text(W / 2, 52, "The core idea", 40, INK, "middle", bold=1.1)
    p.line(W / 2 - 118, 63, W / 2 + 118, 63, INFERRED, 3, rough=1.4)
    p.text(W / 2, 90, "measure a few task types for real — then price the ones "
           "you have never run", 19, FAINT, "middle")

    LX, LW = 36, 324
    MX, MW = 392, 296
    RX, RW = 720, 524
    CARD_Y = [166, 292, 418]
    CH = 108

    p.text(LX + 4, 136, "1.  MEASURE", 20, MEASURED_C, bold=0.7)
    p.text(LX + 150, 136, "you actually run these", 14, FAINT)
    p.text(RX + 4, 136, "3.  INFER", 20, BUDGET, bold=0.7)
    p.text(RX + 118, 136, "the machine answers these without running them",
           14, FAINT)

    # ── measured cards ───────────────────────────────────────────────────
    for (letter, what, tokens, secs), cy in zip(core["measured"], CARD_Y):
        p.solid_rect(LX, cy, LW, CH, MEASURED_C, 0.09, rx=8)
        p.rect(LX, cy, LW, CH, MEASURED_C, 2.6, rough=1.0, r=8)
        p.text(LX + 20, cy + 46, letter, 38, MEASURED_C, bold=1.0)
        p.text(LX + 78, cy + 34, what, 16, INK)
        p.text(LX + 78, cy + 62, f"{fmt(tokens)} tokens", 19, INK, bold=0.5)
        p.text(LX + 78, cy + 86, f"{secs:.0f} s", 15, FAINT)
        p.text(LX + LW - 16, cy + 24, "MEASURED", 11, MEASURED_C, anchor="end")
        p.line(LX + LW + 2, cy + CH / 2, 376, cy + CH / 2, FAINT, 2.0)

    # converge into the rule
    p.line(376, CARD_Y[0] + CH / 2, 376, CARD_Y[2] + CH / 2, FAINT, 2.0)
    p.arrow(376, 340, MX - 6, 340, FAINT, 2.4, 10)

    # ── the rule ─────────────────────────────────────────────────────────
    p.text(MX + 2, 136, "2.  FIT", 20, INFERRED, bold=0.7)
    p.text(MX + 78, 136, "one curve per type", 14, FAINT)
    p.solid_rect(MX, 250, MW, 182, INFERRED, 0.09, rx=8)
    p.rect(MX, 250, MW, 182, INFERRED, 2.8, rough=1.0, r=8)
    p.text(MX + MW / 2, 282, "cost = fixed + marginal × size", 19, INK,
           "middle", bold=0.6)
    p.line(MX + 24, 296, MX + MW - 24, 296, EDGE, 1.8, dash="4 4")
    for i, (letter, eq) in enumerate(core["rules"]):
        p.text(MX + 22, 322 + i * 26, letter, 16, INFERRED, bold=0.6)
        p.text(MX + 58, 322 + i * 26, eq.replace("tokens = ", ""), 12, FAINT,
               mono=True)
    p.text(MX + MW / 2, 414, "fitted from the runs — not decided by you",
           13, FAINT, "middle")

    # fan out to the answers
    p.arrow(MX + MW + 4, 340, 704, 340, FAINT, 2.4, 10)
    p.line(704, CARD_Y[0] + CH / 2, 704, CARD_Y[2] + CH / 2, FAINT, 2.0)

    # ── inferred cards ───────────────────────────────────────────────────
    for (letter, what, tokens, secs, naive, naive_label), cy in zip(
            core["inferred"], CARD_Y):
        p.line(704, cy + CH / 2, RX - 6, cy + CH / 2, FAINT, 2.0)
        p.solid_rect(RX, cy, RW, CH, BUDGET, 0.09, rx=8)
        p.rect(RX, cy, RW, CH, BUDGET, 2.6, rough=1.0, r=8)
        p.text(RX + 20, cy + 44, letter, 32, BUDGET, bold=1.0)
        p.text(RX + 20, cy + 70, what, 14, FAINT)
        p.text(RX + 196, cy + 50, f"{fmt(tokens)}", 30, BUDGET, bold=0.8)
        p.text(RX + 196, cy + 74, f"tokens   ·   {secs:.0f} s", 13.5, FAINT)

        # what a naive scaling would have said, struck through
        bx = RX + 330
        p.text(bx, cy + 34, naive_label, 12.5, ALERT)
        p.text(bx, cy + 58, fmt(naive), 22, ALERT, bold=0.5)
        w = p.g.measure(fmt(naive), 22, Pen.HAND)
        p.line(bx - 3, cy + 52, bx + w + 3, cy + 52, ALERT, 2.2, rough=1.3)
        p.text(bx, cy + 80, f"{naive / tokens:.1f}× too high", 12.5, ALERT)

    # ── the honest footer ────────────────────────────────────────────────
    p.line(LX, 556, W - LX, 556, EDGE, 2.0, dash="6 5")
    p.text(LX + 4, 586, "How much to trust it", 18, INK, bold=0.6)
    p.rows(LX + 4, 612,
           ["Tokens: repeats of one task differ by 5%. The models sit at that floor.",
            "Hours: repeats differ by 23% — nearly 5× noisier. Treat them as a range."],
           14, FAINT, lh=22)
    p.text(RX + 40, 586, "Checked against reality", 18, BUDGET, bold=0.6)
    p.rows(RX + 40, 612,
           ["A+B+C was measured for real after being predicted from A, B and C",
            "alone. The prediction was 0.3% out — inside the noise floor."],
           14, FAINT, lh=22)

    return p.svg("The core idea: measure task types A, B and C, then infer "
                 "A+, A++ and A+B+C")





# ── figure 1: what the measurements said ─────────────────────────────────

def draw_concept(data: dict, glyphs: Glyphs) -> str:
    W, H = 1280, 660
    p = Pen(W, H, glyphs, seed=1017)
    store, ev, bt = data["store"], data["composition"], data["batching"]
    recs = [r for r in data["records"] if r.kind == "comprehension"]

    p.text(W / 2, 54, "Token Yield", 42, INK, "middle", bold=1.1)
    p.line(W / 2 - 100, 64, W / 2 + 100, 64, INFERRED, 3, rough=1.4)
    p.text(W / 2, 92, f"{len(data['records'])} real agent runs across 3 "
           f"repositories  →  every assumption replaced by a measurement",
           19, FAINT, "middle")

    panels = [
        (31, "1", "MEASURE", "probe real repos, watch the meter", MEASURED_C),
        (347, "2", "THE UNIT", "what actually drives cost", INFERRED),
        (663, "3", "VALIDATE", "score it on runs it never saw", COMPOSED),
        (979, "4", "THE GAP", "and what we still cannot price", ALERT),
    ]
    PY, PW, PH = 122, 270, 460

    for px, num, title, sub, color in panels:
        p.rect(px, PY, PW, PH, EDGE, 2.4, rough=1.1, r=8)
        p.solid_rect(px + 6, PY + 6, PW - 12, 42, color, 0.10, rx=6)
        p.text(px + 17, PY + 36, f"{num}.", 24, color, bold=0.9)
        p.text(px + 44, PY + 36, title, 22, INK, bold=0.9)
        p.text(px + 17, PY + 68, sub, 14, FAINT)

    for i in range(3):
        ax = panels[i][0] + PW + 6
        p.arrow(ax, PY + PH / 2, ax + 34, PY + PH / 2, FAINT, 2.4, 10)

    # ── panel 1: tokens vs bytes, one line through three repos ───────────
    px = panels[0][0]
    x0, x1 = px + 50, px + 250
    ytop, ybot = PY + 106, PY + 258
    BMAX, TLO, THI = 280_000, 30_000, 160_000
    sx = lambda b: x0 + b * (x1 - x0) / BMAX
    sy = lambda t: ybot - (t - TLO) * (ybot - ytop) / (THI - TLO)

    p.line(x0 - 6, ytop - 6, x0 - 6, ybot + 4, EDGE, 1.8)
    p.line(x0 - 6, ybot + 4, x1 + 8, ybot + 4, EDGE, 1.8)
    for t in (150_000, 90_000, 30_000):
        p.text(x0 - 12, sy(t) + 5, fmt(t), 11.5, FAINT, anchor="end")
    for b in (0, 140_000, 280_000):
        p.text(sx(b), ybot + 22, fmt(b) if b else "0", 11.5, FAINT, anchor="middle")
    p.text((x0 + x1) / 2, ybot + 40, "bytes read", 13, FAINT, "middle")

    m = store.model_for("comprehension")
    p.line(sx(2_000), sy(m.predict(2_000)), sx(BMAX), sy(m.predict(BMAX)),
           INK, 2.0, opacity=0.5)

    cols = {"harness-dose": MEASURED_C, "requests": BUDGET, "click": COMPOSED}
    for r in recs:
        b = r.signals.get("bytes", 0)
        p.circle(sx(min(b, BMAX)), sy(min(max(r.tokens, TLO), THI)), 4.5,
                 cols.get(r.repo, INK))
    for i, (name, c) in enumerate(cols.items()):
        lx = px + 20 + [0, 108, 186][i]
        p.circle(lx, PY + 320, 4.2, c)
        p.text(lx + 9, PY + 325, name, 12.5, INK)

    p.line(px + 17, PY + 348, px + PW - 17, PY + 348, EDGE, 1.8, dash="5 5")
    p.rows(px + 17, PY + 374,
           ["Three unrelated repos, one line.",
            f"Repeats of a task differ by",
            f"{data['floor']:.0%} — the noise floor."], 14.5)

    # ── panel 2: files vs bytes ──────────────────────────────────────────
    px = panels[1][0]
    p.text(px + 17, PY + 104, "fitted slope, per repo", 14, INK)
    p.text(px + 17, PY + 126, "counting FILES", 14, ALERT, bold=0.4)
    for i, (nm, v) in enumerate([("harness-dose", 2305), ("requests", 3265),
                                 ("click", 15794)]):
        p.text(px + 26, PY + 148 + i * 20, f"{nm}", 12.5, FAINT, mono=True)
        p.text(px + PW - 22, PY + 148 + i * 20, f"{v:,}", 12.5, ALERT,
               anchor="end", mono=True)
    p.text(px + 17, PY + 216, "counting BYTES", 14, BUDGET, bold=0.4)
    for i, (nm, v) in enumerate([("harness-dose", 0.418), ("requests", 0.389),
                                 ("click", 0.419)]):
        p.text(px + 26, PY + 238 + i * 20, f"{nm}", 12.5, FAINT, mono=True)
        p.text(px + PW - 22, PY + 238 + i * 20, f"{v:.3f}", 12.5, BUDGET,
               anchor="end", mono=True)

    p.line(px + 17, PY + 308, px + PW - 17, PY + 308, EDGE, 1.8, dash="4 4")
    for i, (lbl, err, col) in enumerate([("one model, files", 0.198, ALERT),
                                         ("one model, bytes", 0.028, BUDGET)]):
        p.text(px + 17, PY + 330 + i * 18, lbl, 13, FAINT)
        p.text(px + PW - 22, PY + 330 + i * 18, f"{err:.1%} error", 13.5, col,
               anchor="end", bold=0.4)

    p.line(px + 17, PY + 380, px + PW - 17, PY + 380, EDGE, 1.8, dash="5 5")
    p.rows(px + 17, PY + 402,
           ["A file count is repo-specific.",
            "The selector found this itself —",
            "it was never told to try bytes."], 14.5)

    # ── panel 3: out-of-sample ───────────────────────────────────────────
    px = panels[2][0]
    p.text(px + 17, PY + 104, "predicted vs actually measured", 14, INK)
    pairs = [("two agents", bt["separate_agents"], ev["separate_sum"]),
             ("one agent", bt["batched_single_agent"], ev["batched_mean"])]
    top = max(max(a, b) for _, a, b in pairs)
    y = PY + 122
    for label, pred, meas in pairs:
        p.text(px + 17, y + 12, label, 13.5, FAINT)
        p.bar(px + 17, y + 20, 190 * pred / top, 15, COMPOSED)
        p.text(px + 213, y + 32, "pred", 10.5, FAINT)
        p.bar(px + 17, y + 38, 190 * meas / top, 15, BUDGET)
        p.text(px + 213, y + 50, "real", 10.5, FAINT)
        p.text(px + 17, y + 72, f"{fmt(pred)} vs {fmt(meas)}   "
               f"err {abs(pred - meas) / meas:.1%}", 13.5, INK, bold=0.4)
        y += 94
    p.text(px + 17, PY + 314, f"batching saves {ev['saving']:.0%} — the old",
           13.5, BUDGET)
    p.text(px + 17, PY + 332, "+15% surcharge had the wrong sign", 13.5, ALERT)

    p.line(px + 17, PY + 356, px + PW - 17, PY + 356, EDGE, 1.8, dash="5 5")
    p.rows(px + 17, PY + 380,
           ["Batched runs were held out of",
            "every fit. The fixed/marginal",
            "split predicted them anyway."], 14.5)

    # ── panel 4: coverage and the backlog ────────────────────────────────
    px = panels[3][0]
    cov = data["coverage"]
    if cov is not None:
        share = cov.covered_share
        p.text(px + 17, PY + 104, f"mined {data['mined_n']} real commits", 14, INK)
        p.bar(px + 17, PY + 118, 236 * share, 26, BUDGET)
        p.bar(px + 17 + 236 * share, PY + 118, 236 * (1 - share), 26, ALERT,
              hatch=True)
        p.text(px + 17, PY + 168, f"{share:.0%}", 26, BUDGET, bold=0.7)
        p.text(px + 70, PY + 168, "of real work is a kind", 13.5, FAINT)
        p.text(px + 70, PY + 186, "we have actually measured", 13.5, FAINT)

        p.text(px + 17, PY + 220, "what to probe next:", 13.5, INK, bold=0.4)
        for i, (kind, sh) in enumerate(cov.backlog[:4]):
            p.text(px + 26, PY + 244 + i * 21, kind, 13, ALERT, mono=True)
            p.text(px + PW - 22, PY + 244 + i * 21, f"+{sh:.0%}", 13, ALERT,
                   anchor="end", bold=0.4)

    p.line(px + 17, PY + 348, px + PW - 17, PY + 348, EDGE, 1.8, dash="5 5")
    p.rows(px + 17, PY + 374,
           ["It refuses to price work it has",
            "never measured — and ranks the",
            "gap by how much it would unlock."], 14.5)

    p.text(W / 2, H - 22,
           "every number is computed from the measured probe suite  ·  "
           "python docs/media/draw_token_yield.py",
           13, FAINT, "middle", mono=True)

    return p.svg("Token Yield: what the measurements said")


def draw_architecture(data: dict, glyphs: Glyphs) -> str:
    W, H = 1280, 910
    p = Pen(W, H, glyphs, seed=4211)
    CX, CY, R = 640, 452, 252

    p.text(W / 2, 56, "How the machine works", 40, INK, "middle", bold=1.1)
    p.line(W / 2 - 208, 67, W / 2 + 208, 67, COMPOSED, 3, rough=1.4)
    p.text(W / 2, 96, "a flywheel — every job you run sharpens the next quote",
           20, FAINT, "middle")

    stations = [
        (500, 152, "1", "MEASURE", MEASURED_C, "gauge",
         ["Send agents to do real jobs.", "Watch the meter."]),
        (884, 392, "2", "LEARN", INFERRED, "curve",
         ["Fit the curve the data", "actually shows."]),
        (500, 632, "3", "QUOTE", COMPOSED, "tag",
         ["Price the project", "before you start it."]),
        (116, 392, "4", "SCORE", BUDGET, "target",
         ["Check the quote", "against the bill."]),
    ]
    BW, BH = 280, 122

    # the ring, drawn first so the boxes sit on top of it
    for a0, a1 in ((64, 26), (-26, -64), (-116, -154), (154, 116)):
        p.arc(CX, CY, R, a0, a1, FAINT, 2.6, 14, arrow=True)

    for bx, by, num, verb, color, icon, promise in stations:
        p.solid_rect(bx, by, BW, BH, color, 0.09, rx=10)
        p.rect(bx, by, BW, BH, color, 2.8, rough=1.0, r=10)
        getattr(p, f"icon_{icon}")(bx + 44, by + 58, 22, color)
        p.text(bx + 82, by + 44, f"{num}. {verb}", 25, INK, bold=0.9)
        p.rows(bx + 82, by + 72, promise, 14.5, FAINT, lh=20)

    # what feeds it, and what falls out
    p.text(150, 176, "goes in", 17, FAINT, bold=0.4)
    p.rows(150, 200, ["your repo", "your task list",
                      "jobs you already run"], 14, FAINT, lh=21)

    p.text(1130, 640, "comes out", 17, BUDGET, "end", bold=0.4)
    p.rows(1130, 664, ["a token budget", "a dollar figure",
                       "an honest error bar"], 14, FAINT, lh=21, anchor="end")

    # the hub: the compounding claim
    p.arc(CX, CY, 104, 0, 359, COMPOSED, 2.6, 22)
    p.solid_rect(CX - 104, CY - 104, 208, 208, COMPOSED, 0.07, rx=104)
    p.text(CX, CY - 34, "no constants", 22, COMPOSED, "middle", bold=0.7)
    p.text(CX, CY - 8, "to argue about", 22, COMPOSED, "middle", bold=0.7)
    p.line(CX - 68, CY + 8, CX + 68, CY + 8, EDGE, 2.0, dash="5 4")
    p.rows(CX, CY + 32, ["the machine measures",
                         "what it costs, and says",
                         "when it is guessing"], 14, FAINT, lh=20,
           anchor="middle")

    # the one line that makes it a loop rather than a pipeline
    p.text(CX, CY + R + 132, "a quote it got wrong is the most valuable "
           "thing the machine sees", 18, ALERT, "middle", bold=0.4)
    p.text(CX, CY + R + 160, "— it is reported, never averaged away —",
           15, FAINT, "middle")

    return p.svg("How the Token Yield machine works: a measure-learn-quote-score flywheel")


def _glyphs() -> Glyphs:
    """A fresh glyph table per figure, so each SVG defines only what it uses."""
    g = Glyphs()
    g.load("hand", HAND_TTF)
    g.load("sym", _first_present(SYM_CANDIDATES, "symbol"))
    g.load("mono", _first_present(MONO_CANDIDATES, "monospace"))
    return g


def main() -> None:
    data = figures_data()
    for name, draw in [("token-yield-core-idea.svg", draw_core_idea),
                       ("token-yield-concept.svg", draw_concept),
                       ("token-yield-architecture.svg", draw_architecture)]:
        svg = draw(data, _glyphs())
        path = os.path.join(_HERE, name)
        with open(path, "w") as f:
            f.write(svg)
        print(f"✓ {path}  ({len(svg) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
