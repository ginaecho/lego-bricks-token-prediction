# Vendored font

`PatrickHand-Regular.ttf` — **Patrick Hand** by Patrick Wagesreiter, released
under the [SIL Open Font License 1.1](https://openfontlicense.org/), which
permits redistribution. Canonical source:
<https://fonts.google.com/specimen/Patrick+Hand>.

It is vendored here so that
[`docs/media/draw_token_yield.py`](../draw_token_yield.py) can regenerate the
hand-drawn README figures reproducibly, on any machine, without a network
fetch.

The generator converts every glyph to an SVG `<path>`, so the **committed SVGs
carry no font dependency at all** — they render identically whether or not the
viewer has this font installed. Symbols outside Patrick Hand's coverage (`σ`,
`Σ`, `→`) fall back to DejaVu Sans, and code identifiers are set in DejaVu Sans
Mono; both ship with essentially every Linux distribution and are located at
generation time, never redistributed here.
