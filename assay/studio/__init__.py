"""The Token Yield Studio: the whole argument, running, in a browser.

Serving a UI from a project like this carries an obvious risk. A dashboard makes numbers
look settled, and most of these numbers are not: everything the demo computes comes from
`MockAdapter`, which is a simulator. So the evidence class travels with every payload the
server emits, and the interface is built to repeat it rather than to let it fade into the
background.

What the demo is actually evidence *for* is narrower than "bricks predict cost", and more
interesting: run the identical pipeline twice, once against a runtime that prices bricks
and once against one that does not, and watch it find the signal in the first case and
refuse to invent one in the second. That is a claim about the method, and the simulator
can support it honestly.
"""

from __future__ import annotations

__all__ = ["DemoState", "run_demo", "serve"]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    if name in ("DemoState", "run_demo"):
        from assay.studio import demo

        return getattr(demo, name)
    if name == "serve":
        from assay.studio.server import serve

        return serve
    raise AttributeError(name)
