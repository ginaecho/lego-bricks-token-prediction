"""Collection rules for the whole suite.

`openharness` and `token_yield` still support Python 3.9. `assay/` does not: it reads the
brick vocabulary with `tomllib`, which arrived in 3.11. Rather than drop 3.9 for the
published package, the assay suite is simply not collected on older interpreters — so CI
keeps testing the parts that claim 3.9 support, on 3.9.
"""

from __future__ import annotations

import sys

collect_ignore = [] if sys.version_info >= (3, 11) else ["assay"]
