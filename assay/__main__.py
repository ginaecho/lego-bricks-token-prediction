"""``python -m assay`` -- works without installing the package."""

from __future__ import annotations

import sys

from assay.cli import main

if __name__ == "__main__":
    sys.exit(main())
