"""Offline composition research from one completed, frozen marketplace run.

Example:
    python -m examples.marketplace_models --source-run runs\\completed --audit-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from token_yield.marketplace_models import audit_run, run_offline


EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2
EXIT_INTERRUPTED = 130
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the mutually exclusive offline preview/fit argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True,
                        help="Completed measurement source; never altered.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run-dir", type=Path,
                        help="Fit into a new direct child of repository runs; existing destinations are rejected.")
    action.add_argument("--audit-only", action="store_true",
                        help="Read-only preview: validate completed evidence without fitting or writing files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Audit or fit completed evidence, reporting refusals without paid execution.

    Args:
        argv: Explicit arguments for tests, or the process arguments when omitted.

    Returns:
        Zero on success, 130 on interruption, or one on a broken output pipe.

    Raises:
        SystemExit: Exit code two for invalid arguments or refused source/destination.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = create_parser()
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        if args.audit_only:
            audited = audit_run(root, args.source_run)
            result = {
                "status": "audited_research_only", "workflow_count": len(audited["workflows"]),
                "campaign_sha256": audited["provenance"]["campaign_sha256"],
                "fitted": False, "files_written": 0,
            }
        else:
            report = run_offline(root, args.source_run, args.run_dir)
            result = {
                "status": report["status"], "selected": report["selected"],
                "production_recommendation": None, "calibrated_intervals": None,
                "output": str(args.run_dir),
            }
        print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        return EXIT_INTERRUPTED
    except BrokenPipeError:
        return EXIT_FAILURE
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(EXIT_ERROR, f"Offline research refused: {exc}\n")
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
