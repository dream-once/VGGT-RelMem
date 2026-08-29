"""Validate the portable public GPU/D15/D15.5 evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.public_evidence import validate_public_bundle, write_json


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundle",
        nargs="?",
        default=str(root / "evidence/week3/d15-gpu-public"),
    )
    parser.add_argument("--report")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_public_bundle(args.bundle)
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
