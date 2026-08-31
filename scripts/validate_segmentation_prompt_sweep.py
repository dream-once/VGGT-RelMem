"""Validate a prepared D21.1 SAM prompt-sweep plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.segmentation_sweep import read_json, validate_sweep_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument(
        "--project-root", default=str(Path(__file__).resolve().parents[1])
    )
    args = parser.parse_args()
    report = validate_sweep_plan(
        read_json(Path(args.plan)), project_root=Path(args.project_root)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
