"""Validate a real Clio relation benchmark bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.clio_relation_benchmark import validate_relation_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    report = validate_relation_bundle(Path(args.bundle), project_root=Path(args.project_root))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
