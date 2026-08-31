"""Validate and deterministically replay a Clio association benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.clio_association_benchmark import validate_clio_association_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    report = validate_clio_association_benchmark(payload, project_root=Path(args.project_root))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
