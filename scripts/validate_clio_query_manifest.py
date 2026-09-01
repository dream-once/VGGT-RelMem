"""Validate a tracked Clio query manifest against an acquired task YAML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.clio_query_manifest import load_manifest, validate_official_task_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=Path("configs/clio_apartment_queries.json"),
    )
    parser.add_argument("--task-yaml", type=Path, required=True)
    args = parser.parse_args()
    report = validate_official_task_source(
        load_manifest(args.query_manifest), args.task_yaml
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
