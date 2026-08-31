"""Validate the tracked lightweight Clio final summary without raw data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.clio_final_summary import validate_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", default="evidence/final-clio/benchmark_summary.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    report = validate_summary(payload)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
