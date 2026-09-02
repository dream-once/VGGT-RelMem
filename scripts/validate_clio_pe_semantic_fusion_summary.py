"""Validate tracked post-D21 PE fusion evidence without raw Clio data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.pe_semantic_fusion_summary import validate_summary


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "summary",
        nargs="?",
        default=(
            "evidence/post-d21-pe-fusion/"
            "benchmark_summary.json"
        ),
    )
    parser.add_argument("--report")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.project_root).resolve()
    path = Path(args.summary)
    if not path.is_absolute():
        path = root / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = validate_summary(payload, project_root=root)
    if args.report:
        output = Path(args.report)
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
