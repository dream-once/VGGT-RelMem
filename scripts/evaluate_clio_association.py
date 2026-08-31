"""Evaluate frozen A1/A2 association predictions against Clio task OBBs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.clio_association_benchmark import build_clio_association_benchmark


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_clio_association_benchmark(
        project_root=Path(args.project_root),
        query_manifest_path=Path(args.query_manifest),
        task_yaml_path=Path(args.task_yaml),
        world_alignment_path=Path(args.world_alignment),
        run_root=Path(args.run_root),
        created_at=args.created_at,
    )
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--task-yaml", required=True)
    parser.add_argument("--world-alignment", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--created-at")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps({"status": result["status"], "counts": result["counts"], "metrics": result["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
