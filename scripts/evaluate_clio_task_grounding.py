"""Evaluate a frozen ObjectMemory against one official Clio task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.clio_task_evaluation import (
    build_clio_task_evaluation,
    validate_clio_task_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--world-alignment", required=True, type=Path)
    parser.add_argument("--task-yaml", required=True, type=Path)
    parser.add_argument("--task-query", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path.cwd().resolve()
    payload = build_clio_task_evaluation(
        project_root=root,
        object_memory_path=args.memory,
        world_alignment_path=args.world_alignment,
        task_yaml_path=args.task_yaml,
        task_query=args.task_query,
    )
    report = validate_clio_task_evaluation(payload, project_root=root)
    if report["status"] != "PASS":
        raise RuntimeError(json.dumps(report, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "output": args.output.as_posix(),
        "counts": payload["counts"],
        "metrics": payload["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
