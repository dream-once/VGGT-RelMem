"""Build the Q0-versus-Q1 Apartment development Grounding benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.clio_grounding_benchmark import (
    build_clio_grounding_benchmark,
    validate_clio_grounding_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, default=Path("configs/clio_apartment_queries.json"))
    parser.add_argument("--task-yaml", type=Path, required=True)
    parser.add_argument("--world-alignment", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--frozen-policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    payload = build_clio_grounding_benchmark(
        project_root=root,
        query_manifest_path=args.query_manifest,
        task_yaml_path=args.task_yaml,
        world_alignment_path=args.world_alignment,
        run_root=args.run_root,
        frozen_policy_path=args.frozen_policy,
    )
    report = validate_clio_grounding_benchmark(payload, project_root=root)
    if report["status"] != "PASS":
        raise RuntimeError(json.dumps(report, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "metrics": payload["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
