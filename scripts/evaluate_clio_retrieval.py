"""Evaluate Clio Top-K selections using official camera frustum coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.clio_retrieval_evaluation import (
    build_clio_retrieval_evaluation,
    validate_clio_retrieval_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, default=Path("configs/clio_apartment_queries.json"))
    parser.add_argument("--task-yaml", type=Path, required=True)
    parser.add_argument("--scene-transform", type=Path, default=Path("configs/clio_scene_transforms.json"))
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    payload = build_clio_retrieval_evaluation(
        project_root=root,
        query_manifest_path=args.query_manifest,
        task_yaml_path=args.task_yaml,
        scene_transform_path=args.scene_transform,
        cameras_path=args.cameras,
        images_path=args.images,
        geometry_manifest_path=args.geometry_manifest,
        retrieval_root=args.retrieval_root,
    )
    report = validate_clio_retrieval_evaluation(payload, project_root=root)
    if report["status"] != "PASS":
        raise RuntimeError(json.dumps(report, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "aggregates": payload["aggregates"]}, indent=2))


if __name__ == "__main__":
    main()
