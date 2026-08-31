"""Build and run a self-contained real Clio relation benchmark bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from relground.clio_relation_benchmark import (
    _relative,
    _sha256,
    build_relation_queries_and_labels,
    build_scene_object_memory,
    evaluate_clio_relation_prediction,
)
from relground.relation_protocol import run_relation_prediction


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    root = Path(args.project_root).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    query_manifest_path = Path(args.query_manifest).resolve()
    task_yaml_path = Path(args.task_yaml).resolve()
    alignment_path = Path(args.world_alignment).resolve()
    anchor_path = Path(args.geometry_anchor_poses).resolve()
    protocol_path = Path(args.protocol).resolve()
    calibration_path = Path(args.calibration).resolve()
    run_root = Path(args.run_root).resolve()
    created_at = args.created_at or datetime.now(timezone.utc).isoformat()

    memory, memory_sources = build_scene_object_memory(
        project_root=root,
        query_manifest_path=query_manifest_path,
        run_root=run_root,
    )
    memory_path = output / "scene_object_memory.json"
    memory.save(memory_path)
    all_anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor_key = next(iter(all_anchors))
    anchors = {anchor_key: all_anchors[anchor_key]}
    _write(output / "anchor_poses.json", anchors)
    query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    queries, labels, generation = build_relation_queries_and_labels(
        memory=memory,
        query_manifest=query_manifest,
        task_yaml_path=task_yaml_path,
        world_alignment=alignment,
        anchor_poses=anchors,
        protocol=protocol,
    )
    _write(output / "queries.json", queries)
    _write(output / "labels.json", labels)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    _write(output / "calibration_manifest.json", calibration)
    prediction_source = {
        "object_memory": "scene_object_memory.json",
        "object_memory_sha256": _sha256(memory_path),
        "anchor_poses": "anchor_poses.json",
        "anchor_poses_sha256": _sha256(output / "anchor_poses.json"),
        "queries": "queries.json",
        "queries_sha256": _sha256(output / "queries.json"),
        "calibration": "calibration_manifest.json",
        "calibration_sha256": _sha256(output / "calibration_manifest.json"),
    }
    prediction = run_relation_prediction(
        memory, anchors, queries, calibration,
        source=prediction_source,
        created_at=created_at,
    )
    _write(output / "prediction.json", prediction)
    sources = {
        "query_manifest": _relative(root, query_manifest_path),
        "query_manifest_sha256": _sha256(query_manifest_path),
        "task_gt": _relative(root, task_yaml_path),
        "task_gt_sha256": _sha256(task_yaml_path),
        "world_alignment": _relative(root, alignment_path),
        "world_alignment_sha256": _sha256(alignment_path),
        "geometry_anchor_poses": _relative(root, anchor_path),
        "geometry_anchor_poses_sha256": _sha256(anchor_path),
        "protocol": _relative(root, protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "calibration_source": _relative(root, calibration_path),
        "calibration_source_sha256": _sha256(calibration_path),
        "run_root": _relative(root, run_root),
    }
    _write(output / "bundle_sources.json", sources)
    evaluation_source = {
        **sources,
        "scene_memory_sources": memory_sources,
        "scene_object_memory_sha256": _sha256(memory_path),
        "queries_sha256": _sha256(output / "queries.json"),
        "labels_sha256": _sha256(output / "labels.json"),
        "prediction_sha256": _sha256(output / "prediction.json"),
    }
    evaluation = evaluate_clio_relation_prediction(
        prediction, labels,
        source=evaluation_source,
        generation=generation,
        created_at=created_at,
    )
    _write(output / "evaluation.json", evaluation)
    return evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--task-yaml", required=True)
    parser.add_argument("--world-alignment", required=True)
    parser.add_argument("--geometry-anchor-poses", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--protocol", default="configs/clio_relation_confirmatory_protocol.json")
    parser.add_argument("--calibration", default="configs/relation_calibration_manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--created-at")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps({"status": result["status"], "generation": result["generation"], "metrics": result["metrics"], "abstain_reasons": result["abstain_reasons"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
