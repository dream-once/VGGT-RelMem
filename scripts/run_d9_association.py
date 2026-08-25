"""Run D9 exact-class spatial association on a frozen D8 memory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

from relground.association import ObjectMemory
from relground.d9_association import (
    D9_ACCEPTANCE_FIELDS,
    D9_ARTIFACT_FIELDS,
    D9_BASELINE_ID,
    D9_COUNT_FIELDS,
    D9_RESULT_FIELDS,
    D9_SCHEMA_VERSION,
    D9_SOURCE_FIELDS,
    ManualInstanceLabels,
    SpatialGateConfig,
    associate_pending,
)
from relground.observation_cache import sha256_file
from relground.schemas import RunManifest


def git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return (
        completed.stdout.strip()
        if completed.returncode == 0
        else "unknown"
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    input_memory_path = Path(args.memory).resolve()
    input_labels_path = Path(args.labels).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 <= args.min_pairwise_f1 <= 1.0:
        raise ValueError("min_pairwise_f1 must be in [0, 1]")

    source_hash = sha256_file(input_memory_path)
    source_reference = Path(
        os.path.relpath(input_memory_path, output_dir)
    ).as_posix()
    memory = ObjectMemory.load(input_memory_path)
    labels = ManualInstanceLabels.load(input_labels_path)
    labels_path = output_dir / "pair_labels.json"
    write_json(labels_path, labels.to_dict())
    labels_hash = sha256_file(labels_path)

    config = SpatialGateConfig(
        center_distance_threshold=args.center_distance_threshold,
        min_overlap_iou=args.min_overlap_iou,
        min_distinct_frames=args.min_distinct_frames,
    )
    memory.metadata.update({
        "association_source_stage": "D8",
        "association_source_memory": source_reference,
        "association_source_memory_sha256": source_hash,
        "association_stage": "D9",
        "association_labels": labels_path.name,
        "association_labels_sha256": labels_hash,
    })
    outcome = associate_pending(memory, labels, config)

    memory_path = output_dir / "object_memory.json"
    memory.save(memory_path)
    restored = ObjectMemory.load(memory_path)
    round_trip_equal = memory.to_dict() == restored.to_dict()
    metrics = outcome["metrics"]
    pairwise_f1_pass = metrics["f1"] >= args.min_pairwise_f1
    cross_frame_object_pass = bool(restored.objects) and all(
        len(item.evidence_frames) >= config.min_distinct_frames
        for item in restored.objects.values()
    )
    status = (
        "PASS"
        if pairwise_f1_pass
        and cross_frame_object_pass
        and round_trip_equal
        else "FAIL"
    )

    components = outcome["components"]
    counts = {
        "input_observations": (
            len(restored.pending_observations)
            + sum(
                len(item.observations)
                for item in restored.objects.values()
            )
        ),
        "pair_count": len(outcome["pairs"]),
        "predicted_match_pairs": sum(
            bool(item["predicted_same"]) for item in outcome["pairs"]
        ),
        "candidate_components": len(components),
        "promoted_components": sum(
            bool(item["promoted"]) for item in components
        ),
        "deferred_components": sum(
            not bool(item["promoted"]) for item in components
        ),
        "permanent_objects": len(restored.objects),
        "pending_observations": len(restored.pending_observations),
        "association_decisions": len(restored.decisions),
    }
    if tuple(counts) != D9_COUNT_FIELDS:
        raise AssertionError("D9 count fields changed")
    source = {
        "d8_memory": source_reference,
        "d8_memory_sha256": source_hash,
        "pair_labels": labels_path.name,
        "pair_labels_sha256": labels_hash,
    }
    if tuple(source) != D9_SOURCE_FIELDS:
        raise AssertionError("D9 source fields changed")
    acceptance = {
        "min_pairwise_f1": args.min_pairwise_f1,
        "pairwise_f1_pass": pairwise_f1_pass,
        "cross_frame_object_pass": cross_frame_object_pass,
        "round_trip_equal": round_trip_equal,
    }
    if tuple(acceptance) != D9_ACCEPTANCE_FIELDS:
        raise AssertionError("D9 acceptance fields changed")
    artifacts = {
        "object_memory": memory_path.name,
        "pair_labels": labels_path.name,
    }
    if tuple(artifacts) != D9_ARTIFACT_FIELDS:
        raise AssertionError("D9 artifact fields changed")

    result = {
        "schema_version": D9_SCHEMA_VERSION,
        "status": status,
        "stage": "D9",
        "baseline_id": D9_BASELINE_ID,
        "scene_id": labels.scene_id,
        "query": labels.query,
        "source": source,
        "gate_config": config.to_dict(),
        "counts": counts,
        "metrics": metrics,
        "components": components,
        "pairs": outcome["pairs"],
        "failure_cases": outcome["failure_cases"],
        "acceptance": acceptance,
        "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if tuple(result) != D9_RESULT_FIELDS:
        raise AssertionError("D9 result fields changed")
    write_json(output_dir / "d9_result.json", result)

    RunManifest(
        git_sha=git_commit(project_root),
        env_lock="D9 association is deterministic and model-free",
        dataset_split=labels.scene_id,
        seed=0,
        config={
            "pipeline": result["baseline_id"],
            "source_memory": source_reference,
            "source_memory_sha256": source_hash,
            "pair_labels": labels_path.name,
            "pair_labels_sha256": labels_hash,
            "gate_config": config.to_dict(),
            "min_pairwise_f1": args.min_pairwise_f1,
        },
        command=shlex.join(sys.argv),
        runtime_seconds=time.perf_counter() - started,
        peak_vram_mb=None,
    ).save(output_dir / "run_manifest.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory",
        default=(
            "runs/office-loop-mv-d8-trash-can/object_memory.json"
        ),
    )
    parser.add_argument(
        "--labels",
        default="configs/d9_office_loop_trash_can_labels.json",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/office-loop-mv-d9-trash-can",
    )
    parser.add_argument(
        "--center-distance-threshold",
        type=float,
        default=0.15,
        help="Uncalibrated reconstruction units, not metres.",
    )
    parser.add_argument("--min-overlap-iou", type=float, default=0.0)
    parser.add_argument("--min-distinct-frames", type=int, default=2)
    parser.add_argument("--min-pairwise-f1", type=float, default=0.95)
    parser.add_argument("--project-root", default=str(root))
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
