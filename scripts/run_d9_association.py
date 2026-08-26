"""Run label-free D9 spatial association on a frozen D8 memory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping

from relground.association import ObjectMemory
from relground.d9_association import (
    D9_BASELINE_ID,
    D9_COUNT_FIELDS,
    D9_PREDICTION_ACCEPTANCE_FIELDS,
    D9_PREDICTION_ARTIFACT_FIELDS,
    D9_PREDICTION_RESULT_FIELDS,
    D9_PREDICTION_SCHEMA_VERSION,
    D9_PREDICTION_SOURCE_FIELDS,
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_output_dir(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def memory_observation_ids(
    memory: ObjectMemory,
) -> tuple[set[str], set[str]]:
    pending_ids = set(memory.pending_observations)
    associated_ids = {
        observation.obs_id
        for item in memory.objects.values()
        for observation in item.observations
    }
    return pending_ids, associated_ids


def prediction_metadata(
    *,
    source_reference: str,
    source_hash: str,
) -> dict[str, Any]:
    return {
        "association_source_stage": "D8",
        "association_source_memory": source_reference,
        "association_source_memory_sha256": source_hash,
        "association_stage": "D9-prediction",
    }


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    input_memory_path = Path(args.memory).resolve()
    output_dir = Path(args.output_dir).resolve()
    prepare_output_dir(output_dir)

    source_memory = ObjectMemory.load(input_memory_path)
    if source_memory.objects or source_memory.decisions:
        raise ValueError("D9 prediction requires a pristine D8 memory")
    scene_id = source_memory.metadata.get("scene_id")
    query = source_memory.metadata.get("query")
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError("D8 memory metadata.scene_id is required")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("D8 memory metadata.query is required")

    source_memory_path = output_dir / "source_memory.json"
    source_memory.save(source_memory_path)
    source_hash = sha256_file(source_memory_path)
    source_reference = source_memory_path.name

    config = SpatialGateConfig(
        center_distance_threshold=args.center_distance_threshold,
        min_overlap_iou=args.min_overlap_iou,
        min_distinct_frames=args.min_distinct_frames,
    )
    metadata_update = prediction_metadata(
        source_reference=source_reference,
        source_hash=source_hash,
    )

    memory = ObjectMemory.load(source_memory_path)
    memory.metadata.update(metadata_update)
    outcome = associate_pending(memory, config)
    memory_path = output_dir / "object_memory.json"
    memory.save(memory_path)
    restored = ObjectMemory.load(memory_path)
    round_trip_equal = memory.to_dict() == restored.to_dict()

    replay = ObjectMemory.load(source_memory_path)
    replay.metadata.update(metadata_update)
    replay_outcome = associate_pending(replay, config)
    deterministic_recompute = (
        outcome == replay_outcome
        and memory.to_dict() == replay.to_dict()
    )

    source_ids = set(source_memory.pending_observations)
    pending_ids, associated_ids = memory_observation_ids(restored)
    observation_conservation = (
        not (pending_ids & associated_ids)
        and pending_ids | associated_ids == source_ids
        and len(pending_ids) + len(associated_ids) == len(source_ids)
    )
    cross_frame_object_pass = all(
        len(set(item.evidence_frames)) >= config.min_distinct_frames
        for item in restored.objects.values()
    )
    acceptance = {
        "observation_conservation": observation_conservation,
        "deterministic_recompute": deterministic_recompute,
        "cross_frame_object_pass": cross_frame_object_pass,
        "round_trip_equal": round_trip_equal,
    }
    if tuple(acceptance) != D9_PREDICTION_ACCEPTANCE_FIELDS:
        raise AssertionError("D9 prediction acceptance fields changed")
    status = "PASS" if all(acceptance.values()) else "FAIL"

    components = outcome["components"]
    counts = {
        "input_observations": len(source_ids),
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
    }
    if tuple(source) != D9_PREDICTION_SOURCE_FIELDS:
        raise AssertionError("D9 prediction source fields changed")
    artifacts = {
        "source_memory": source_memory_path.name,
        "object_memory": memory_path.name,
    }
    if tuple(artifacts) != D9_PREDICTION_ARTIFACT_FIELDS:
        raise AssertionError("D9 prediction artifact fields changed")

    result = {
        "schema_version": D9_PREDICTION_SCHEMA_VERSION,
        "status": status,
        "stage": "D9-prediction",
        "baseline_id": D9_BASELINE_ID,
        "scene_id": scene_id,
        "query": query,
        "source": source,
        "gate_config": config.to_dict(),
        "counts": counts,
        "components": components,
        "pairs": outcome["pairs"],
        "acceptance": acceptance,
        "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if tuple(result) != D9_PREDICTION_RESULT_FIELDS:
        raise AssertionError("D9 prediction result fields changed")
    write_json(output_dir / "d9_result.json", result)

    RunManifest(
        git_sha=git_commit(project_root),
        env_lock="D9 prediction is deterministic and model-free",
        dataset_split=scene_id,
        seed=0,
        config={
            "stage": "D9-prediction",
            "pipeline": D9_BASELINE_ID,
            "source_memory": source_reference,
            "source_memory_sha256": source_hash,
            "gate_config": config.to_dict(),
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
        "--output-dir",
        default="runs/office-loop-mv-d9-trash-can/prediction",
    )
    parser.add_argument(
        "--center-distance-threshold",
        type=float,
        default=0.15,
        help="Uncalibrated reconstruction units, not metres.",
    )
    parser.add_argument("--min-overlap-iou", type=float, default=0.0)
    parser.add_argument("--min-distinct-frames", type=int, default=2)
    parser.add_argument("--project-root", default=str(root))
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
