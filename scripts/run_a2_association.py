"""Run D12 label-free A2 complete-link association on frozen D8 memory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import time

from relground.a2_association import (
    A2_ACCEPTANCE_FIELDS,
    A2_ASSOCIATION_ID,
    A2_COUNT_FIELDS,
    A2_PREDICTION_ARTIFACT_FIELDS,
    A2_PREDICTION_RESULT_FIELDS,
    A2_PREDICTION_SOURCE_FIELDS,
    A2_SCHEMA_VERSION,
    EvidenceAssociationConfig,
    associate_pending_a2,
)
from relground.association import ObjectMemory
from relground.observation_cache import sha256_file
from relground.schemas import RunManifest
from scripts.run_d9_association import (
    git_commit,
    memory_observation_ids,
    prepare_output_dir,
    write_json,
)


def prediction_metadata(
    *,
    source_reference: str,
    source_hash: str,
) -> dict[str, str]:
    return {
        "association_source_stage": "D8",
        "association_source_memory": source_reference,
        "association_source_memory_sha256": source_hash,
        "association_stage": "D12-A2-prediction",
        "association_id": A2_ASSOCIATION_ID,
    }


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    input_memory_path = Path(args.memory).resolve()
    output_dir = Path(args.output_dir).resolve()
    prepare_output_dir(output_dir)

    source_memory = ObjectMemory.load(input_memory_path)
    if source_memory.objects or source_memory.decisions:
        raise ValueError("A2 prediction requires pristine D8 memory")
    scene_id = source_memory.metadata.get("scene_id")
    query = source_memory.metadata.get("query")
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError("D8 metadata.scene_id is required")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("D8 metadata.query is required")

    source_path = output_dir / "source_memory.json"
    source_memory.save(source_path)
    source_hash = sha256_file(source_path)
    metadata = prediction_metadata(
        source_reference=source_path.name,
        source_hash=source_hash,
    )
    config = EvidenceAssociationConfig(
        semantic_threshold=args.semantic_threshold,
        min_observation_quality=args.min_observation_quality,
        center_distance_threshold=args.center_distance_threshold,
        min_overlap_iou=args.min_overlap_iou,
        min_distinct_frames=args.min_distinct_frames,
    )

    memory = ObjectMemory.load(source_path)
    memory.metadata.update(metadata)
    outcome = associate_pending_a2(memory, config)
    memory_path = output_dir / "object_memory.json"
    memory.save(memory_path)
    restored = ObjectMemory.load(memory_path)
    round_trip_equal = memory.to_dict() == restored.to_dict()

    replay = ObjectMemory.load(source_path)
    replay.metadata.update(metadata)
    replay_outcome = associate_pending_a2(replay, config)
    deterministic_recompute = (
        outcome == replay_outcome
        and memory.to_dict() == replay.to_dict()
    )
    source_ids = set(source_memory.pending_observations)
    pending_ids, associated_ids = memory_observation_ids(restored)
    observation_conservation = (
        not pending_ids & associated_ids
        and pending_ids | associated_ids == source_ids
        and len(pending_ids) + len(associated_ids) == len(source_ids)
    )
    complete_link_pass = all(
        not pair["predicted_same"] or pair["gate_pass"]
        for pair in outcome["pairs"]
    )
    cross_frame_object_pass = all(
        len(set(item.evidence_frames)) >= config.min_distinct_frames
        for item in restored.objects.values()
    )
    acceptance = {
        "observation_conservation": observation_conservation,
        "deterministic_recompute": deterministic_recompute,
        "complete_link_pass": complete_link_pass,
        "cross_frame_object_pass": cross_frame_object_pass,
        "round_trip_equal": round_trip_equal,
    }
    if tuple(acceptance) != A2_ACCEPTANCE_FIELDS:
        raise AssertionError("A2 acceptance fields changed")
    status = "PASS" if all(acceptance.values()) else "FAIL"

    clusters = outcome["clusters"]
    counts = {
        "input_observations": len(source_ids),
        "pair_count": len(outcome["pairs"]),
        "gate_pass_pairs": sum(
            bool(item["gate_pass"]) for item in outcome["pairs"]
        ),
        "predicted_match_pairs": sum(
            bool(item["predicted_same"]) for item in outcome["pairs"]
        ),
        "merge_count": len(outcome["merge_decisions"]),
        "cluster_count": len(clusters),
        "promoted_clusters": sum(bool(item["promoted"]) for item in clusters),
        "deferred_clusters": sum(not bool(item["promoted"]) for item in clusters),
        "permanent_objects": len(restored.objects),
        "pending_observations": len(restored.pending_observations),
        "association_decisions": len(restored.decisions),
    }
    if tuple(counts) != A2_COUNT_FIELDS:
        raise AssertionError("A2 count fields changed")
    source = {
        "d8_memory": source_path.name,
        "d8_memory_sha256": source_hash,
    }
    artifacts = {
        "source_memory": source_path.name,
        "object_memory": memory_path.name,
    }
    if tuple(source) != A2_PREDICTION_SOURCE_FIELDS:
        raise AssertionError("A2 source fields changed")
    if tuple(artifacts) != A2_PREDICTION_ARTIFACT_FIELDS:
        raise AssertionError("A2 artifact fields changed")

    result = {
        "schema_version": A2_SCHEMA_VERSION,
        "status": status,
        "stage": "D12-A2-prediction",
        "association_id": A2_ASSOCIATION_ID,
        "scene_id": scene_id,
        "query": query,
        "source": source,
        "config": config.to_dict(),
        "counts": counts,
        "pairs": outcome["pairs"],
        "merge_decisions": outcome["merge_decisions"],
        "clusters": clusters,
        "acceptance": acceptance,
        "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if tuple(result) != A2_PREDICTION_RESULT_FIELDS:
        raise AssertionError("A2 prediction fields changed")
    write_json(output_dir / "a2_result.json", result)

    RunManifest(
        git_sha=git_commit(project_root),
        env_lock="D12 A2 prediction is deterministic and model-free",
        dataset_split=scene_id,
        seed=0,
        config={
            "stage": "D12-A2-prediction",
            "status": "CPU_COMPLETE",
            "gpu_acceptance": "PENDING",
            "association_id": A2_ASSOCIATION_ID,
            "source_memory": source_path.name,
            "source_memory_sha256": source_hash,
            "association_config": config.to_dict(),
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
            "evidence/week1/runs/office-loop-mv-d8-trash-can/"
            "object_memory.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="runs/office-loop-mv-d12-a2-trash-can/prediction",
    )
    parser.add_argument("--semantic-threshold", type=float, default=0.70)
    parser.add_argument(
        "--min-observation-quality", type=float, default=0.25
    )
    parser.add_argument(
        "--center-distance-threshold", type=float, default=0.15
    )
    parser.add_argument("--min-overlap-iou", type=float, default=0.0)
    parser.add_argument("--min-distinct-frames", type=int, default=2)
    parser.add_argument("--project-root", default=str(root))
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
