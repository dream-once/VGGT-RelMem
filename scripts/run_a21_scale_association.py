"""Run the label-free A2.1 scale-aware development candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from relground.a21_scale_association import (
    A21_ASSOCIATION_ID,
    A21_SCHEMA_VERSION,
    A21_STATUS,
    ScaleAwareAssociationConfig,
    associate_pending_a21,
)
from relground.association import ObjectMemory
from relground.observation_cache import sha256_file
from scripts.run_d9_association import memory_observation_ids, prepare_output_dir, write_json


def prediction_metadata() -> dict[str, str]:
    return {
        "association_stage": "D21.1-A2.1-prediction",
        "association_id": A21_ASSOCIATION_ID,
        "development_status": A21_STATUS,
    }


def build_result(
    *, source_memory_path: Path, output_dir: Path,
    config: ScaleAwareAssociationConfig,
) -> dict:
    source_memory = ObjectMemory.load(source_memory_path)
    if source_memory.objects or source_memory.decisions:
        raise ValueError("A2.1 requires pristine D8 memory")
    scene_id = str(source_memory.metadata.get("scene_id", ""))
    query = str(source_memory.metadata.get("query", ""))
    if not scene_id or not query:
        raise ValueError("A2.1 source memory needs scene_id and query")
    prepare_output_dir(output_dir)
    copied_source = output_dir / "source_memory.json"
    source_memory.save(copied_source)
    source_hash = sha256_file(copied_source)
    memory = ObjectMemory.load(copied_source)
    memory.metadata.update({
        "association_stage": "D21.1-A2.1-prediction",
        "association_id": A21_ASSOCIATION_ID,
        "development_status": A21_STATUS,
    })
    outcome = associate_pending_a21(memory, config)
    object_memory_path = output_dir / "object_memory.json"
    memory.save(object_memory_path)
    restored = ObjectMemory.load(object_memory_path)
    replay = ObjectMemory.load(copied_source)
    replay.metadata.update(memory.metadata)
    replay_outcome = associate_pending_a21(replay, config)
    pending, associated = memory_observation_ids(restored)
    source_ids = set(source_memory.pending_observations)
    acceptance = {
        "observation_conservation": not pending & associated and pending | associated == source_ids,
        "deterministic_recompute": outcome == replay_outcome and memory.to_dict() == replay.to_dict(),
        "complete_link_pass": all(not row["predicted_same"] or row["gate_pass"] for row in outcome["pairs"]),
        "cross_frame_object_pass": all(len(set(obj.evidence_frames)) >= config.min_distinct_frames for obj in restored.objects.values()),
        "round_trip_equal": memory.to_dict() == restored.to_dict(),
    }
    counts = {
        "input_observations": len(source_ids),
        "pair_count": len(outcome["pairs"]),
        "gate_pass_pairs": sum(bool(row["gate_pass"]) for row in outcome["pairs"]),
        "predicted_match_pairs": sum(bool(row["predicted_same"]) for row in outcome["pairs"]),
        "cluster_count": len(outcome["clusters"]),
        "promoted_clusters": sum(bool(row["promoted"]) for row in outcome["clusters"]),
        "permanent_objects": len(restored.objects),
        "pending_observations": len(restored.pending_observations),
    }
    result = {
        "schema_version": A21_SCHEMA_VERSION,
        "status": "PASS" if all(acceptance.values()) else "FAIL",
        "stage": "D21.1-A2.1-prediction",
        "association_id": A21_ASSOCIATION_ID,
        "development_status": A21_STATUS,
        "scene_id": scene_id,
        "query": query,
        "source": {"d8_memory": "source_memory.json", "d8_memory_sha256": source_hash},
        "config": config.to_dict(),
        "counts": counts,
        "pairs": outcome["pairs"],
        "merge_decisions": outcome["merge_decisions"],
        "clusters": outcome["clusters"],
        "acceptance": acceptance,
        "artifacts": {"source_memory": "source_memory.json", "object_memory": "object_memory.json"},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "a21_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-normalized-center-distance", type=float, default=1.0)
    args = parser.parse_args()
    result = build_result(
        source_memory_path=Path(args.memory).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        config=ScaleAwareAssociationConfig(
            max_normalized_center_distance=args.max_normalized_center_distance
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
