"""Independently replay and validate an A2.1 prediction bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from relground.a21_scale_association import (
    A21_ASSOCIATION_ID,
    A21_SCHEMA_VERSION,
    A21_STATUS,
    ScaleAwareAssociationConfig,
    associate_pending_a21,
)
from relground.a2_association import A2_PAIR_FIELDS
from relground.association import ObjectMemory
from relground.observation_cache import sha256_file
from scripts.run_a21_scale_association import prediction_metadata
from scripts.run_d9_association import memory_observation_ids


RESULT_FIELDS = (
    "schema_version", "status", "stage", "association_id",
    "development_status", "scene_id", "query", "source", "config",
    "counts", "pairs", "merge_decisions", "clusters", "acceptance",
    "artifacts", "created_at",
)
SOURCE_FIELDS = ("d8_memory", "d8_memory_sha256")
COUNT_FIELDS = (
    "input_observations", "pair_count", "gate_pass_pairs",
    "predicted_match_pairs", "cluster_count", "promoted_clusters",
    "permanent_objects", "pending_observations",
)
ACCEPTANCE_FIELDS = (
    "observation_conservation", "deterministic_recompute",
    "complete_link_pass", "cross_frame_object_pass", "round_trip_equal",
)
ARTIFACT_FIELDS = ("source_memory", "object_memory")
PAIR_FIELDS = (
    *A2_PAIR_FIELDS,
    "center_scale", "normalized_center_distance",
    "max_normalized_center_distance", "scale_center_pass",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} root must be an object")
    return payload


def resolve_reference(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError("A2.1 bundle references must be relative")
    boundary = root.resolve()
    candidate = (boundary / relative).resolve()
    if candidate != boundary and boundary not in candidate.parents:
        raise ValueError("A2.1 bundle reference escapes its root")
    return candidate


def _evaluation_keys(payload: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            name = str(key)
            current = f"{prefix}.{name}" if prefix else name
            if "label" in name.lower() or name in {
                "expected_same", "metrics", "error_type", "ground_truth",
            }:
                found.append(current)
            found.extend(_evaluation_keys(value, current))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_evaluation_keys(value, f"{prefix}[{index}]"))
    return found


def _required_payload(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = read_json(root / "a21_result.json")
    raw_source = read_json(root / "source_memory.json")
    raw_memory = read_json(root / "object_memory.json")
    if tuple(result) != RESULT_FIELDS:
        raise ValueError("A2.1 result fields are not frozen")
    for payload, fields, name in (
        (result["source"], SOURCE_FIELDS, "source"),
        (result["counts"], COUNT_FIELDS, "counts"),
        (result["acceptance"], ACCEPTANCE_FIELDS, "acceptance"),
        (result["artifacts"], ARTIFACT_FIELDS, "artifacts"),
    ):
        if not isinstance(payload, Mapping) or tuple(payload) != fields:
            raise ValueError(f"A2.1 {name} fields are not frozen")
    if not isinstance(result["pairs"], list):
        raise ValueError("A2.1 pairs must be a list")
    for row in result["pairs"]:
        if not isinstance(row, Mapping) or tuple(row) != PAIR_FIELDS:
            raise ValueError("A2.1 pair fields are not frozen")
    return result, raw_source, raw_memory


def validate_output(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    failures: list[str] = []
    required = ("a21_result.json", "source_memory.json", "object_memory.json")
    for name in required:
        artifact = root / name
        if not artifact.is_file() or artifact.stat().st_size == 0:
            failures.append(f"missing or empty artifact: {name}")
    if failures:
        return {"status": "FAIL", "failures": failures}

    try:
        result, raw_source, raw_memory = _required_payload(root)
        config = ScaleAwareAssociationConfig.from_dict(result["config"])
        source_path = resolve_reference(
            root, str(result["source"]["d8_memory"])
        )
        artifact_source = resolve_reference(
            root, str(result["artifacts"]["source_memory"])
        )
        memory_path = resolve_reference(
            root, str(result["artifacts"]["object_memory"])
        )
        if source_path != root / "source_memory.json" or artifact_source != source_path:
            raise ValueError("A2.1 source references are inconsistent")
        if memory_path != root / "object_memory.json":
            raise ValueError("A2.1 memory reference is inconsistent")
        source_memory = ObjectMemory.load(source_path)
        memory = ObjectMemory.load(memory_path)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {"status": "FAIL", "failures": [f"invalid A2.1 bundle: {error}"]}

    if result["schema_version"] != A21_SCHEMA_VERSION:
        failures.append("unsupported A2.1 schema")
    if result["status"] != "PASS" or result["stage"] != "D21.1-A2.1-prediction":
        failures.append("A2.1 status/stage is inconsistent")
    if result["association_id"] != A21_ASSOCIATION_ID:
        failures.append("A2.1 association id changed")
    if result["development_status"] != A21_STATUS:
        failures.append("A2.1 development boundary changed")
    if source_memory.objects or source_memory.decisions:
        failures.append("A2.1 source memory is not pristine")
    if raw_source != source_memory.to_dict() or raw_memory != memory.to_dict():
        failures.append("A2.1 memory JSON is not canonical")
    source_hash = sha256_file(source_path)
    if source_hash != result["source"]["d8_memory_sha256"]:
        failures.append("A2.1 source hash changed")
    if result["scene_id"] != source_memory.metadata.get("scene_id"):
        failures.append("A2.1 scene differs from source memory")
    if result["query"] != source_memory.metadata.get("query"):
        failures.append("A2.1 query differs from source memory")
    leaked = (
        _evaluation_keys(result)
        + _evaluation_keys(raw_source)
        + _evaluation_keys(raw_memory)
    )
    if leaked:
        failures.append(
            "evaluation-only data leaked into A2.1 prediction: "
            + ", ".join(leaked)
        )

    expected_memory = ObjectMemory.load(source_path)
    expected_memory.metadata.update(prediction_metadata())
    replay_memory = ObjectMemory.load(source_path)
    replay_memory.metadata.update(prediction_metadata())
    try:
        expected = associate_pending_a21(expected_memory, config)
        replay = associate_pending_a21(replay_memory, config)
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"cannot recompute A2.1 prediction: {error}")
        expected = replay = None

    source_ids = set(source_memory.pending_observations)
    pending_ids, associated_ids = memory_observation_ids(memory)
    acceptance = {
        "observation_conservation": (
            not pending_ids & associated_ids
            and pending_ids | associated_ids == source_ids
            and len(pending_ids) + len(associated_ids) == len(source_ids)
        ),
        "deterministic_recompute": bool(
            expected is not None and replay is not None and expected == replay
            and expected_memory.to_dict() == replay_memory.to_dict()
        ),
        "complete_link_pass": all(
            not row["predicted_same"] or row["gate_pass"]
            for row in result["pairs"]
        ),
        "cross_frame_object_pass": all(
            len(set(item.evidence_frames)) >= config.min_distinct_frames
            for item in memory.objects.values()
        ),
        "round_trip_equal": raw_memory == memory.to_dict(),
    }
    if result["acceptance"] != acceptance:
        failures.append("A2.1 acceptance flags are inconsistent")
    if expected is not None:
        for key in ("pairs", "merge_decisions", "clusters"):
            if result[key] != expected[key]:
                failures.append(f"saved A2.1 {key} differs from recompute")
        if memory.to_dict() != expected_memory.to_dict():
            failures.append("saved A2.1 memory differs from recompute")
        expected_counts = {
            "input_observations": len(source_ids),
            "pair_count": len(expected["pairs"]),
            "gate_pass_pairs": sum(
                bool(row["gate_pass"]) for row in expected["pairs"]
            ),
            "predicted_match_pairs": sum(
                bool(row["predicted_same"]) for row in expected["pairs"]
            ),
            "cluster_count": len(expected["clusters"]),
            "promoted_clusters": sum(
                bool(row["promoted"]) for row in expected["clusters"]
            ),
            "permanent_objects": len(expected_memory.objects),
            "pending_observations": len(expected_memory.pending_observations),
        }
        if result["counts"] != expected_counts:
            failures.append("A2.1 counts are inconsistent")

    return {
        "schema_version": "0.1",
        "status": "PASS" if not failures else "FAIL",
        "stage": "D21.1-A2.1-validation",
        "association_id": A21_ASSOCIATION_ID,
        "scene_id": result.get("scene_id"),
        "query": result.get("query"),
        "counts": result.get("counts"),
        "checks": acceptance,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_dir")
    parser.add_argument("--report")
    args = parser.parse_args()
    report = validate_output(args.prediction_dir)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
