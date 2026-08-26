"""Independently replay and validate a D12 A2 prediction bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from relground.a2_association import (
    A2_ACCEPTANCE_FIELDS,
    A2_ASSOCIATION_ID,
    A2_COUNT_FIELDS,
    A2_PREDICTION_ARTIFACT_FIELDS,
    A2_PREDICTION_RESULT_FIELDS,
    A2_PREDICTION_SOURCE_FIELDS,
    A2_SCHEMA_VERSION,
    EvidenceAssociationConfig,
    EvidencePair,
    associate_pending_a2,
)
from relground.association import ObjectMemory
from relground.observation_cache import sha256_file
from scripts.run_a2_association import prediction_metadata
from scripts.run_d9_association import memory_observation_ids


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} root must be an object")
    return payload


def resolve_bundle_reference(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError("A2 artifact paths must be relative")
    boundary = root.resolve()
    candidate = (boundary / relative).resolve()
    if candidate != boundary and boundary not in candidate.parents:
        raise ValueError("A2 artifact path escapes the prediction bundle")
    return candidate


def _label_keys(payload: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            name = str(key)
            path = f"{prefix}.{name}" if prefix else name
            if (
                "label" in name.lower()
                or name in {"expected_same", "metrics", "error_type"}
            ):
                found.append(path)
            found.extend(_label_keys(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_label_keys(value, f"{prefix}[{index}]"))
    return found


def validate_output(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    failures: list[str] = []
    required = (
        "a2_result.json",
        "source_memory.json",
        "object_memory.json",
        "run_manifest.json",
    )
    for name in required:
        artifact = root / name
        if not artifact.is_file() or artifact.stat().st_size == 0:
            failures.append(f"missing or empty artifact: {name}")
    if failures:
        return {"status": "FAIL", "failures": failures}

    try:
        result = read_json(root / "a2_result.json")
        raw_source = read_json(root / "source_memory.json")
        raw_memory = read_json(root / "object_memory.json")
        manifest = read_json(root / "run_manifest.json")
        if set(result) != set(A2_PREDICTION_RESULT_FIELDS):
            raise ValueError("A2 prediction fields are not frozen")
        source = result["source"]
        artifacts = result["artifacts"]
        counts = result["counts"]
        acceptance = result["acceptance"]
        for payload, fields, name in (
            (source, A2_PREDICTION_SOURCE_FIELDS, "source"),
            (artifacts, A2_PREDICTION_ARTIFACT_FIELDS, "artifacts"),
            (counts, A2_COUNT_FIELDS, "counts"),
            (acceptance, A2_ACCEPTANCE_FIELDS, "acceptance"),
        ):
            if not isinstance(payload, Mapping) or set(payload) != set(fields):
                raise ValueError(f"A2 {name} fields are not frozen")
        config = EvidenceAssociationConfig.from_dict(result["config"])
        for item in result["pairs"]:
            EvidencePair.from_dict(item)
        source_path = resolve_bundle_reference(
            root, str(source["d8_memory"])
        )
        artifact_source = resolve_bundle_reference(
            root, str(artifacts["source_memory"])
        )
        memory_path = resolve_bundle_reference(
            root, str(artifacts["object_memory"])
        )
        if source_path != (root / "source_memory.json"):
            raise ValueError("A2 source reference is inconsistent")
        if artifact_source != source_path:
            raise ValueError("A2 source artifact reference is inconsistent")
        if memory_path != (root / "object_memory.json"):
            raise ValueError("A2 memory artifact reference is inconsistent")
        source_memory = ObjectMemory.load(source_path)
        memory = ObjectMemory.load(memory_path)
        source_hash = sha256_file(source_path)
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return {
            "status": "FAIL",
            "failures": [f"invalid A2 prediction artifact: {error}"],
        }

    if result["schema_version"] != A2_SCHEMA_VERSION:
        failures.append("unsupported A2 prediction schema")
    if result["stage"] != "D12-A2-prediction":
        failures.append("unexpected A2 prediction stage")
    if result["association_id"] != A2_ASSOCIATION_ID:
        failures.append("unexpected A2 association id")
    if result["status"] != "PASS":
        failures.append("A2 prediction result is not PASS")
    if source_hash != source["d8_memory_sha256"]:
        failures.append("A2 source memory hash changed")
    if raw_source != source_memory.to_dict():
        failures.append("A2 source memory is not canonical")
    if raw_memory != memory.to_dict():
        failures.append("A2 output memory is not canonical")
    if source_memory.objects or source_memory.decisions:
        failures.append("A2 source memory is not pristine")
    forbidden = []
    for name, payload in (
        ("result", result),
        ("source", raw_source),
        ("memory", raw_memory),
        ("manifest", manifest),
    ):
        forbidden.extend(f"{name}.{item}" for item in _label_keys(payload))
    if forbidden:
        failures.append(
            "A2 prediction contains evaluation-only fields: "
            + ", ".join(forbidden)
        )

    metadata = prediction_metadata(
        source_reference=source_path.name,
        source_hash=source_hash,
    )
    expected_memory = ObjectMemory.load(source_path)
    expected_memory.metadata.update(metadata)
    replay_memory = ObjectMemory.load(source_path)
    replay_memory.metadata.update(metadata)
    try:
        expected = associate_pending_a2(expected_memory, config)
        replay = associate_pending_a2(replay_memory, config)
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"cannot recompute A2 prediction: {error}")
        expected = replay = None

    source_ids = set(source_memory.pending_observations)
    pending_ids, associated_ids = memory_observation_ids(memory)
    observation_conservation = (
        not pending_ids & associated_ids
        and pending_ids | associated_ids == source_ids
        and len(pending_ids) + len(associated_ids) == len(source_ids)
    )
    cross_frame = all(
        len(set(item.evidence_frames)) >= config.min_distinct_frames
        for item in memory.objects.values()
    )
    complete_link = all(
        not item["predicted_same"] or item["gate_pass"]
        for item in result["pairs"]
    )
    deterministic = bool(
        expected is not None
        and replay is not None
        and expected == replay
        and expected_memory.to_dict() == replay_memory.to_dict()
    )
    round_trip = raw_memory == memory.to_dict()
    expected_acceptance = {
        "observation_conservation": observation_conservation,
        "deterministic_recompute": deterministic,
        "complete_link_pass": complete_link,
        "cross_frame_object_pass": cross_frame,
        "round_trip_equal": round_trip,
    }
    if acceptance != expected_acceptance:
        failures.append("A2 acceptance flags are inconsistent")

    if expected is not None:
        if result["pairs"] != expected["pairs"]:
            failures.append("saved A2 pairs differ from recompute")
        if result["merge_decisions"] != expected["merge_decisions"]:
            failures.append("saved A2 merge decisions differ from recompute")
        if result["clusters"] != expected["clusters"]:
            failures.append("saved A2 clusters differ from recompute")
        if memory.to_dict() != expected_memory.to_dict():
            failures.append("saved A2 memory differs from recompute")
        clusters = expected["clusters"]
        expected_counts = {
            "input_observations": len(source_ids),
            "pair_count": len(expected["pairs"]),
            "gate_pass_pairs": sum(
                bool(item["gate_pass"]) for item in expected["pairs"]
            ),
            "predicted_match_pairs": sum(
                bool(item["predicted_same"]) for item in expected["pairs"]
            ),
            "merge_count": len(expected["merge_decisions"]),
            "cluster_count": len(clusters),
            "promoted_clusters": sum(
                bool(item["promoted"]) for item in clusters
            ),
            "deferred_clusters": sum(
                not bool(item["promoted"]) for item in clusters
            ),
            "permanent_objects": len(expected_memory.objects),
            "pending_observations": len(expected_memory.pending_observations),
            "association_decisions": len(expected_memory.decisions),
        }
        if counts != expected_counts:
            failures.append("A2 counts are inconsistent")

    config_manifest = manifest.get("config", {})
    if not isinstance(config_manifest, Mapping):
        failures.append("A2 run manifest config is not an object")
    else:
        expected_manifest = {
            "stage": "D12-A2-prediction",
            "status": "CPU_COMPLETE",
            "gpu_acceptance": "PENDING",
            "association_id": A2_ASSOCIATION_ID,
            "source_memory": source_path.name,
            "source_memory_sha256": source_hash,
            "association_config": config.to_dict(),
        }
        for key, value in expected_manifest.items():
            if config_manifest.get(key) != value:
                failures.append(f"A2 run manifest {key} is inconsistent")
    if manifest.get("peak_vram_mb") is not None:
        failures.append("CPU-only A2 prediction records GPU memory")

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "stage": "D12-A2-prediction",
        "association_id": A2_ASSOCIATION_ID,
        "scene_id": result.get("scene_id"),
        "query": result.get("query"),
        **dict(counts),
        **expected_acceptance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--report")
    args = parser.parse_args()
    report = validate_output(args.output_dir)
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
