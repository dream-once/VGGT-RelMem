"""Independently validate a label-free D9 prediction bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from scripts.run_d9_association import (
    memory_observation_ids,
    prediction_metadata,
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} root must be an object")
    return payload


def resolve_bundle_reference(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError("D9 source paths must be relative")
    boundary = root.resolve()
    candidate = (boundary / relative).resolve()
    if candidate != boundary and boundary not in candidate.parents:
        raise ValueError("D9 source path escapes the artifact bundle")
    return candidate


def _forbidden_label_keys(
    payload: Any,
    *,
    prefix: str = "",
) -> list[str]:
    found: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            name = str(key)
            path = f"{prefix}.{name}" if prefix else name
            if "label" in name.lower():
                found.append(path)
            found.extend(_forbidden_label_keys(value, prefix=path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(
                _forbidden_label_keys(value, prefix=f"{prefix}[{index}]")
            )
    return found


def validate_output(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    failures: list[str] = []
    result_path = root / "d9_result.json"
    source_memory_path = root / "source_memory.json"
    memory_path = root / "object_memory.json"
    run_manifest_path = root / "run_manifest.json"
    for artifact in (
        result_path,
        source_memory_path,
        memory_path,
        run_manifest_path,
    ):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            failures.append(
                f"missing or empty artifact: {artifact.name}"
            )
    if failures:
        return {"status": "FAIL", "failures": failures}

    try:
        result = read_json(result_path)
        raw_source_memory = read_json(source_memory_path)
        raw_memory = read_json(memory_path)
        run_manifest = read_json(run_manifest_path)
        source_memory = ObjectMemory.load(source_memory_path)
        memory = ObjectMemory.load(memory_path)
        if set(result) != set(D9_PREDICTION_RESULT_FIELDS):
            raise ValueError("D9 prediction result fields are not frozen")
        source = result["source"]
        artifacts = result["artifacts"]
        counts = result["counts"]
        acceptance = result["acceptance"]
        for payload, fields, name in (
            (source, D9_PREDICTION_SOURCE_FIELDS, "source"),
            (
                artifacts,
                D9_PREDICTION_ARTIFACT_FIELDS,
                "artifacts",
            ),
            (counts, D9_COUNT_FIELDS, "counts"),
            (
                acceptance,
                D9_PREDICTION_ACCEPTANCE_FIELDS,
                "acceptance",
            ),
        ):
            if not isinstance(payload, Mapping) or set(payload) != set(fields):
                raise ValueError(
                    f"D9 prediction {name} fields are not frozen"
                )
        config = SpatialGateConfig.from_dict(result["gate_config"])
        referenced_source_path = resolve_bundle_reference(
            root,
            str(source["d8_memory"]),
        )
        referenced_artifact_source = resolve_bundle_reference(
            root,
            str(artifacts["source_memory"]),
        )
        referenced_memory_path = resolve_bundle_reference(
            root,
            str(artifacts["object_memory"]),
        )
        source_hash = sha256_file(referenced_source_path)
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return {
            "status": "FAIL",
            "failures": [f"invalid D9 prediction artifact: {error}"],
        }

    if result["schema_version"] != D9_PREDICTION_SCHEMA_VERSION:
        failures.append("unsupported D9 prediction schema")
    if result["stage"] != "D9-prediction":
        failures.append("result stage is not D9-prediction")
    if result["baseline_id"] != D9_BASELINE_ID:
        failures.append("unexpected D9 baseline id")
    if artifacts != {
        "source_memory": source_memory_path.name,
        "object_memory": memory_path.name,
    }:
        failures.append("D9 prediction artifact references are inconsistent")
    if source["d8_memory"] != artifacts["source_memory"]:
        failures.append("D9 source and artifact memory references differ")
    if referenced_source_path != source_memory_path.resolve():
        failures.append("D9 source reference is not source_memory.json")
    if referenced_artifact_source != source_memory_path.resolve():
        failures.append("D9 source artifact reference is inconsistent")
    if referenced_memory_path != memory_path.resolve():
        failures.append("D9 output memory reference is inconsistent")
    if source_hash != source["d8_memory_sha256"]:
        failures.append("source D8 memory hash changed")
    if raw_source_memory != source_memory.to_dict():
        failures.append("source ObjectMemory is not canonical")
    if raw_memory != memory.to_dict():
        failures.append("D9 ObjectMemory canonical round-trip changed JSON")
    if source_memory.objects or source_memory.decisions:
        failures.append("D9 source is not a pristine D8 memory")
    if not source_memory.pending_observations:
        failures.append("D9 source has no pending observations")

    scene_id = source_memory.metadata.get("scene_id")
    query = source_memory.metadata.get("query")
    if result["scene_id"] != scene_id:
        failures.append("result scene_id differs from source memory")
    if result["query"] != query:
        failures.append("result query differs from source memory")

    forbidden_keys = []
    for name, payload in (
        ("d9_result", result),
        ("source_memory", raw_source_memory),
        ("object_memory", raw_memory),
        ("run_manifest", run_manifest),
    ):
        forbidden_keys.extend(
            f"{name}.{item}" for item in _forbidden_label_keys(payload)
        )
    if forbidden_keys:
        failures.append(
            "prediction bundle contains label-bearing keys: "
            + ", ".join(forbidden_keys)
        )

    metadata_update = prediction_metadata(
        source_reference=str(source["d8_memory"]),
        source_hash=source_hash,
    )
    expected_memory = ObjectMemory.load(source_memory_path)
    expected_memory.metadata.update(metadata_update)
    try:
        expected = associate_pending(expected_memory, config)
        replay_memory = ObjectMemory.load(source_memory_path)
        replay_memory.metadata.update(metadata_update)
        replay = associate_pending(replay_memory, config)
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"cannot recompute D9 prediction: {error}")
        expected = None
        replay = None

    source_ids = set(source_memory.pending_observations)
    pending_ids, associated_ids = memory_observation_ids(memory)
    observation_conservation = (
        not (pending_ids & associated_ids)
        and pending_ids | associated_ids == source_ids
        and len(pending_ids) + len(associated_ids) == len(source_ids)
    )
    cross_frame_object_pass = all(
        len(set(item.evidence_frames)) >= config.min_distinct_frames
        for item in memory.objects.values()
    )
    round_trip_equal = raw_memory == memory.to_dict()

    if expected is not None and replay is not None:
        deterministic_recompute = (
            expected == replay
            and expected_memory.to_dict() == replay_memory.to_dict()
        )
        if result["pairs"] != expected["pairs"]:
            failures.append(
                "saved pairs differs from recomputed D9 prediction"
            )
        if result["components"] != expected["components"]:
            failures.append(
                "saved components differs from recomputed D9 prediction"
            )
        if memory.to_dict() != expected_memory.to_dict():
            failures.append(
                "saved ObjectMemory differs from recomputed D9 prediction"
            )
        expected_counts = {
            "input_observations": len(source_ids),
            "pair_count": len(expected["pairs"]),
            "predicted_match_pairs": sum(
                bool(item["predicted_same"])
                for item in expected["pairs"]
            ),
            "candidate_components": len(expected["components"]),
            "promoted_components": sum(
                bool(item["promoted"])
                for item in expected["components"]
            ),
            "deferred_components": sum(
                not bool(item["promoted"])
                for item in expected["components"]
            ),
            "permanent_objects": len(expected_memory.objects),
            "pending_observations": len(
                expected_memory.pending_observations
            ),
            "association_decisions": len(expected_memory.decisions),
        }
        if counts != expected_counts:
            failures.append("D9 prediction counts are inconsistent")
    else:
        deterministic_recompute = False

    expected_acceptance = {
        "observation_conservation": observation_conservation,
        "deterministic_recompute": deterministic_recompute,
        "cross_frame_object_pass": cross_frame_object_pass,
        "round_trip_equal": round_trip_equal,
    }
    if acceptance != expected_acceptance:
        failures.append("D9 prediction acceptance flags are inconsistent")
    expected_status = (
        "PASS" if all(expected_acceptance.values()) else "FAIL"
    )
    if result["status"] != expected_status:
        failures.append(
            "D9 prediction status is inconsistent with acceptance"
        )

    if not observation_conservation:
        failures.append("D9 lost, duplicated, or invented observation ids")
    if any(
        len(set(item.evidence_frames)) < config.min_distinct_frames
        for item in memory.objects.values()
    ):
        failures.append("a permanent object lacks cross-frame support")
    if len(memory.decisions) != len(associated_ids):
        failures.append("association decision count differs from evidence")

    manifest_config = run_manifest.get("config", {})
    if not isinstance(manifest_config, Mapping):
        failures.append("run manifest config is not an object")
    else:
        expected_manifest_values = {
            "stage": "D9-prediction",
            "pipeline": D9_BASELINE_ID,
            "source_memory": source["d8_memory"],
            "source_memory_sha256": source_hash,
            "gate_config": config.to_dict(),
        }
        for key, expected_value in expected_manifest_values.items():
            if manifest_config.get(key) != expected_value:
                failures.append(
                    f"run manifest {key} is inconsistent"
                )
    if run_manifest.get("dataset_split") != result["scene_id"]:
        failures.append("run manifest dataset split is inconsistent")
    if run_manifest.get("peak_vram_mb") is not None:
        failures.append("model-free D9 unexpectedly records GPU memory")

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "scene_id": result.get("scene_id"),
        "query": result.get("query"),
        "input_observations": counts.get("input_observations"),
        "pair_count": counts.get("pair_count"),
        "predicted_match_pairs": counts.get("predicted_match_pairs"),
        "candidate_components": counts.get("candidate_components"),
        "permanent_objects": len(memory.objects),
        "pending_observations": len(memory.pending_observations),
        "association_decisions": len(memory.decisions),
        "object_frame_support": {
            object_id: item.evidence_frames
            for object_id, item in memory.objects.items()
        },
        "observation_conservation": observation_conservation,
        "deterministic_recompute": deterministic_recompute,
        "round_trip_equal": round_trip_equal,
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
