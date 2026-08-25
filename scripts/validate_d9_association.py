"""Independently validate a D9 spatial-association artifact bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

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


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} root must be an object")
    return payload


def resolve_bundle_reference(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError("D9 source paths must be relative")
    bundle_root = root.resolve().parent
    candidate = (root / relative).resolve()
    if candidate != bundle_root and bundle_root not in candidate.parents:
        raise ValueError("D9 source path escapes the artifact bundle")
    return candidate


def validate_output(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    failures: list[str] = []
    result_path = root / "d9_result.json"
    memory_path = root / "object_memory.json"
    labels_path = root / "pair_labels.json"
    run_manifest_path = root / "run_manifest.json"
    for artifact in (
        result_path,
        memory_path,
        labels_path,
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
        raw_memory = read_json(memory_path)
        raw_labels = read_json(labels_path)
        run_manifest = read_json(run_manifest_path)
        memory = ObjectMemory.load(memory_path)
        labels = ManualInstanceLabels.from_dict(raw_labels)
        if set(result) != set(D9_RESULT_FIELDS):
            raise ValueError("D9 result fields are not frozen")
        source = result["source"]
        artifacts = result["artifacts"]
        counts = result["counts"]
        acceptance = result["acceptance"]
        for payload, fields, name in (
            (source, D9_SOURCE_FIELDS, "source"),
            (artifacts, D9_ARTIFACT_FIELDS, "artifacts"),
            (counts, D9_COUNT_FIELDS, "counts"),
            (acceptance, D9_ACCEPTANCE_FIELDS, "acceptance"),
        ):
            if not isinstance(payload, Mapping) or set(payload) != set(fields):
                raise ValueError(f"D9 {name} fields are not frozen")
        config = SpatialGateConfig.from_dict(result["gate_config"])
        source_path = resolve_bundle_reference(
            root, str(source["d8_memory"])
        )
        source_hash = sha256_file(source_path)
        source_memory = ObjectMemory.load(source_path)
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return {
            "status": "FAIL",
            "failures": [f"invalid D9 artifact: {error}"],
        }

    if result["schema_version"] != D9_SCHEMA_VERSION:
        failures.append("unsupported D9 result schema")
    if result["stage"] != "D9":
        failures.append("result stage is not D9")
    if result["baseline_id"] != D9_BASELINE_ID:
        failures.append("unexpected D9 baseline id")
    if raw_labels != labels.to_dict():
        failures.append("D9 pair labels are not canonical")
    if result["scene_id"] != labels.scene_id:
        failures.append("result scene_id differs from pair labels")
    if result["query"] != labels.query:
        failures.append("result query differs from pair labels")
    if source_hash != source["d8_memory_sha256"]:
        failures.append("source D8 memory hash changed")
    if source["pair_labels"] != labels_path.name:
        failures.append("pair label reference is inconsistent")
    labels_hash = sha256_file(labels_path)
    if labels_hash != source["pair_labels_sha256"]:
        failures.append("pair label hash changed")
    if artifacts != {
        "object_memory": memory_path.name,
        "pair_labels": labels_path.name,
    }:
        failures.append("D9 artifact references are inconsistent")
    if source_memory.objects or source_memory.decisions:
        failures.append("D9 source is not a pristine D8 memory")
    if not source_memory.pending_observations:
        failures.append("D9 source has no pending observations")
    if raw_memory != memory.to_dict():
        failures.append("D9 ObjectMemory canonical round-trip changed JSON")

    source_ids = set(source_memory.pending_observations)
    expected_memory = source_memory
    expected_memory.metadata.update({
        "association_source_stage": "D8",
        "association_source_memory": source["d8_memory"],
        "association_source_memory_sha256": source_hash,
        "association_stage": "D9",
        "association_labels": labels_path.name,
        "association_labels_sha256": labels_hash,
    })
    try:
        expected = associate_pending(expected_memory, labels, config)
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"cannot recompute D9 association: {error}")
        expected = None

    if expected is not None:
        if memory.to_dict() != expected_memory.to_dict():
            failures.append(
                "saved ObjectMemory differs from recomputed D9 association"
            )
        for key in ("metrics", "components", "pairs", "failure_cases"):
            if result[key] != expected[key]:
                failures.append(
                    f"saved {key} differs from recomputed D9 result"
                )

        expected_counts = {
            "input_observations": (
                len(expected_memory.pending_observations)
                + sum(
                    len(item.observations)
                    for item in expected_memory.objects.values()
                )
            ),
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
            failures.append("D9 counts are inconsistent")

        min_pairwise_f1 = float(acceptance["min_pairwise_f1"])
        pairwise_f1_pass = (
            expected["metrics"]["f1"] >= min_pairwise_f1
        )
        cross_frame_object_pass = bool(
            expected_memory.objects
        ) and all(
            len(item.evidence_frames) >= config.min_distinct_frames
            for item in expected_memory.objects.values()
        )
        expected_acceptance = {
            "min_pairwise_f1": min_pairwise_f1,
            "pairwise_f1_pass": pairwise_f1_pass,
            "cross_frame_object_pass": cross_frame_object_pass,
            "round_trip_equal": raw_memory == memory.to_dict(),
        }
        if acceptance != expected_acceptance:
            failures.append("D9 acceptance flags are inconsistent")
        expected_status = (
            "PASS"
            if pairwise_f1_pass
            and cross_frame_object_pass
            and expected_acceptance["round_trip_equal"]
            else "FAIL"
        )
        if result["status"] != expected_status:
            failures.append("D9 status is inconsistent with acceptance")

    pending_ids = set(memory.pending_observations)
    associated_ids = {
        observation.obs_id
        for item in memory.objects.values()
        for observation in item.observations
    }
    if pending_ids & associated_ids:
        failures.append("pending and associated observation ids overlap")
    if pending_ids | associated_ids != source_ids:
        failures.append("D9 lost or invented observation ids")
    if any(
        len(item.evidence_frames) < config.min_distinct_frames
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
            "source_memory": source["d8_memory"],
            "pipeline": D9_BASELINE_ID,
            "source_memory_sha256": source_hash,
            "pair_labels": labels_path.name,
            "pair_labels_sha256": labels_hash,
            "gate_config": config.to_dict(),
            "min_pairwise_f1": acceptance["min_pairwise_f1"],
        }
        for key, expected_value in expected_manifest_values.items():
            if manifest_config.get(key) != expected_value:
                failures.append(
                    f"run manifest {key} is inconsistent"
                )
    if run_manifest.get("peak_vram_mb") is not None:
        failures.append("model-free D9 unexpectedly records GPU memory")

    metrics = result.get("metrics", {})
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "scene_id": result.get("scene_id"),
        "query": result.get("query"),
        "input_observations": counts.get("input_observations"),
        "pair_count": counts.get("pair_count"),
        "pairwise_precision": metrics.get("precision"),
        "pairwise_recall": metrics.get("recall"),
        "pairwise_f1": metrics.get("f1"),
        "failure_case_count": len(result.get("failure_cases", [])),
        "candidate_components": counts.get("candidate_components"),
        "permanent_objects": len(memory.objects),
        "pending_observations": len(memory.pending_observations),
        "association_decisions": len(memory.decisions),
        "object_frame_support": {
            object_id: item.evidence_frames
            for object_id, item in memory.objects.items()
        },
        "round_trip_equal": raw_memory == memory.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    args = parser.parse_args()
    report = validate_output(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
