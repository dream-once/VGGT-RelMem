"""Independently validate the separate labelled D12 A2 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from relground.a2_association import (
    A2_ASSOCIATION_ID,
    A2_EVALUATION_ACCEPTANCE_FIELDS,
    A2_EVALUATION_ARTIFACT_FIELDS,
    A2_EVALUATION_RESULT_FIELDS,
    A2_EVALUATION_SCHEMA_VERSION,
    A2_EVALUATION_SOURCE_FIELDS,
    A2_METRIC_FIELDS,
    EvidenceAssociationConfig,
    evaluate_a2_predictions,
)
from relground.association import ObjectMemory
from relground.d9_association import ManualInstanceLabels
from relground.observation_cache import sha256_file
from scripts.validate_a2_association import (
    read_json,
    resolve_bundle_reference,
    validate_output as validate_prediction_output,
)


def resolve_evaluation_reference(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError("A2 evaluation references must be relative")
    boundary = root.resolve().parent
    candidate = (root / relative).resolve()
    if candidate != boundary and boundary not in candidate.parents:
        raise ValueError("A2 evaluation reference escapes the daily bundle")
    return candidate


def validate_output(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    failures: list[str] = []
    for name in ("a2_evaluation.json", "pair_labels.json", "run_manifest.json"):
        artifact = root / name
        if not artifact.is_file() or artifact.stat().st_size == 0:
            failures.append(f"missing or empty artifact: {name}")
    if failures:
        return {"status": "FAIL", "failures": failures}

    prediction_report: dict[str, Any] = {
        "status": "FAIL",
        "failures": ["prediction was not resolved"],
    }
    try:
        result = read_json(root / "a2_evaluation.json")
        raw_labels = read_json(root / "pair_labels.json")
        manifest = read_json(root / "run_manifest.json")
        if set(result) != set(A2_EVALUATION_RESULT_FIELDS):
            raise ValueError("A2 evaluation fields are not frozen")
        source = result["source"]
        artifacts = result["artifacts"]
        acceptance = result["acceptance"]
        metrics = result["metrics"]
        for payload, fields, name in (
            (source, A2_EVALUATION_SOURCE_FIELDS, "source"),
            (artifacts, A2_EVALUATION_ARTIFACT_FIELDS, "artifacts"),
            (acceptance, A2_EVALUATION_ACCEPTANCE_FIELDS, "acceptance"),
            (metrics, A2_METRIC_FIELDS, "metrics"),
        ):
            if not isinstance(payload, Mapping) or set(payload) != set(fields):
                raise ValueError(f"A2 evaluation {name} fields are not frozen")
        prediction_path = resolve_evaluation_reference(
            root, str(source["prediction_result"])
        )
        labels_path = resolve_evaluation_reference(
            root, str(source["pair_labels"])
        )
        artifact_labels = resolve_evaluation_reference(
            root, str(artifacts["pair_labels"])
        )
        if labels_path != (root / "pair_labels.json"):
            raise ValueError("A2 label source reference is inconsistent")
        if artifact_labels != labels_path:
            raise ValueError("A2 label artifact reference is inconsistent")
        prediction = read_json(prediction_path)
        prediction_dir = prediction_path.parent
        prediction_report = validate_prediction_output(prediction_dir)
        prediction_hash = sha256_file(prediction_path)
        labels_hash = sha256_file(labels_path)
        labels = ManualInstanceLabels.from_dict(raw_labels)
        source_memory_path = resolve_bundle_reference(
            prediction_dir,
            str(prediction["artifacts"]["source_memory"]),
        )
        source_memory = ObjectMemory.load(source_memory_path)
        expected = evaluate_a2_predictions(
            list(source_memory.pending_observations.values()),
            prediction["pairs"],
            labels,
        )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return {
            "status": "FAIL",
            "failures": [f"invalid A2 evaluation artifact: {error}"],
        }

    if result["schema_version"] != A2_EVALUATION_SCHEMA_VERSION:
        failures.append("unsupported A2 evaluation schema")
    if result["stage"] != "D12-A2-evaluation":
        failures.append("unexpected A2 evaluation stage")
    if result["association_id"] != A2_ASSOCIATION_ID:
        failures.append("unexpected A2 evaluation association id")
    if prediction_report["status"] != "PASS":
        failures.append("referenced A2 prediction does not pass validation")
    if prediction_hash != source["prediction_result_sha256"]:
        failures.append("A2 prediction hash changed")
    if labels_hash != source["pair_labels_sha256"]:
        failures.append("A2 labels hash changed")
    if prediction.get("scene_id") != labels.scene_id:
        failures.append("A2 prediction and label scene differ")
    if prediction.get("query") != labels.query:
        failures.append("A2 prediction and label query differ")
    if result["scene_id"] != labels.scene_id or result["query"] != labels.query:
        failures.append("A2 evaluation scene/query differ from labels")
    if result["pairs"] != expected["pairs"]:
        failures.append("saved A2 evaluated pairs differ from recompute")
    if metrics != expected["metrics"]:
        failures.append("saved A2 metrics differ from recompute")
    if result["failure_cases"] != expected["failure_cases"]:
        failures.append("saved A2 failures differ from recompute")

    frozen = (
        EvidenceAssociationConfig.from_dict(prediction["config"]).to_dict()
        == EvidenceAssociationConfig().to_dict()
    )
    expected_acceptance = {
        "prediction_valid": prediction_report["status"] == "PASS",
        "evaluation_recomputed": True,
        "thresholds_frozen_before_evaluation": frozen,
    }
    if acceptance != expected_acceptance:
        failures.append("A2 evaluation acceptance flags are inconsistent")
    expected_status = "PASS" if all(expected_acceptance.values()) else "FAIL"
    if result["status"] != expected_status:
        failures.append("A2 evaluation status is inconsistent")

    manifest_config = manifest.get("config", {})
    if not isinstance(manifest_config, Mapping):
        failures.append("A2 evaluation manifest config is not an object")
    else:
        expected_manifest = {
            "stage": "D12-A2-evaluation",
            "status": "CPU_COMPLETE",
            "gpu_acceptance": "PENDING",
            "association_id": A2_ASSOCIATION_ID,
            "prediction_result": source["prediction_result"],
            "prediction_result_sha256": prediction_hash,
            "pair_labels": "pair_labels.json",
            "pair_labels_sha256": labels_hash,
            "thresholds_frozen_before_evaluation": frozen,
        }
        for key, value in expected_manifest.items():
            if manifest_config.get(key) != value:
                failures.append(
                    f"A2 evaluation manifest {key} is inconsistent"
                )
    if manifest.get("peak_vram_mb") is not None:
        failures.append("CPU-only A2 evaluation records GPU memory")

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "stage": "D12-A2-evaluation",
        "association_id": A2_ASSOCIATION_ID,
        "scene_id": result.get("scene_id"),
        "query": result.get("query"),
        **dict(metrics),
        "failure_case_count": len(result["failure_cases"]),
        "prediction_status": prediction_report["status"],
        "thresholds_frozen_before_evaluation": frozen,
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
